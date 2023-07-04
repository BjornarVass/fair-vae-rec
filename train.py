import numpy as np
import wandb
import argparse

import torch
import torch.utils.data
from torch import optim
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from data_processing import init_dataloader, preprocess_movielens, preprocess_lastfm, load_and_uncompress
from model import FairVAERec, VAESettings, RunSettings, EvalStorage, EvalResults, BinaryClassifier
from utils import (
    ndcg_at_k,
    generate_network_dims,
    chi_square_rec_k,
    kendall_tau_rec,
    get_aggregated_item_ranks,
)


def parse_arguments():
    bool_t = "store_true"
    parser = argparse.ArgumentParser(description="Process arguments")
    # RUN
    parser.add_argument("--run_group", type=str, default="RUN", help="Specify wandb run group name")
    parser.add_argument("--dataset", type=str, default="movielens", help="Options: 'movielens' and 'lastfm'")
    parser.add_argument("--verbose", default=False, action=bool_t, help="Enable logging")
    parser.add_argument("--n_adv_train", type=int, default=1, help="# adversarial updates per minibatch")
    parser.add_argument("--n_epochs", type=int, default=1, help="# training epochs")
    parser.add_argument("--n_adv_pre", type=int, default=100, help="# adversarial pre-train epochs")
    parser.add_argument("--tr_batch_size", type=int, default=100, help="Training batch size")
    parser.add_argument("--nbatch_per_update", type=int, default=1, help="# batches per model updates")
    parser.add_argument("--te_batch_size", type=int, default=1, help="Testing batch size")
    parser.add_argument("--train_eval", default=False, action=bool_t, help="Enable evaluation during training")
    parser.add_argument(
        "--partial", default=False, action=bool_t, help="Enable evaluation on each minibatch to save memory"
    )
    parser.add_argument(
        "--rep_eval_interval", type=int, default=40, help="# epochs between evaluating Representation Neutrality"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Name of the device used for computation")

    # PREPROCESSING AND DATA
    # Note: Integrated preprocessing has limited support and assumes standard filenames.
    parser.add_argument("--preprocess", default=False, action=bool_t, help="Enable integrated data preprocessing")
    parser.add_argument("--data_dir", type=str, default="data/", help="PREPROCESSING: Path to data directory")
    parser.add_argument("--item_split", default=False, action=bool_t, help="Split data on items, not users")
    parser.add_argument(
        "--imbalanced_sensitive", default=False, action=bool_t, help="Disable demographic data stratification"
    )
    parser.add_argument(
        "--valtest_frac", type=float, default=0.2, help="Fraction of users/items in valtest. Valtest is split 50/50"
    )
    # Note: Assumes processed file names
    parser.add_argument("--processed_dir", type=str, default="pro_lf/0/", help="Path to processsed data")
    parser.add_argument("--csr", default=False, action=bool_t, help="Load data as csr to reduce memory impact")

    # MODEL STRUCTURE
    parser.add_argument("--hidden_activation", type=str, default="selu", help="Options: 'selu', 'tanh' and 'relu'")
    parser.add_argument("--ganKL", default=False, action=bool_t, help="Enable gan KL mode. Requires alpha and gamma")
    parser.add_argument("--hidden_dim", type=int, default=500, help="Hidden dimensionality of encoder and decoder")
    parser.add_argument(
        "--hidden_dim_b", type=int, default=250, help="Hidden dimensionality of additional hidden layers in b encoder"
    )
    parser.add_argument("--hidden_dim_adv", type=int, default=12, help="Hidden dimensionality of adversarial models")
    parser.add_argument("--n_hidden_vae", type=int, default=1, help="# hidden layers in VAE")
    parser.add_argument("--n_hidden_b_enc", type=int, default=2, help="# hidden layers in b encoder")
    parser.add_argument("--n_hidden_b_dec", type=int, default=1, help="# hidden layers in b decoder")
    parser.add_argument("--n_hidden_adv", type=int, default=1, help="# hidden layers in adversarial models")
    parser.add_argument("--z_dim", type=int, default=64, help="Dimension of latent state/z")
    parser.add_argument("--b_dim", type=int, default=0, help="Dimension of b. Requires alpha and gamma")
    parser.add_argument("--dropout_vae", type=float, default=0.2, help="VAE dropout rate")
    parser.add_argument("--dropout_adv", type=float, default=0.0, help="Adversarial dropout rate")
    parser.add_argument("--noise_rate", type=float, default=0.5, help="Fraction of items dropped during training")

    # MODEL OPTIMIZATION
    parser.add_argument("--lr", type=float, default=0.001, help="Optimizer learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Optimizer weight decay")
    parser.add_argument("--beta", type=float, default=1.0, help="Loss parameter beta")
    parser.add_argument("--alpha", type=float, default=0.0, help="Loss parameter alpha. Used by VAEgan and VAEemp")
    parser.add_argument("--gamma", type=float, default=0.0, help="Loss parameter gamma. Used by VAEgan and VAEemp")
    parser.add_argument("--delta", type=float, default=0.0, help="Loss parameter delta. Used by VAEadv")

    # EVALUATION
    parser.add_argument("--k", type=int, default=10, help="# of ranks considered in NDCG metric")
    parser.add_argument(
        "--indv_k", type=int, default=100, help="# of user ranks considered in Recommendation Parity metrics"
    )
    parser.add_argument("--agg_k", type=int, default=100, help="# of aggregated ranks in Kendall Tau metric")
    parser.add_argument("--n_chi2_items", type=int, default=10, help="# of items considered in Chi squared metric")

    args = parser.parse_args()
    run(args)


def run(args):

    # Pre-process datasets
    user_split = not args.item_split
    if args.preprocess:
        preprocess_datasets(args, user_split)

    # Load datasets
    train_data, val_tr, val_te, test_tr, test_te, sensitive_labels, train_s, val_s, test_s = load_and_uncompress(
        args.processed_dir, user_split=user_split, csr=args.csr
    )

    # Define network structures based processed data and hyperparameters
    n_sensitive = len(sensitive_labels)
    n_items = train_data.shape[1]
    discriminate = args.ganKL or args.delta > 0
    z_enc_dims = generate_network_dims(n_items, args.hidden_dim, 2 * args.z_dim, n_hidden=args.n_hidden_vae)
    z_dec_dims = generate_network_dims(args.z_dim, args.hidden_dim, n_items, n_hidden=args.n_hidden_vae)

    b_enc_dims = generate_network_dims(
        n_items, args.hidden_dim, 2 * args.b_dim, n_hidden=args.n_hidden_b_enc, red_hidden_dim=args.hidden_dim_b
    )
    b_dec_dims = generate_network_dims(args.b_dim, args.b_dim, n_sensitive, n_hidden=args.n_hidden_b_dec)

    adv_dims = (
        generate_network_dims(args.z_dim + args.b_dim, args.hidden_dim_adv, 1, n_hidden=args.n_hidden_adv)
        if discriminate
        else []
    )

    # Settings
    vae_settings = VAESettings(
        z_enc_dims=z_enc_dims,
        z_dec_dims=z_dec_dims,
        b_enc_dims=b_enc_dims,
        b_dec_dims=b_dec_dims,
        adv_dims=adv_dims,
        noise_rate=0.5,
        dropout_vae=args.dropout_vae,
        dropout_adv=args.dropout_adv,
        beta=args.beta,
        alpha=args.alpha,
        gamma=args.gamma,
        delta=args.delta,
        ganKL=args.ganKL,
        n_sensitive=n_sensitive,
        sensitive_labels=sensitive_labels,
        b_dim=args.b_dim,
        hidden_activation=args.hidden_activation,
        device=args.device,
        verbose=args.verbose,
    )

    # Init model and optimizers
    vae, opt_vae, opt_adv = init_model_and_optimizers(vae_settings, discriminate, args.weight_decay)

    # Init data loaders
    if user_split:
        train_loader = init_dataloader(train_data, None, train_s, vae.c.device, args.tr_batch_size, args.csr)
        val_loader = init_dataloader(val_tr, val_te, val_s, vae.c.device, args.te_batch_size, args.csr)
        test_loader = init_dataloader(test_tr, test_te, test_s, vae.c.device, args.te_batch_size, args.csr)
    else:
        train_loader = init_dataloader(train_data, val_te, train_s, vae.c.device, args.tr_batch_size, args.csr)
        val_loader = train_loader
        test_loader = init_dataloader(train_data, test_te, train_s, vae.c.device, args.te_batch_size, args.csr)

    # Start logging
    if vae.c.verbose:
        hyperparams = vars(vae_settings)
        hyperparams["dataset"] = args.dataset
        wandb.init(group=args.run_group, config=hyperparams)

    # Collect run settings for model training
    run_settings = RunSettings(
        vae=vae,
        opt_vae=opt_vae,
        opt_adv=opt_adv,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.n_epochs,
        n_adv_pre=args.n_adv_pre,
        nbatch_per_update=args.nbatch_per_update,
        discriminate=discriminate,
        n_adv_train=args.n_adv_train,
        k=args.k,
        agg_k=args.agg_k,
        indv_k=args.indv_k,
        n_chi2_items=args.n_chi2_items,
        train_eval=args.train_eval,
        rep_eval_interval=args.rep_eval_interval,
        partial=args.partial,
        csr=args.csr,
    )

    # Train model
    # Turn off logging during training if train_eval is set to false
    verbose_setting = vae.c.verbose
    if not args.train_eval:
        vae.c.verbose = False
    train_model(run_settings)
    # Revert verbose setting
    vae.c.verbose = verbose_setting

    # Evaluate on test set
    # Set model to eval mode and replace validation loader with the test set
    test_log = {}
    run_settings.vae.eval()
    run_settings.val_loader = test_loader
    evaluate_model(
        run_settings,
        test_log,
        eval_representations=True,
        test=True,
    )

    # Log test results and close log
    if vae.c.verbose:
        wandb.log(test_log)
        wandb.finish()


def preprocess_datasets(args, user_split):
    balance_sensitive = not args.imbalanced_sensitive
    if args.dataset == "movielens":
        ratings_filename = "ratings.dat"
        user_info_filename = "users.dat"
        preprocess_movielens(
            args.data_dir,
            args.processed_dir,
            ratings_filename,
            user_info_filename,
            args.valtest_frac,
            balance_sensitive=balance_sensitive,
            user_split=user_split,
        )

    elif args.dataset == "lastfm":
        ratings_filename = "lastfm.tsv"
        user_info_filename = "users.tsv"
        album_filename = "albums.tsv"
        preprocess_lastfm(
            args.data_dir,
            args.processed_dir,
            ratings_filename,
            user_info_filename,
            album_filename,
            args.valtest_frac,
            balance_sensitive=balance_sensitive,
            user_split=user_split,
        )
    else:
        raise ValueError(f"Unsupported dataset: '{args.dataset}'")


def init_model_and_optimizers(vae_settings, discriminate, weight_decay):
    vae = FairVAERec(vae_settings)
    print(vae)
    vae_params = vae.get_params()
    opt_vae = optim.Adam(vae_params, lr=0.001, weight_decay=weight_decay)

    # Add adversarial optimizer, if defined
    if not discriminate:
        return vae, opt_vae, None

    adv_params = vae.get_params(adversarial=True)

    opt_adv = optim.Adam(adv_params, lr=0.001, weight_decay=weight_decay)

    return vae, opt_vae, opt_adv


def train_model(settings):
    vae = settings.vae
    vae.to(vae.c.device)
    virtual_counter = 1
    for epoch in range(settings.n_epochs):
        # Set periodic flags
        eval_representations = epoch != 0 and epoch % settings.rep_eval_interval == 0

        # Storage for latent reps
        zbs = []
        sensitive_targets = []

        # Log dict for wandb
        epoch_logs = {}

        # Pre train adversarial(s)
        if epoch == 0 and settings.discriminate:
            pre_train_adv(settings, epoch_logs)

        # Batch storage
        batch_storage = EvalStorage()

        # MAIN TRAINNG LOOP
        vae.train()
        for i, (x, _, s) in enumerate(settings.train_loader):
            # Move batch to GPU if Csr is used
            if settings.csr:
                x = x.to(vae.c.device)
                s = s.to(device=vae.c.device, dtype=torch.float32)

            # Train VAE
            if virtual_counter < settings.nbatch_per_update:
                run_step = False
                virtual_counter += 1
            else:
                run_step = True
            zb, _, _ = call_fw_bw(settings, x, s, epoch_logs, step=run_step)

            # Store latent reps and sensitive targets
            if settings.train_eval and eval_representations:
                zbs.append(zb.cpu().detach().numpy())
                sensitive_targets.append(s.cpu().detach().numpy())

            # Train adversarial models
            if settings.discriminate:
                _, _, batch_storage = call_fw_bw(
                    settings,
                    x,
                    s,
                    epoch_logs,
                    step=run_step,
                    discriminate=True,
                    batch_storage=batch_storage,
                )

            # Reset virtual batch
            if run_step:
                virtual_counter = 1
                batch_storage = EvalStorage()

        # EVALUATION
        if settings.train_eval:
            # Set to evaluation mode
            vae.eval()

            # Prepare train reps
            if eval_representations:
                # print(epoch)
                zbs = np.concatenate(zbs)
                sensitive_targets = np.concatenate(sensitive_targets)

            evaluate_model(
                settings,
                epoch_logs,
                eval_representations=eval_representations,
                train_zbs_s=zbs,
                train_s_s=sensitive_targets,
            )

        # Update wand
        if vae.c.verbose:
            wandb.log(epoch_logs)


def pre_train_adv(settings, epoch_logs):
    device = settings.vae.c.device
    for j in range(settings.n_adv_pre):
        batch_storage = EvalStorage()
        virtual_counter = 1
        for i, (x, _, s) in enumerate(settings.train_loader):
            # Move batch to GPU if Csr is used
            if settings.csr:
                x = x.to(device=device)
                s = s.to(device=device, dtype=torch.float32)

            if virtual_counter < settings.nbatch_per_update:
                run_step = False
                virtual_counter += 1
            else:
                run_step = True

            zb, _, batch_storage = call_fw_bw(
                settings,
                x,
                s,
                epoch_logs,
                step=run_step,
                discriminate=True,
                batch_storage=batch_storage,
            )

            # Reset virtual batch
            if run_step:
                virtual_counter = 1
                batch_storage = EvalStorage()
    print("Pre-training Done")


def call_fw_bw(
    settings,
    x,
    s,
    wandb_log,
    backwards=True,
    step=True,
    discriminate=False,
    batch_storage=None,
):
    vae = settings.vae
    # Skip decoding when training adversarials
    decode = not discriminate

    latent, loss, probs = vae(x, s, wandb_log, decode=decode)
    latent = latent.detach()

    # For evaluation purposes
    if not backwards and discriminate and vae.c.verbose:
        if vae.split_latent:
            loss = vae.discriminate(latent, wandb_log)
        else:
            loss = vae.discriminate_simple(latent, s, wandb_log)

    if backwards and discriminate:
        # In case we have multiple virtual batches we have to store latents from different parts
        batch_storage.update(latent, s)
        if step:
            # Multiple iterations of training
            batch_latents = torch.cat(batch_storage.zbs)
            batch_s = torch.cat(batch_storage.s)
            for i in range(settings.n_adv_train):
                if vae.split_latent:
                    loss = vae.discriminate(batch_latents, wandb_log)
                else:
                    loss = vae.discriminate_simple(batch_latents, batch_s, wandb_log)

                settings.opt_adv.zero_grad()
                loss.backward()
                settings.opt_adv.step()
    elif backwards:
        loss.backward()
        if step:
            settings.opt_vae.step()
            settings.opt_vae.zero_grad()

    return latent, probs, batch_storage


def evaluate_model(
    settings,
    wandb_log,
    eval_representations=False,
    train_zbs_s=None,
    train_s_s=None,
    test=False,
):
    vae = settings.vae
    # Store sample setting and set flag to False
    sample_setting = vae.sample
    vae.sample = False
    eval_storage, eval_res = evaluate_model_recs(settings, wandb_log)
    # When called for the final test: also evaluate recommendations with sampled latents
    if test:
        vae.sample = True
        _, eval_res_sampled = evaluate_model_recs(settings, wandb_log)
        vae.sample = False

    # Log recommendation performance
    if vae.c.verbose:
        test_prefix = "test " if test else ""
        log_metrics(eval_res, test_prefix, vae.c.sensitive_labels, wandb_log)
        if test:
            log_metrics(eval_res_sampled, "test_s ", vae.c.sensitive_labels, wandb_log)

    # Representation neutrality evaluation
    # Sampling is set to False at this point. Will be reverted through the sample_setting variable later
    if eval_representations:
        main_prefix = "test" if test else "val"
        if not test:
            evaluate_representation(train_zbs_s, train_s_s, "train_s", vae, wandb_log)
        evaluate_representation(eval_storage.zbs, eval_storage.s, main_prefix + vae.sample * "_s", vae, wandb_log)

        # Disable logging during inference call for new zbs, disable sampling for train reps
        verbose = vae.c.verbose
        vae.c.verbose = False

        # Run inference on train without sampling and validation with sampling
        if not test:
            eval_train, _ = inference(settings, wandb_log, latent_only=True, train_set=True)

        vae.sample = True
        eval_val_s, _ = inference(settings, wandb_log, latent_only=True)

        # Revert logging deactivation
        vae.c.verbose = verbose
        if not test:
            evaluate_representation(eval_train.zbs, eval_train.s, "train", vae, wandb_log)
        evaluate_representation(eval_val_s.zbs, eval_val_s.s, main_prefix + vae.sample * "_s", vae, wandb_log)

    # Revert sampling setting
    vae.sample = sample_setting


def evaluate_model_recs(settings, wandb_log):
    # All probs are stored to perform single evaluation
    if not settings.partial:
        eval_storage, _ = inference(settings, wandb_log)
        eval_res = evaluate_all_recommendations(settings, eval_storage.u_probs, eval_storage.targets, eval_storage.s)
    # NDCG is calculated and chi2 + kendall ranks are aggregated for each minibatch to save memory. NDCG has to be
    # aggregated and Chi2 and kendall has to be calculated using the final aggregated ranks.
    else:
        eval_res = EvalResults(settings.vae.c.n_sensitive)
        eval_storage, eval_res = inference(settings, wandb_log, eval_res=eval_res)
        eval_res.aggregate_parts(eval_storage.s.shape[0])
        chi100 = [chi_square_rec_k(None, None, None, settings.n_chi2_items, ranks) for ranks in eval_storage.chi_ranks]
        kendall100 = [kendall_tau_rec(None, None, None, settings.agg_k, ranks) for ranks in eval_storage.kendall_ranks]
        eval_res.set_results(chi100, kendall100)
    return eval_storage, eval_res


def log_metrics(eval_res, test_prefix, sensitive_labels, wandb_log):
    wandb_log[f"metric/{test_prefix}NDCG"] = eval_res.ndcg
    for i, sensitive_name in enumerate(sensitive_labels):
        metric_logs = {
            f"metric/{test_prefix}{sensitive_name} Chi100": eval_res.chi100[i],
            f"metric/{test_prefix}{sensitive_name} Kendall100": eval_res.kendall100[i],
        }
        wandb_log.update(metric_logs)


def inference(settings, wandb_log, latent_only=False, train_set=False, eval_res=None):
    vae = settings.vae
    # Structures for assembling the correct evaluation data with respect to mini-batching
    eval_storage = EvalStorage()
    loader = settings.train_loader if train_set else settings.val_loader
    for i, (x, y, s) in enumerate(loader):
        # Move batch to GPU if Csr is used
        if settings.csr:
            x = x.to(vae.c.device)
            s = s.to(device=vae.c.device, dtype=torch.float32)

        # Forward pass
        zb, probs, _ = call_fw_bw(settings, x, s, wandb_log, backwards=False)

        inference_update_eval(settings, probs, x, y, zb, s, latent_only, eval_storage, eval_res)

        # Eval adversarials on the first minibatch of each epoch
        if vae.c.ganKL or vae.adv and i == 0:
            call_fw_bw(settings, x, s, wandb_log, backwards=False, discriminate=True)

    # Assemble evaluation
    eval_storage.concat()
    return eval_storage, eval_res


def inference_update_eval(settings, probs, x, y, zb, s, latent_only, eval_storage, eval_res):
    zb_np = zb.cpu().detach().numpy()
    s_np = s.cpu().detach().numpy()

    if latent_only:
        eval_storage.update(zb_np, s_np)
        return

    # Move probs to CPU for evaluation
    probs = probs.cpu().detach().numpy()
    # Ignore items we know the user likes in NDCG scores
    util_probs = np.array(probs)
    x_np = x.cpu().detach().numpy()
    util_probs[x_np.nonzero()] = -np.inf

    if settings.partial:
        eval_res, chi_ranks, kendall_ranks = evaluate_partial_recommendations(
            settings, util_probs, y.detach().numpy(), s_np, eval_res
        )
        eval_storage.update(zb_np, s_np)
        eval_storage.update_ranks(chi_ranks, kendall_ranks)
    else:
        eval_storage.update(zb_np, s_np, probs, util_probs, y.detach().numpy())


# Called once for each epoch (partial=False)
def evaluate_all_recommendations(settings, u_probs, targets, s):
    ndcg = ndcg_at_k(u_probs, targets, settings.k).mean()
    chi100 = []
    kendall100 = []

    for i in range(settings.vae.c.n_sensitive):
        chi100.append(chi_square_rec_k(u_probs, s[:, i], settings.indv_k, settings.n_chi2_items))
        kendall100.append(kendall_tau_rec(u_probs, s[:, i], settings.indv_k, settings.agg_k))

    eval_res = EvalResults(settings.vae.c.n_sensitive)
    eval_res.set_results(chi100, kendall100, ndcg)
    return eval_res


# Called once for each minibatch (partial=True)
def evaluate_partial_recommendations(settings, u_probs, targets, s, eval_res):
    ndcg = ndcg_at_k(u_probs, targets, settings.k).mean()
    chi_ranks = []
    kendall_ranks = []

    # Chi2 and Kendall should be based on the full data. Reduce memory requirement by aggregating up
    # ranks for each sensitive group (n_items*n_sens*n_metrics*2) instead of probs (n_items*n_users)
    for i in range(settings.vae.c.n_sensitive):
        chi_ranks.append(get_aggregated_item_ranks(u_probs, s[:, i], settings.indv_k, False))
        kendall_ranks.append(get_aggregated_item_ranks(u_probs, s[:, i], settings.indv_k, True))

    eval_res.update(ndcg, u_probs.shape[0])
    return eval_res, chi_ranks, kendall_ranks


def evaluate_representation(zbs, s, mode_str, vae, wandb_log):
    # Evaluate how well sensitive features can be inferred from latent representation
    # Extract z
    if vae.split_latent:
        z = zbs[:, : -vae.c.b_dim]
    else:
        z = zbs

    # Fit and evaluate auxiliary model
    n_redundancy = 4
    k_split = 5
    logreg = logreg_training(n_redundancy, k_split, z, s)
    for i, sensitive_name in enumerate(vae.c.sensitive_labels):
        wandb_log[f"analysis/{sensitive_name} rep logreg, {mode_str}"] = logreg[i]


def logreg_training(n_redundancy, k_split, z, s):
    n_sensitive = s.shape[1]
    logreg = [[] for i in range(n_sensitive)]
    for i in range(n_redundancy):
        for j in range(n_sensitive):
            y = s[:, j]

            # Perform random split of data
            kf = KFold(n_splits=k_split, shuffle=True)

            for train_index, val_index in kf.split(z):
                train_z, train_y = z[train_index], y[train_index]
                val_z, val_y = z[val_index], y[val_index]

                # Fit and evaluate model
                logreg_model = LogisticRegression(max_iter=200)
                logreg_model.fit(train_z, train_y)
                logreg_probs = logreg_model.predict_proba(val_z)
                logreg[j].append(roc_auc_score(val_y, logreg_probs[:, 1]))

    logreg = [np.mean(logre) for logre in logreg]

    return logreg


# Option for applying neutral network based models for neutrality evaluations
# LogisticRegression was found to consistently achieve higher AUC, likely because of overfitting issues with
# neutral classifiers and since the applied strategies do not encode much sensitive information that is not
# linearly separable.
def neural_classification_training(n_redundancy, z, s, device, adv_dims, train_split=0.8, max_it=100):
    n_sensitive = s.shape[1]
    scores = [[] for i in range(n_sensitive)]
    n_samples = z.shape[0]
    min_train = 20
    for round in range(n_redundancy):
        for j in range(n_sensitive):
            y = s[:, j]
            all_tr = []
            all_va = []

            indexes = np.arange(n_samples)
            np.random.shuffle(indexes)
            val_split = int(n_samples * train_split)

            def to_t(arr, device):
                return torch.tensor(arr, dtype=torch.float32, device=device)

            z_t, y_t = to_t(z[:val_split], device), to_t(y[:val_split], device)
            z_v, y_v = to_t(z[val_split:], device), to_t(y[val_split:], device)

            model = BinaryClassifier(adv_dims, "selu")
            model = model.to(device)

            opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
            best = 0.0
            stop_counter = 0
            for epoch in range(max_it):
                model.train()
                loss, _ = model(z_t, y_t)
                loss.backward()
                opt.step()
                opt.zero_grad()

                model.eval()
                _, tr_logit = model(z_t, y_t)
                _, va_logit = model(z_v, y_v)
                tr_roc = roc_auc_score(y_t.cpu().detach().numpy(), tr_logit.cpu().detach().numpy())
                va_roc = roc_auc_score(y_v.cpu().detach().numpy(), va_logit.cpu().detach().numpy())
                all_tr.append(tr_roc)
                all_va.append(va_roc)

                if va_roc > best * 1.005:
                    stop_counter = 0
                    best = va_roc
                else:
                    stop_counter += 1
                    if stop_counter >= 6 and epoch >= min_train:
                        pass
            scores[j].append(max(all_va))
    scores = [np.mean(score) for score in scores]
    return scores


if __name__ == "__main__":
    parse_arguments()
