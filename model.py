import wandb
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from sklearn.metrics import roc_auc_score

# Base class with common functionality used in all other models
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def get_activation(self, hidden_activation):
        # Activation
        if hidden_activation == "tanh":
            return nn.Tanh()
        elif hidden_activation == "relu":
            return nn.ReLU()
        elif hidden_activation == "selu":
            return nn.SELU()
        elif hidden_activation == "gelu":
            return nn.GELU()
        elif hidden_activation == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f'Unknown hidden activation: "{hidden_activation}"')
            return

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            # torch.nn.init.normal_(m.bias, std=0.001)
            m.bias.data.fill_(0.0)

    def setup_layer(self, dims, hidden_activation, last_activation="", last_dropout=False, dropout=0.0):
        layers = []
        n_dims = len(dims)
        n_layers = n_dims - 1
        for i in range(n_dims - 1):
            from_dim = dims[i]
            to_dim = dims[i + 1]
            layers.append(nn.Linear(from_dim, to_dim))

            if i != n_layers - 1:
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                layers.append(self.get_activation(hidden_activation))
        if last_dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        if last_activation != "":
            # For some parts, we want an activation after the final layer
            layers.append(self.get_activation(last_activation))
        return nn.Sequential(*layers)


class FairVAERec(BaseModel):
    def __init__(self, settings):
        super().__init__()

        self.c = settings

        # Derived flags
        self.z_dim = self.c.z_enc_dims[-1] // 2
        self.baseline = self.c.gamma == 0.0 and self.c.alpha == 0.0 and self.c.delta == 0.0
        self.split_latent = self.c.gamma > 0.0 and self.c.alpha > 0.0
        self.adv = self.c.delta > 0.0

        # Flag for sampling in eval mode (drop out will not be enabled)
        self.sample = False

        # Epsilon for log expressions
        self.eps = 1e-7

        # Binary cross entropy loss
        self.bce_loss = nn.BCELoss()

        # Validity checks
        self.validity_check()

        # Noise dropout
        self.noise = nn.Dropout(self.c.noise_rate)

        # Add z encoder
        self.z_encoder = self.setup_layer(self.c.z_enc_dims, self.c.hidden_activation, dropout=self.c.dropout_vae)
        self.z_encoder.apply(self.init_weights)

        # Add z decoder
        self.z_decoder = self.setup_layer(
            self.c.z_dec_dims,
            self.c.hidden_activation,
            dropout=self.c.dropout_vae,
        )

        # Add b encoder
        if self.split_latent:
            self.b_encoder = self.setup_layer(self.c.b_enc_dims, self.c.hidden_activation, dropout=self.c.dropout_vae)
            self.b_encoder.apply(self.init_weights)

            self.b_decoder = self.setup_layer(
                self.c.b_dec_dims, self.c.hidden_activation, last_activation="sigmoid", dropout=self.c.dropout_vae
            )
            self.b_decoder.apply(self.init_weights)

        self.z_decoder.apply(self.init_weights)

        # Add KL adversarial
        if self.split_latent and len(self.c.adv_dims) > 0:
            self.kl_adversarial = self.setup_layer(
                self.c.adv_dims, self.c.hidden_activation, dropout=self.c.dropout_adv
            )
            self.kl_adversarial.apply(self.init_weights)

        # Add simple adversarials
        if self.adv:
            adversarials = []
            for i in range(self.c.n_sensitive):
                adversarials.append(
                    self.setup_layer(
                        self.c.adv_dims,
                        self.c.hidden_activation,
                        last_activation="sigmoid",
                        dropout=self.c.dropout_adv,
                    )
                )
                adversarials[i].apply(self.init_weights)
            self.adversarials = nn.ModuleList(adversarials)

    def validity_check(self):
        # Dimensions
        assert self.z_dim == self.c.z_dec_dims[0]
        if self.split_latent:
            assert self.c.b_enc_dims[-1] == 2 * self.c.b_dim

        # Options
        if self.adv:
            assert not self.split_latent
        if self.split_latent:
            assert self.c.b_dim > 0

    def get_params(self, adversarial=False):
        if not adversarial:
            params = list(self.z_encoder.parameters()) + list(self.z_decoder.parameters())
            if self.split_latent:
                params = params + list(self.b_encoder.parameters()) + list(self.b_decoder.parameters())
        else:
            if self.split_latent:
                params = list(self.kl_adversarial.parameters())
            else:
                params = []
                for adv in self.adversarials:
                    params = params + list(adv.parameters())
        return params

    def encode(self, x):
        # Normalize to remedy issues related to active vs passive users
        h = F.normalize(x, dim=1)
        # Add noise, noisy VAE
        h = self.noise(h)

        # Propagate through split encoders
        h_z = self.z_encoder(h)
        b = self.b_encoder(h) if self.split_latent else None

        # Extract mu and log(sigma^2) from h_z
        mu = h_z[:, : self.z_dim]
        logvar = h_z[:, self.z_dim :]

        # Calculate standard deviation and KL divergence
        std = torch.exp(0.5 * logvar)
        KL = torch.mean(torch.sum(0.5 * (-logvar + torch.exp(logvar) + mu ** 2 - 1), dim=1))

        if not self.split_latent:
            return mu, std, None, None, KL

        # Gaussian b
        mu_b = b[:, : self.c.b_dim]
        logvar_b = b[:, self.c.b_dim :]
        std_b = torch.exp(0.5 * logvar_b)

        # Compute KL divergence with respect to both z and b
        mu_zb = torch.cat((mu, mu_b), dim=1)
        logvar_zb = torch.cat((logvar, logvar_b), dim=1)
        KL = torch.mean(torch.sum(0.5 * (-logvar_zb + torch.exp(logvar_zb) + mu_zb ** 2 - 1), dim=1))

        return mu, std, mu_b, std_b, KL

    def forward(self, x, s, wandb_log, decode=True):
        # Encode and calc KL divergence
        mu, std, mu_b, std_b, KL = self.encode(x)

        # Sample latents
        sampled_z = self.reparam_trick(mu, std)
        if self.split_latent:
            b = self.reparam_trick(mu_b, std_b)
            sampled_latent = torch.cat((sampled_z, b), dim=1)
        else:
            sampled_latent = sampled_z

        # Skip decoding
        if not decode:
            return sampled_latent, None, None

        # x reconstruct term
        logits = self.z_decoder(sampled_z)
        log_softmax = F.log_softmax(logits, dim=1)
        negative_ll = -torch.mean(torch.sum(log_softmax * x, dim=1))

        # Shared loss parts and logging
        neg_elbo = negative_ll + self.c.beta * KL

        prefix = "train" if self.training else "validation"
        if self.c.verbose:
            wandb_log[f"{prefix}/rec recon loss"] = negative_ll
            wandb_log[f"{prefix}/KL loss"] = self.c.beta * KL

        # Add additional loss terms
        if self.split_latent:
            # a reconstruction
            logit_ab = self.b_decoder(b)

            # Sum over each sensitive attribute (we assume independence)
            negative_ab_loss = 0
            for i in range(self.c.n_sensitive):
                negative_ab_loss += self.bce_loss(logit_ab[:, i], s[:, i])
            # Divide by n_sesitive to normalize loss contribution
            negative_ab_loss = negative_ab_loss / self.c.n_sensitive

            # Split latent KL approx
            if not self.c.ganKL:
                factor_KL = self.empiric_factor_KL(sampled_latent)
            else:
                # One can show that the resulting loss is equal to the sigmoid input,
                # when using sigmoids to produce probabilities
                # ln(p(x))-ln(1-p(x)): ln(sigmoid(x))-ln(1-sigmoid(x)) = x
                adv_out = self.kl_adversarial(sampled_latent)
                factor_KL = torch.mean(adv_out)

            # Add loss terms
            neg_elbo += self.c.alpha * negative_ab_loss + self.c.gamma * factor_KL

            # Logging
            if self.c.verbose:
                self.log_reclassification(logit_ab, s, wandb_log)
                wandb_log[f"{prefix}/a recon loss"] = self.c.alpha * negative_ab_loss
                wandb_log[f"{prefix}/factor KL loss"] = self.c.gamma * factor_KL

        elif self.adv:
            # Adversarial losses
            # Sum over each sensitive attribute
            logit_s = self.adversarials[0](sampled_latent)
            s_reclass_loss = self.bce_loss(logit_s[:, 0], s[:, 0])
            for i in range(1, self.c.n_sensitive):
                logit_s = torch.cat((logit_s, self.adversarials[i](sampled_latent)), dim=1)
                s_reclass_loss += self.bce_loss(logit_s[:, i], s[:, i])

            # Negate classification loss to optimize the main model to maximize it
            adv_loss = -s_reclass_loss

            # Add loss term
            neg_elbo += self.c.delta * adv_loss

            # Logging
            if self.c.verbose:
                wandb_log[f"{prefix}/adv loss"] = self.c.delta * adv_loss

        if self.c.verbose:
            wandb_log[f"{prefix}/loss"] = neg_elbo
        return sampled_latent, neg_elbo, F.softmax(logits, dim=1)

    def discriminate(self, zb_real, wandb_log):
        zb_fake = zb_real.detach().clone()

        zb_fake = self.shuffle_fake_zb(zb_fake)

        probs_real = torch.sigmoid(self.kl_adversarial(zb_real))
        probs_fake = torch.sigmoid(self.kl_adversarial(zb_fake))
        if self.c.verbose:
            probs_real_np = probs_real.cpu().detach().numpy()
            probs_fake_np = probs_fake.cpu().detach().numpy()
            n_probs = probs_real_np.shape[0]
            targets = np.zeros(n_probs + probs_fake_np.shape[0])
            targets[:n_probs] = 1
            probs = np.concatenate((probs_real_np, probs_fake_np), axis=0)
            roc_auc = roc_auc_score(targets, probs)
            prefix = "train" if self.training else "val"
            wandb_log[f"analysis/{prefix} disc auc"] = roc_auc
            wandb_log[f"analysis/{prefix} real prob"] = probs_real_np.mean()
            wandb_log[f"analysis/{prefix} fake prob"] = probs_fake_np.mean()
            # print(f"AUC: {roc_auc}, real: {probs_real_np.mean()}, fake: {probs_fake_np.mean()}")

        # All "real" zbs are given the target 1, while the "fake" are given 0, resulting in:
        n_samples = probs_real.shape[0] + probs_fake.shape[0]
        binary_cross_entropy = torch.sum(torch.log(probs_real + self.eps)) + torch.sum(
            torch.log(1 - probs_fake + self.eps)
        )
        binary_cross_entropy_loss = -binary_cross_entropy / n_samples
        return binary_cross_entropy_loss

    def discriminate_simple(self, z, targets, wandb_log):
        z_clone = z.detach().clone()

        cross_entropy_loss = 0
        for i in range(self.c.n_sensitive):
            target = targets[:, i]
            probs = self.adversarials[i](z_clone)

            # Logging
            sensitive_label = self.c.sensitive_labels[i]
            target_np = target.cpu().detach().numpy()
            probs_np = probs.cpu().detach().numpy()
            roc_auc = roc_auc_score(target_np, probs_np)
            prefix = "train" if self.training else "val"
            n_true = np.count_nonzero(target_np)
            n_false = target_np.shape[0] - n_true
            mean_true_prob = np.matmul(target_np, probs_np) / n_true
            mean_false_prob = np.matmul(1 - target_np, probs_np) / n_false
            wandb_log[f"analysis/{prefix} {sensitive_label} disc auc A"] = roc_auc
            wandb_log[f"analysis/{prefix} {sensitive_label} real prob A"] = mean_true_prob
            wandb_log[f"analysis/{prefix} {sensitive_label} fake prob A"] = mean_false_prob

            # Loss
            new_loss_part = torch.matmul(target, torch.log(probs + self.eps)) + torch.matmul(
                1 - target, torch.log((1 - probs) + self.eps)
            )
            cross_entropy_loss += new_loss_part / probs.shape[0]
        cross_entropy_loss = -cross_entropy_loss
        return cross_entropy_loss

    def reparam_trick(self, mu, std):
        epsilon = torch.empty(std.shape, dtype=torch.float32, device=self.c.device).normal_(mean=0, std=1)
        # Turn on sampling during training or whenever sampling is turned on
        sample_flag = self.training or self.sample
        sampled = mu + sample_flag * epsilon * std
        return sampled

    def empiric_factor_KL(self, sampled_zb):
        # Detach "ideal" distribution for which we will zero covars of z and b
        # NB: Detaching this has little effect on results, but is more correct
        # as we only want gradients from how the actual encoded distribution
        # should be changed to be more like the ideal one, not ALSO vice versa
        ideal_zb = sampled_zb.detach()

        # Get empiric covariance matrix
        cov = torch.cov(sampled_zb.T)

        # Get empiric covariance of ideal distribution and set covars of z and be to 0
        cov_factor = torch.cov(ideal_zb.T)
        cov_factor[: self.z_dim, self.z_dim :] = 0
        cov_factor[self.z_dim :, : self.z_dim] = 0

        # Invert block diagonal cov matrix
        inv_cov_factor = torch.zeros(cov_factor.shape, dtype=torch.float32, device=self.c.device)
        inv_cov_factor[: self.z_dim, : self.z_dim] = torch.inverse(cov_factor[: self.z_dim, : self.z_dim])
        inv_cov_factor[self.z_dim :, self.z_dim :] = torch.inverse(cov_factor[self.z_dim :, self.z_dim :])

        # Calculate approximated KL divergence from actual latent distribution to idealized distribution
        # with no (empiric) covariance between z and b
        d = cov.shape[0]
        factor_KL = 0.5 * (
            torch.log(torch.det(cov_factor) / torch.det(cov) + self.eps)
            - d
            + torch.trace(torch.matmul(inv_cov_factor, cov))
        )
        return factor_KL

    def log_reclassification(self, logits, s, wandb_log):
        # a reclassification for split latents and adversarial reclassification in non-split latents
        if self.c.verbose and not self.training:
            probs_np = logits.cpu().detach().numpy()
            s_np = s.cpu().detach().numpy()
            for i in range(self.c.n_sensitive):
                sensitive_label = self.c.sensitive_labels[i]
                auc = roc_auc_score(s_np[:, i], probs_np[:, i])
                wandb_log[f"analysis/{sensitive_label} auc"] = auc

    def shuffle_fake_zb(self, fake_zb):
        n = fake_zb.shape[0]
        fake_zb[:, -self.c.b_dim :] = fake_zb[torch.randperm(n, device=self.c.device), -self.c.b_dim :]
        return fake_zb


class VAESettings(object):
    def __init__(
        self,
        z_enc_dims=[],
        z_dec_dims=[],
        b_enc_dims=[],
        b_dec_dims=[],
        adv_dims=[],
        noise_rate=0.5,
        dropout_vae=0.0,
        dropout_adv=0.0,
        beta=0.2,
        alpha=300,
        gamma=80,
        delta=0.0,
        ganKL=False,
        n_sensitive=0,
        sensitive_labels=None,
        b_dim=0,
        hidden_activation="tanh",
        device="cuda:0",
        verbose=False,
    ):
        self.z_enc_dims = z_enc_dims
        self.z_dec_dims = z_dec_dims
        self.b_enc_dims = b_enc_dims
        self.b_dec_dims = b_dec_dims
        self.adv_dims = adv_dims
        self.noise_rate = noise_rate
        self.dropout_vae = dropout_vae
        self.dropout_adv = dropout_adv
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.ganKL = ganKL
        self.n_sensitive = n_sensitive
        self.sensitive_labels = sensitive_labels
        self.b_dim = b_dim
        self.hidden_activation = hidden_activation
        self.device = device
        self.verbose = verbose


class RunSettings(object):
    def __init__(
        self,
        vae=None,
        opt_vae=None,
        opt_adv=None,
        train_loader=None,
        val_loader=None,
        n_epochs=0,
        n_adv_pre=0,
        nbatch_per_update=1,
        discriminate=False,
        n_adv_train=1,
        k=10,
        indv_k=100,
        agg_k=100,
        n_chi2_items=0,
        train_eval=True,
        rep_eval_interval=100,
        partial=False,
        csr=False,
    ):
        self.vae = vae
        self.opt_vae = opt_vae
        self.opt_adv = opt_adv
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epochs = n_epochs
        self.n_adv_pre = n_adv_pre
        self.nbatch_per_update = nbatch_per_update
        self.discriminate = discriminate
        self.n_adv_train = n_adv_train
        self.k = k
        self.indv_k = indv_k
        self.agg_k = agg_k
        self.n_chi2_items = n_chi2_items
        self.train_eval = train_eval
        self.rep_eval_interval = rep_eval_interval
        self.partial = partial
        self.csr = csr


class EvalStorage(object):
    def __init__(self):
        self.zbs = []
        self.s = []
        self.probs = []
        self.u_probs = []
        self.targets = []
        self.chi_ranks = None
        self.kendall_ranks = None

    def update(self, zbs, s, probs=None, u_probs=None, targets=None):
        self.zbs.append(zbs)
        self.s.append(s)
        if probs is not None:
            self.probs.append(probs)
        if u_probs is not None:
            self.u_probs.append(u_probs)
        if targets is not None:
            self.targets.append(targets)

    def update_ranks(self, chi_ranks, kendall_ranks):
        if self.chi_ranks is None and self.kendall_ranks is None:
            self.chi_ranks = chi_ranks
            self.kendall_ranks = kendall_ranks
        else:
            for i, sensitive_attr in enumerate(chi_ranks):
                for j, sensitive_group_ranks in enumerate(sensitive_attr):
                    self.chi_ranks[i][j] += sensitive_group_ranks
            for i, sensitive_attr in enumerate(kendall_ranks):
                for j, sensitive_group_ranks in enumerate(sensitive_attr):
                    self.kendall_ranks[i][j] += sensitive_group_ranks

    def concat(self):
        self.zbs = np.concatenate(self.zbs)
        self.s = np.concatenate(self.s)
        self.probs = np.concatenate(self.probs) if self.probs != [] else []
        self.u_probs = np.concatenate(self.u_probs) if self.u_probs != [] else []
        self.targets = np.concatenate(self.targets) if self.targets != [] else []


class EvalResults(object):
    def __init__(self, n_sensitive):
        self.n_sensitive = n_sensitive
        self.ndcg = 0
        self.chi100 = [0] * n_sensitive
        self.kendall100 = [0] * n_sensitive

    def set_results(self, chi100, kendall100, ndcg=None):
        if ndcg is not None:
            self.ndcg = ndcg
        for i in range(self.n_sensitive):
            self.chi100[i] = chi100[i]
            self.kendall100[i] = kendall100[i]

    def update(self, ndcg, batch_size):
        self.ndcg += ndcg * batch_size

    def aggregate_parts(self, n_users):
        self.ndcg = self.ndcg / n_users


class BinaryClassifier(BaseModel):
    def __init__(self, dims, hidden_activation, dropout_rate=0.0):
        super().__init__()

        self.dims = dims
        self.eps = 1e-7
        self.bce_loss = nn.BCELoss()

        # Add classifier
        self.classifier = self.setup_layer(
            dims,
            hidden_activation,
            last_activation="sigmoid",
            last_dropout=False,
            dropout=dropout_rate,
        )
        self.classifier.apply(self.init_weights)

    def forward(self, x, s):
        logit = self.classifier(x)
        negative_loss = self.bce_loss(logit.flatten(), s)

        return negative_loss, logit
