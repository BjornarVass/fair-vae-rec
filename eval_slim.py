import numpy as np
import pandas as pd
import wandb

from scipy.sparse import csr_matrix
from scipy.special import softmax
from SLIM import SLIM, SLIMatrix

from data_processing import preprocess_movielens, preprocess_lastfm, load_and_uncompress
from utils import ndcg_at_k, chi_square_rec_k, kendall_tau_rec


def dictnp_to_np(data, n_users, k, dtype=None):
    if dtype is None:
        dtype = data[0].dtype
    np_array = np.zeros((n_users, k), dtype=dtype)
    for k, v in data.items():
        np_array[k] = v.astype(dtype)
    return np_array


def align_all_scores(recs, scores):
    # SLIM may assign multiple items a 0-score and all 0-scored items in a recommendation
    # list will have the same label. This function reverts to one-hot encoding and replaces
    # duplicate item labels.
    n_users = len(recs)
    n_items = recs[0].shape[0]
    recs = dictnp_to_np(recs, n_users, n_items, dtype=np.int32)
    scores = dictnp_to_np(scores, n_users, n_items)

    for i in range(n_users):
        u_scores = scores[i][scores[i] > 0]
        insert_ind = u_scores.shape[0]
        missing = np.setdiff1d(np.arange(n_items), recs[i, :insert_ind])
        recs[i, insert_ind:] = missing
        scores[i] = scores[i, np.argsort(recs[i])]

    return scores


def main():
    # Run parameters
    dataset = "lastfm"
    group_name = "SLIM lastfm chi750"
    redundancy = 1

    for run in range(redundancy):
        train_slim(dataset, group_name, run)


def train_slim(dataset, group_name, run):
    balance_sensitive = True
    validation_frac = 0.2
    n_sensitive = 2
    k = 10
    user_split = True
    csr = True
    skip_pre_process = True

    n_chi2_items = 0
    agg_k = 100
    # How many of individual user's recommendations considered when aggregating
    # sensitive group rankings
    indv_k = 100

    if dataset == "movielens":
        DATA_DIR = "data/"
        PROCESSED_DIR = f"pro_ml/{run}/"
        ratings_filename = "ratings.dat"
        user_info_filename = "users.dat"

        # Number of items considered in Chi2 metric for k = 10 and k = 100
        chi_n_items = 600

        if not skip_pre_process:
            preprocess_movielens(
                DATA_DIR,
                PROCESSED_DIR,
                ratings_filename,
                user_info_filename,
                validation_frac,
                balance_sensitive=balance_sensitive,
                user_split=user_split,
            )

    elif dataset == "lastfm":
        csr = True
        DATA_DIR = "data_lastfm/"
        PROCESSED_DIR = f"pro_lf/{run}/"
        ratings_filename = "lastfm.tsv"
        user_info_filename = "users.tsv"
        album_filename = "albums.tsv"
        chi_n_items = 750
        if not skip_pre_process:
            preprocess_lastfm(
                DATA_DIR,
                PROCESSED_DIR,
                ratings_filename,
                user_info_filename,
                album_filename,
                validation_frac,
                balance_sensitive=balance_sensitive,
                user_split=user_split,
            )
    if dataset == "movielens20":
        PROCESSED_DIR = "pro_sg/"
        train_file = "train.csv"
        val_tr_file = "validation_tr.csv"
        val_te_file = "validation_te.csv"

        def load_train_data(csv_file):
            tp = pd.read_csv(csv_file)
            n_users = tp["uid"].max() + 1
            n_items = tp["sid"].unique().shape[0]

            rows, cols = tp["uid"], tp["sid"]
            data = csr_matrix((np.ones_like(rows), (rows, cols)), dtype="float64", shape=(n_users, n_items))
            return data, n_items

        def load_tr_te_data(csv_file_tr, csv_file_te, n_items):
            tp_tr = pd.read_csv(csv_file_tr)
            tp_te = pd.read_csv(csv_file_te)

            start_idx = min(tp_tr["uid"].min(), tp_te["uid"].min())
            end_idx = max(tp_tr["uid"].max(), tp_te["uid"].max())

            rows_tr, cols_tr = tp_tr["uid"] - start_idx, tp_tr["sid"]
            rows_te, cols_te = tp_te["uid"] - start_idx, tp_te["sid"]

            data_tr = csr_matrix(
                (np.ones_like(rows_tr), (rows_tr, cols_tr)), dtype="float64", shape=(end_idx - start_idx + 1, n_items)
            )
            data_te = csr_matrix(
                (np.ones_like(rows_te), (rows_te, cols_te)), dtype="float64", shape=(end_idx - start_idx + 1, n_items)
            )
            return data_tr, data_te

        train_data, n_items = load_train_data(PROCESSED_DIR + train_file)
        val_tr, val_te = load_tr_te_data(PROCESSED_DIR + val_tr_file, PROCESSED_DIR + val_te_file, n_items)
    else:
        train_data, val_tr, val_te, test_tr, test_te, sensitive_labels, train_s, val_s, test_s = load_and_uncompress(
            PROCESSED_DIR, user_split=user_split, csr=csr
        )
    print(train_data.shape)
    if csr:
        trainmat = SLIMatrix(train_data)
        if user_split:
            testmat = SLIMatrix(test_tr)
    else:
        train_data_csr = csr_matrix(train_data)
        trainmat = SLIMatrix(train_data_csr)
        if user_split:
            test_in_csr = csr_matrix(test_tr)
            testmat = SLIMatrix(test_in_csr)
    if not user_split:
        testmat = trainmat

    params = {"nthreads": 8, "l1r": 1.0, "l2r": 1.0}

    model = SLIM()
    model.train(params, trainmat)

    preds = model.predict(testmat, nrcmds=k)

    preds = dictnp_to_np(preds, test_te.shape[0], k, dtype=np.int32)

    if csr:
        test_dat = test_te.toarray()
    else:
        test_dat = test_te

    ndcg = ndcg_at_k(None, test_dat, k=k, rec_indexes=preds)
    wandb_log = {}
    wandb_log["NDCG"] = ndcg.mean()

    all_preds, all_scores = model.predict(testmat, nrcmds=test_te.shape[1], returnscores=True)
    full_scores = align_all_scores(all_preds, all_scores)

    for i, s in enumerate(sensitive_labels):
        chi2_stat = chi_square_rec_k(full_scores, test_s[:, i], indv_k, n_chi2_items)
        kendall = kendall_tau_rec(full_scores, test_s[:, i], indv_k, agg_k)
        wandb_log[f"{s} chi2@{fair_k}"] = chi2_stat
        wandb_log[f"{s} kendall@{fair_k}"] = kendall
    wandb.init(group=group_name)
    wandb.log(wandb_log)
    wandb.finish()


if __name__ == "__main__":
    main()
