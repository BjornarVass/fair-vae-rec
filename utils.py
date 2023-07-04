import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import kendalltau


def generate_network_dims(input_dim, hidden_dim, output_dim, n_hidden=0, red_hidden_dim=-1):
    network_dims = [input_dim]
    for i in range(n_hidden):
        if i == 0 or red_hidden_dim < 0:
            network_dims.append(hidden_dim)
        else:
            network_dims.append(red_hidden_dim)
    network_dims.append(output_dim)
    return network_dims


def get_ranking_info(probs, rec_targets, k, rec_indexes):
    # Disregard users with no correct recommendations
    mask = rec_targets.sum(axis=1) != 0
    targets = rec_targets[mask]

    n_users = targets.shape[0]

    if rec_indexes is None:
        probs = probs[mask]

        # Partition to place indexes of k top probs (unsorted) in the begining of the list
        index_partition = np.argpartition(-probs, k, axis=1)

        # Top k probs (unsorted)
        topk_probs = probs[np.arange(n_users)[:, np.newaxis], index_partition[:, :k]]
        # Top k internal indexes (sorted, indexed 0-(k-1))
        topk_sorted_indexes = np.argsort(-topk_probs, axis=1)

        # Top k SORTED ITEM indexes
        rec_indexes = index_partition[np.arange(n_users)[:, np.newaxis], topk_sorted_indexes]
    else:
        rec_indexes = rec_indexes[mask]

    return rec_indexes, targets, n_users


def ndcg_at_k(probs, rec_targets, k=10, rec_indexes=None):
    rec_indexes, targets, n_users = get_ranking_info(probs, rec_targets, k, rec_indexes)

    # Rank discount
    rank_discount = 1.0 / np.log2(np.arange(2, k + 2))

    # DCG utilize that targets are either 1 or 0 and multiplies with the rank discounts
    # IDCG simply assumes that min(k, n_targets) targets inhabits the best rankings
    DCG = (targets[np.arange(n_users)[:, np.newaxis], rec_indexes] * rank_discount).sum(axis=1)
    IDCG = np.array([(rank_discount[: min(int(n), k)]).sum() for n in targets.sum(axis=1)])

    NDCG = DCG / IDCG
    return NDCG


def get_aggregated_item_ranks(scores, sensitive_labels, k, discounted):
    mask = sensitive_labels.astype(bool)

    # item rankings per sensitive group
    def sensitive_ranks(scores, k, discounted):
        # New array for housing aggregated scores of sensitive group
        n_users = scores.shape[0]
        rank_scores = np.zeros(scores.shape)

        # Partition to place indexes of top k scores (unsorted) in the beginning of the list
        index_partition = np.argpartition(-scores, k, axis=1)

        # Top k probs (unsorted)
        topk_scores = scores[np.arange(n_users)[:, np.newaxis], index_partition[:, :k]]
        # Top k internal indexes (sorted, indexed 0-(k-1))
        topk_sorted_indexes = np.argsort(-topk_scores, axis=1)

        # Sorted top k item indexes
        sorted_score_indexes = index_partition[np.arange(n_users)[:, np.newaxis], topk_sorted_indexes]

        # Aggregate discounted recommendation lists over all users of the same sensitive group
        if discounted:
            rank_values = 1.0 / np.log2(np.arange(2, k + 2))
        else:
            rank_values = np.ones(k)
        rank_scores[np.arange(n_users)[:, np.newaxis], sorted_score_indexes] = rank_values
        # Aggregate, 2D->1D
        rank_scores = rank_scores.sum(0)
        return rank_scores

    s1 = sensitive_ranks(scores[mask], k, discounted)
    s0 = sensitive_ranks(scores[~mask], k, discounted)
    return [s0, s1]


def chi_square_rec_k(scores, sensitive_labels, k, n_considered, precomputed_ranks=None):
    if precomputed_ranks is None:
        discounted = False
        cols = get_aggregated_item_ranks(scores, sensitive_labels, k, discounted)
    else:
        cols = precomputed_ranks

    # Create contingency table
    ct = np.concatenate((cols[0].reshape(-1, 1), cols[1].reshape(-1, 1)), axis=1)

    # Get expected values adjusted for the number of users that can be recommended each item
    n_items = ct.shape[0]
    adjusted_exps = np.zeros(ct.shape)
    mask = sensitive_labels.astype(bool)
    scores0 = scores[~mask]
    scores1 = scores[mask]
    n0 = scores0.shape[0]
    n1 = scores1.shape[0]
    for i in range(n_items):
        n_obs = ct[i].sum()

        # User who have interacted with an item cannot be recommended the same item
        # I.e., we subtract the number of times we have flagged the item with
        # a score lower than 0 for each user group
        n_available0 = n0 - np.count_nonzero(scores0[:, i] < 0)
        n_available1 = n1 - np.count_nonzero(scores1[:, i] < 0)
        n_available = n_available0 + n_available1

        adjusted_exps[i, 0] = (n_available0 / n_available) * n_obs
        adjusted_exps[i, 1] = (n_available1 / n_available) * n_obs

    index_partition = np.argpartition(-adjusted_exps.sum(1), n_considered)
    adjusted_exps = adjusted_exps[index_partition[:n_considered]]
    ct = ct[index_partition[:n_considered]]
    if adjusted_exps.sum(1).min() < 3:
        return -1
    chi2_ad = (ct - adjusted_exps) ** 2 / adjusted_exps
    return chi2_ad.sum()


def kendall_tau_rec(scores, sensitive_labels, agg_k, indv_k, precomputed_ranks=None):
    if precomputed_ranks is None:
        discounted = True
        all_ranks = get_aggregated_item_ranks(scores, sensitive_labels, indv_k, discounted)
    else:
        all_ranks = precomputed_ranks

    def get_top_k_aggregated_ranks(all_ranks, agg_k):
        out_ranks = []
        for s_ranks in all_ranks:
            recommended_items = np.argpartition(-s_ranks, agg_k)[:agg_k]
            item_order = np.argsort(-s_ranks[recommended_items])
            out_ranks.append(recommended_items[item_order])
        return out_ranks

    ranks = get_top_k_aggregated_ranks(all_ranks, agg_k)

    return extended_tau(ranks[0], ranks[1])


############################ START BORROWED CODE #########################################
# Implementation borrowed from https://godatadriven.com/blog/using-kendalls-tau-to-compare-recommendations/
# All credit goes to Rogier van der Geeer
# Blogpost date: 26. July 2016
def extended_tau(list_a, list_b):
    """Calculate the extended Kendall tau from two lists."""
    ranks = join_ranks(create_rank(list_a), create_rank(list_b)).fillna(len(list_a))
    dummy_df = pd.DataFrame(
        [{"rank_a": len(list_a), "rank_b": len(list_b)} for i in range(len(list_a) * 2 - len(ranks))]
    )
    total_df = ranks.append(dummy_df)
    return scale_tau(len(list_a), kendalltau(total_df["rank_a"], total_df["rank_b"])[0])


def scale_tau(length, value):
    """Scale an extended tau correlation such that it falls in [-1, +1]."""
    n_0 = 2 * length * (2 * length - 1)
    n_a = length * (length - 1)
    n_d = n_0 - n_a
    min_tau = (2.0 * n_a - n_0) / (n_d)
    return 2 * (value - min_tau) / (1 - min_tau) - 1


def create_rank(a):
    """Convert an ordered list to a DataFrame with ranks."""
    return pd.DataFrame(zip(a, range(len(a))), columns=["key", "rank"]).set_index("key")


def join_ranks(rank_a, rank_b):
    """Join two rank DataFrames."""
    return rank_a.join(rank_b, lsuffix="_a", rsuffix="_b", how="outer")


############################ END BORROWED CODE ##########################################
