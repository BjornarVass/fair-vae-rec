import os
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix


def preprocess_movielens(
    DATA_DIR,
    PROCESSED_DATA_DIR,
    ratings_filename,
    user_filename,
    validation_frac,
    balance_sensitive=True,
    user_split=True,
):
    # Load raw data
    ratings = pd.read_csv(
        os.path.join(DATA_DIR, ratings_filename),
        sep="::",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    user_info = pd.read_csv(
        os.path.join(DATA_DIR, user_filename),
        sep="::",
        names=["user_id", "gender", "age", "occupation", "zip code"],
        engine="python",
    )

    # Make movie ratings implicit using established cutoff rating from literature
    ratings = ratings[ratings["rating"] > 3.5]
    ratings = ratings.drop(columns=["rating", "timestamp"])

    # Create binary labels for sensitive user information, female=1 and age>35=1
    user_info["gender"] = (user_info["gender"] == "F").astype(np.int64)
    user_info["age"] = (user_info["age"] > 35).astype(np.int64)

    # Discard unused fields
    user_info = user_info.loc[:, ["user_id", "gender", "age"]]

    # Call general preprocessing function
    preprocess(PROCESSED_DATA_DIR, validation_frac, balance_sensitive, user_split, ratings, user_info)


def preprocess_lastfm(
    DATA_DIR,
    PROCESSED_DATA_DIR,
    events_filename,
    user_filename,
    album_filename,
    validation_frac,
    balance_sensitive=True,
    user_split=True,
):
    # Load prepared data. Chunking is applied for memory concerns
    # (2 last years of the 2 bill version of lastFM, excluding users for whom we do not have both gender and age)
    chunksize = 5 * 10 ** 7

    # Events are listed for albums. Create dictionary that map albums to artists
    # A few artist names contains the separator character (tabular), and are dropped
    full_artist_dict = {}
    with pd.read_csv(
        os.path.join(DATA_DIR, album_filename), chunksize=chunksize, sep="\t", error_bad_lines=False
    ) as reader:
        for chunk in reader:
            artist_dict = pd.Series(chunk.artist_name.values, index=chunk.album_id).to_dict()
            full_artist_dict.update(artist_dict)

    # Introduce artist_ids to map album ids to artist ids and not names
    unique_artists = set(full_artist_dict.values())
    artist_id_remap = {artist: i for i, artist in enumerate(unique_artists)}
    artist_id_dict = {album_id: artist_id_remap[artist] for album_id, artist in full_artist_dict.items()}

    # Read prepared event data in chunks while mapping album ids to artist ids and dropping duplicate entries
    df = pd.DataFrame()
    with pd.read_csv(
        os.path.join(DATA_DIR, events_filename), chunksize=chunksize, sep="\t", parse_dates=[2]
    ) as reader:
        for chunk in reader:
            out = chunk.loc[:, ["user_id", "album_id"]]
            out["item_id"] = out.album_id.map(artist_id_dict)

            # Mapping step introduces NaNs because of missing artists, remove these and change type back to int
            out = out.dropna()
            out.item_id = out.item_id.astype(int)

            # Drop album column and drop duplicates in concated DataFrame
            out = out.drop(columns=["album_id"])
            df = pd.concat([df, out])
            df = df.drop_duplicates()

    # Reduce item space by removing artists with few associated events
    item_counts = df.item_id.value_counts()
    filtered_items = item_counts.index[item_counts >= 50]
    # 110: 22k, 100: 24k, 90: 26k, 80:29k, 70: 32k, 60: 36.5k, 50: 42k,  40: 50k, 30: 61k, 20: 82k, 10: 133k, 5: 221k
    df = df.loc[df.item_id.isin(filtered_items)]

    # Read user data, only include users with age info and where gender is either m or f
    user_info = pd.read_csv(os.path.join(DATA_DIR, user_filename), sep="\t")
    user_info = user_info.loc[
        (user_info.age != -1) & ((user_info.gender == "m") | (user_info.gender == "f")), ["user_id", "age", "gender"]
    ]

    # Create binary labels for sensitive user information, female=1 and age>35=1
    user_info["gender"] = (user_info["gender"] == "f").astype(np.int64)
    user_info["age"] = (user_info["age"] > 35).astype(np.int64)

    # Discard unused fields and pre-filter users due to size of user_info
    user_info = user_info.loc[:, ["user_id", "gender", "age"]]
    user_info = user_info.loc[user_info.user_id.isin(df.user_id.unique())]

    # Call general preprocessing function
    preprocess(PROCESSED_DATA_DIR, validation_frac, balance_sensitive, user_split, df, user_info)


def preprocess(
    PROCESSED_DATA_DIR,
    validation_frac,
    balance_sensitive,
    user_split,
    events,
    user_info,
):
    if user_split:
        if balance_sensitive:
            # Join in gender and age info temporarily
            events = events.join(user_info.set_index("user_id"), on="user_id")

            # Ensure similar compositions wrt sensitive attributes within each split
            train_df = val_df = test_df = pd.DataFrame()
            for gender_int in [0, 1]:
                for age_int in [0, 1]:
                    tr_df, va_df, te_df = split_data(
                        events.query(f"gender == {gender_int} & age == {age_int}"),
                        validation_frac,
                        on_user=True,
                    )
                    train_df = pd.concat([train_df, tr_df])
                    val_df = pd.concat([val_df, va_df])
                    test_df = pd.concat([test_df, te_df])
        else:
            train_df, val_df, test_df = split_data(events, validation_frac, on_user=True)
    else:
        train_df, val_df, test_df = split_data(events, validation_frac, on_user=False)

    # Filter out items not found in training set and reindex
    train_df, val_df, test_df = filter_and_reindex_items(train_df, val_df, test_df)

    # Save compressed processed data
    unique_train_items = train_df["item_id"].unique()
    with open(os.path.join(PROCESSED_DATA_DIR, "unique_items.txt"), "w") as f:
        for item_id in unique_train_items:
            f.write(f"{item_id}\n")

    save_df(PROCESSED_DATA_DIR, train_df, "train.csv")
    if user_split:
        # Split validation and test sets for vae evaluation
        val_tr, _, val_te = split_on_item(val_df, validation_frac, valtest=False)
        test_tr, _, test_te = split_on_item(test_df, validation_frac, valtest=False)

        save_df(PROCESSED_DATA_DIR, val_tr, "val_tr.csv")
        save_df(PROCESSED_DATA_DIR, val_te, "val_te.csv")
        save_df(PROCESSED_DATA_DIR, test_tr, "test_tr.csv")
        save_df(PROCESSED_DATA_DIR, test_te, "test_te.csv")
    else:
        save_df(PROCESSED_DATA_DIR, val_df, "val.csv")
        save_df(PROCESSED_DATA_DIR, test_df, "test.csv")

    save_df(PROCESSED_DATA_DIR, user_info, "user_info.csv", index=False)

    # with open(os.path.join(PROCESSED_DATA_DIR, "reverse_item_map.pickle"), "wb") as f:
    #    pickle.dump(reverse_map, f)


def load_and_uncompress(PROCESSED_DATA_DIR, user_split=True, csr=False, uncompress=True):
    unique_item_ids = []
    with open(os.path.join(PROCESSED_DATA_DIR, "unique_items.txt"), "r") as f:
        for line in f:
            unique_item_ids.append(line.strip())
    n_items = len(unique_item_ids)

    # reverse_item_map = {}
    # with open(os.path.join(PROCESSED_DATA_DIR, "reverse_item_map.pickle"), "rb") as f:
    #    reverse_item_map = pickle.load(f)

    user_info = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "user_info.csv"))
    sensitive_labels = list(user_info.columns[1:])

    if user_split:
        train_data, train_s = load_train_on_user(
            os.path.join(PROCESSED_DATA_DIR, "train.csv"), n_items, user_info, csr, uncompress
        )
        val_tr, _, val_te, val_s = load_tr_te_on_item(
            os.path.join(PROCESSED_DATA_DIR, "val_tr.csv"),
            None,
            os.path.join(PROCESSED_DATA_DIR, "val_te.csv"),
            n_items,
            user_info,
            csr,
            uncompress,
            valtest=False,
        )
        test_tr, _, test_te, test_s = load_tr_te_on_item(
            os.path.join(PROCESSED_DATA_DIR, "test_tr.csv"),
            None,
            os.path.join(PROCESSED_DATA_DIR, "test_te.csv"),
            n_items,
            user_info,
            csr,
            uncompress,
            valtest=False,
        )
        return train_data, val_tr, val_te, test_tr, test_te, sensitive_labels, train_s, val_s, test_s
    else:
        train_data, val_data, test_data, train_s = load_tr_te_on_item(
            os.path.join(PROCESSED_DATA_DIR, "train.csv"),
            os.path.join(PROCESSED_DATA_DIR, "val.csv"),
            os.path.join(PROCESSED_DATA_DIR, "test.csv"),
            n_items,
            user_info,
            csr,
            uncompress,
        )
        return train_data, None, val_data, None, test_data, sensitive_labels, train_s, None, None


def load_tr_te_on_item(file_path_tr, file_path_val, file_path_te, n_items, user_info, csr, uncompress, valtest=True):
    df_tr = pd.read_csv(file_path_tr)
    if valtest:
        df_val = pd.read_csv(file_path_val)
    df_te = pd.read_csv(file_path_te)

    # Extract relevant user info
    unique_users_tr = df_tr["user_id"].unique()
    n_users_tr = unique_users_tr.shape[0]
    user_info_tr = user_info.loc[user_info.user_id.isin(unique_users_tr)]

    # No point in adding zero rows for unseen users, reindex user_ids
    reindex_user_tr = dict((uid, i) for (i, uid) in enumerate(unique_users_tr))
    # reindex_user_te = dict((uid, i) for (i, uid) in enumerate(unique_users_te))
    df_tr.loc[:, "user_id"] = df_tr["user_id"].apply(lambda x: reindex_user_tr[x])
    if valtest:
        df_val.loc[:, "user_id"] = df_val["user_id"].apply(lambda x: reindex_user_tr[x])
    df_te.loc[:, "user_id"] = df_te["user_id"].apply(lambda x: reindex_user_tr[x])
    # The following line + subsequent sort is only needed in case pandas change unique()
    user_info_tr.loc[:, "user_id"] = user_info_tr["user_id"].map(lambda x: reindex_user_tr[x])

    # Sort user_info to ensure user mapping and drop user_id
    user_info_tr = user_info_tr.sort_values(by="user_id")
    if not uncompress:
        return df_tr, df_val, df_te, user_info_tr

    # n_users_tr is used in both to match the samples in case some users are not present in df_te
    data_tr = uncompress_data(df_tr, n_users_tr, n_items, csr)
    data_val = uncompress_data(df_val, n_users_tr, n_items, csr) if valtest else None
    data_te = uncompress_data(df_te, n_users_tr, n_items, csr)

    user_info_tr = user_info_tr.drop(columns=["user_id"])
    return data_tr, data_val, data_te, user_info_tr.values


def load_train_on_user(file_path, n_items, user_info, csr, uncompress):
    df = pd.read_csv(file_path)

    # Extract relevant user info
    unique_users = df["user_id"].unique()
    n_users = unique_users.shape[0]
    train_user_info = user_info.loc[user_info.user_id.isin(unique_users)]

    # Reindex user ids
    user_reindex = dict((uid, i) for i, uid in enumerate(unique_users))
    df.loc[:, "user_id"] = df["user_id"].map(lambda x: user_reindex[x])
    # The following line + subsequent sort are only needed in case pandas change unique()
    train_user_info.loc[:, "user_id"] = train_user_info["user_id"].map(lambda x: user_reindex[x])

    # Sort user_info to ensure user mapping and drop user_id
    train_user_info = train_user_info.sort_values(by="user_id")
    if not uncompress:
        return df, train_user_info

    data = uncompress_data(df, n_users, n_items, csr)

    train_user_info = train_user_info.drop(columns=["user_id"])
    return data, train_user_info.values


def uncompress_data(data, n_users, n_items, csr):
    if csr:
        out_data = csr_matrix(
            (np.ones(data.shape[0], dtype=np.float32), (data.user_id.values, data.item_id.values)), (n_users, n_items)
        )
    else:
        out_data = np.zeros((n_users, n_items), dtype=np.float32)
        for row in data.itertuples(index=False):
            out_data[row.user_id, row.item_id] = 1
    return out_data


def save_df(PROCESSED_DATA_DIR, df, filename, index=False, sep=","):
    df.to_csv(os.path.join(PROCESSED_DATA_DIR, filename), sep=sep, index=index)


def split_data(df, test_frac, on_user=True):
    if on_user == True:
        return split_on_user(df, test_frac)
    else:
        return split_on_item(df, test_frac)


def split_on_user(df, test_frac):
    # Identify properties and prepare split indices
    user_n_ratings = df.user_id.value_counts()
    n_users = user_n_ratings.shape[0]

    unique_users = user_n_ratings.index.values

    val_split_ind = int((1 - test_frac) * n_users)
    test_split_ind = int((1 - test_frac / 2) * n_users)

    user_idx = np.arange(n_users)
    np.random.shuffle(user_idx)

    # Identify users in each set
    train_users = unique_users[user_idx[:val_split_ind]]
    val_users = unique_users[user_idx[val_split_ind:test_split_ind]]
    test_users = unique_users[user_idx[test_split_ind:]]

    # Extract data from users
    train_df = df.loc[df["user_id"].isin(train_users), ["user_id", "item_id"]]
    val_df = df.loc[df["user_id"].isin(val_users), ["user_id", "item_id"]]
    test_df = df.loc[df["user_id"].isin(test_users), ["user_id", "item_id"]]

    return train_df, val_df, test_df


def split_on_item(df, test_frac, valtest=True):
    df_grouped_by_user = df.groupby("user_id")
    tr_list, val_list, te_list = [], [], []

    for i, (_, group) in enumerate(df_grouped_by_user):
        n_items_u = len(group)

        # Split the items of users that have rated more than 5 items
        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype="bool")
            n_valtest_items = int(test_frac * n_items_u)
            valtest_inds = np.random.choice(n_items_u, size=n_valtest_items, replace=False).astype("int64")
            idx[valtest_inds] = True

            tr_list.append(group[np.logical_not(idx)])
            if not valtest:
                te_list.append(group[idx])
                continue

            # Do a coin-toss in cases where the test items cannot be evenly split between val and test
            test_inds_start = n_valtest_items // 2 + (np.random.randint(0, 2) if n_valtest_items % 2 == 1 else 0)
            val_inds = valtest_inds[:test_inds_start]
            test_inds = valtest_inds[test_inds_start:]
            val_list.append(group.iloc[val_inds])
            te_list.append(group.iloc[test_inds])
        else:
            tr_list.append(group)

    df_tr = pd.concat(tr_list)
    df_val = pd.concat(val_list) if valtest else None
    df_te = pd.concat(te_list)

    return df_tr, df_val, df_te


def filter_and_reindex_items(train_df, val_df, test_df):
    # We can only process items seen in the training set
    unique_train_items = train_df["item_id"].unique()
    reindex_item = dict((iid, i) for (i, iid) in enumerate(unique_train_items))
    # reverse_map = dict((v, k) for k, v in reindex_item.items())

    def reindex(df, unique_train_items, item_map):
        # Filter
        df_copy = df.copy()
        df_copy = df_copy[df_copy["item_id"].isin(unique_train_items)]
        # Reindex
        df_copy["item_id"] = df_copy["item_id"].map(lambda x: item_map[x])
        return df_copy

    train_df = reindex(train_df, unique_train_items, reindex_item)
    val_df = reindex(val_df, unique_train_items, reindex_item)
    test_df = reindex(test_df, unique_train_items, reindex_item)
    return train_df, val_df, test_df  # , reverse_map


class RecDataset(Dataset):
    def __init__(self, x, y, s, device):
        self.x = torch.tensor(x, dtype=torch.float32, device=device)
        self.y = y
        self.s = torch.tensor(s, dtype=torch.float32, device=device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        y_val = self.y[idx] if self.y is not None else np.array([0], dtype=np.float32)
        return self.x[idx], y_val, self.s[idx]


class CsrDataset(Dataset):
    def __init__(self, x, y, s):
        self.x = x
        self.y = y
        self.s = s

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        y_val = self.y[idx].toarray().squeeze() if self.y is not None else np.array([0], dtype=np.float32)
        return self.x[idx].toarray().squeeze(), y_val, self.s[idx]


class BinaryDataset(Dataset):
    def __init__(self, x, y, device):
        self.x = torch.tensor(x, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def init_dataloader(x, y, s, device, batch_size, csr):
    if csr:
        dataset = CsrDataset(x, y, s)
    else:
        dataset = RecDataset(x, y, s, device)
    return DataLoader(dataset, batch_size, shuffle=True)
