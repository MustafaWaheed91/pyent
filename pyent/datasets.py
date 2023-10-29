from typing import Tuple, Any

import pandas as pd
from pandas import MultiIndex, DataFrame
from sklearn.model_selection import train_test_split
import recordlinkage as rl
from recordlinkage.datasets import load_febrl4


def generate_febrl_data(block_cols: Tuple[str] = ("given_name", "surname"), init_seed: int = 42) -> pd.DataFrame:
    """
        Generate person entity duplicates. This returns a tuple of 2 data frames and an index i.e.
        (pandas.core.frame.DataFrame, pandas.core.frame.DataFrame, pandas.core.indexes.multi.MultiIndex)

        :param block_cols: The column name in the data set that you'd want to block on
        :param init_seed: Integer representing the seed initialization for
            Random Variables :default 42

        :returns (pandas.DataFrame): this is the concatenated dataframe of candidate pairs
    """
    # positive class for target
    # noinspection PyTupleAssignmentBalance
    dfl, dfr, links = load_febrl4(return_links=True)
    df_links = pd.DataFrame(index=links)

    dfl.reset_index(drop=False, inplace=True)
    dfl.rename(columns={"rec_id": "rec_idL"}, inplace=True)
    dfr.reset_index(drop=False, inplace=True)
    dfr.rename(columns={"rec_id": "rec_idR"}, inplace=True)
    df_links.reset_index(drop=False, inplace=True)
    df_links.rename(
        columns={
            "level_0": "rec_idL",
            "level_1": "rec_idR"
        },
        inplace=True
    )

    df_lr_links = df_links.merge(
        right=dfl,
        left_on="rec_idL",
        right_on="rec_idL",
        how='left',
    ).merge(
        right=dfr,
        left_on="rec_idR",
        right_on="rec_idR",
        how='left',
        suffixes=("_l", "_r")
    )

    df_lr_links["labels"] = "match"

    # negative class for target
    dfa, dfb = load_febrl4()

    indexer = rl.Index()
    indexer.block(block_cols[1])
    candidate_links: MultiIndex = indexer.index(dfa, dfb)

    df_clinks: DataFrame = pd.DataFrame(index=candidate_links)
    df_clinks.reset_index(inplace=True, drop=False)
    df_clinks.rename(
        columns={
            "rec_id_1": "rec_idL",
            "rec_id_2": "rec_idR"
        },
        inplace=True
    )

    df_clinks["rec_idL"] = df_clinks["rec_idL"].astype(str)
    df_clinks["rec_idR"] = df_clinks["rec_idR"].astype(str)
    df_clinks["org_flag"] = df_clinks.loc[:, "rec_idL"].apply(lambda x: "-".join(x.split("-")[:-1]))
    df_clinks["dup_flag"] = df_clinks.loc[:, "rec_idR"].apply(lambda x: "-".join(x.split("-")[:-2]))
    df_clinks = df_clinks.loc[df_clinks["org_flag"] != df_clinks["dup_flag"], ["rec_idL", "rec_idR"]]

    df_lr_clinks = df_clinks.merge(
        right=dfl,
        left_on="rec_idL",
        right_on="rec_idL",
        how='left',
    ).merge(
        right=dfr,
        left_on="rec_idR",
        right_on="rec_idR",
        how='left',
        suffixes=("_l", "_r")
    )

    df_lr_clinks["labels"] = "no_match"
    return pd.concat([df_lr_clinks, df_lr_links]).sample(frac=1, random_state=init_seed).reset_index(drop=True)


def remove_nan(master_df: pd.DataFrame) -> pd.DataFrame:
    """
        removes the rows from dataset if NaN in principal cols

        :param master_df: the febrl dataframe generated

        :returns: pd.DataFrame
    """
    print(f"Before Dropping NaN's shape of data is {master_df.shape}")
    drop_check_subset = [
        "given_name_l",
        "surname_l",
        "street_number_l",
        "address_1_l",
        "address_2_l",
        "postcode_l",
        "given_name_r",
        "surname_r",
        "street_number_r",
        "address_1_r",
        "address_2_r",
        "postcode_r",
    ]
    master_df_nona = master_df.dropna(subset=drop_check_subset)
    print(f"After Dropping NaN's shape of data is {master_df_nona.shape}")
    return master_df_nona


def train_test_validate_strat_split(features, targets, test_size: float = 0.1, validate_size: float = 0.2) -> Tuple:
    """
        Split data into dev (i.e. train and validate) and a test set to hold out.

        :param features:
        :param targets:
        :param test_size:
        :param validate_size:

        :return A tuple of dataframes of train test and validate:
        :rtype: Tuple
    """
    # Get test sets
    features_train, features_test, targets_train, targets_test = train_test_split(
        features,
        targets,
        stratify=targets,
        test_size=test_size
    )

    # Run train_test_split again to get train and validate sets
    post_split_validate_size = validate_size / (1 - test_size)
    features_train, features_validate, targets_train, targets_validate = train_test_split(
        features_train,
        targets_train,
        stratify=targets_train,
        test_size=post_split_validate_size
    )
    return features_train, features_test, features_validate, targets_train, targets_test, targets_validate


def sample_xy(X: pd.DataFrame, y: pd.Series, num: int | None = None) -> Tuple[Any, pd.DataFrame]:
    """
        sample any df or series by record count and series val

        :param num: this is an integer that
        :param y: this is the column representing the target
        :param X: this is the dataframe of features

        :returns Tuple[pandas.DataFrame, pandas.DataFrame]: this is the 2 
            dataframes of features and targets for training.
    """
    x_df = X.copy()
    y_df = pd.DataFrame(index=y.index, data=y).copy()

    lab_lst = []
    lab_set = list(set(y.tolist()))
    for lab in lab_set:
        lab_lst.append(y_df[y_df[y_df.columns[0]] == lab].sample(n=num))

    y_sample_df = pd.concat(lab_lst, ignore_index=False)
    x_sample_df = x_df.loc[y_sample_df.index, :]

    return x_sample_df, y_sample_df


if __name__ == "__main__":
    remove_nan(generate_febrl_data())
