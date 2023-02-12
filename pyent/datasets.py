import pandas as pd
from sklearn.model_selection import train_test_split
import recordlinkage as rl
from recordlinkage.datasets import load_febrl4


def generate_febrl_data(block_col: str ="surname", init_seed: int =0) -> pd.DataFrame:
    """geerate person entity duplicates

    args:
        block_col (str): the column name in the data set that you'd want to block on   
    """
    # postive class for target
    dfL, dfR, df_links = load_febrl4(return_links=True)
    df_links = pd.DataFrame(index=df_links)

    dfL.reset_index(drop=False, inplace=True)
    dfL.rename(columns={"rec_id": "rec_idL"}, inplace=True)
    dfR.reset_index(drop=False, inplace=True)
    dfR.rename(columns={"rec_id": "rec_idR"}, inplace=True)
    df_links.reset_index(drop=False, inplace=True)
    df_links.rename(
        columns={
            "level_0": "rec_idL", 
            "level_1": "rec_idR"
        }, 
        inplace=True
    )

    df_LR_links = df_links.merge(
        right=dfL,
        left_on="rec_idL",
        right_on="rec_idL",
        how='left',
    ).merge(
        right=dfR,
        left_on="rec_idR",
        right_on="rec_idR",
        how='left',
        suffixes=("_l", "_r")
    )

    df_LR_links["labels"] = "match"

    # negative class for target
    dfA, dfB = load_febrl4()

    indexer = rl.Index()
    indexer.block(block_col)
    candidate_links = indexer.index(dfA, dfB)

    df_clinks = pd.DataFrame(index=candidate_links)
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

    df_LR_clinks = df_clinks.merge(
        right=dfL,
        left_on="rec_idL",
        right_on="rec_idL",
        how='left',
    ).merge(
        right=dfR,
        left_on="rec_idR",
        right_on="rec_idR",
        how='left',
        suffixes=("_l", "_r")
    )

    df_LR_clinks["labels"] = "no_match"
    return pd.concat([df_LR_clinks, df_LR_links]).sample(frac=1, random_state=init_seed).reset_index(drop=True)


def remove_nan(master_df: pd.DataFrame) -> pd.DataFrame:
    """removes the rows from dataset if NaN in principal cols
    
    params:
        master_df (pd.DataFrame): the febrl datafrane geberated
    return:
        pd.DataFrame
    """
    print(f"Before Droping NaN's shape of data is {master_df.shape}")
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
    print(f"After Droping NaN's shape of data is {master_df_nona.shape}")
    return master_df_nona


def train_test_validate_stratified_split(features, targets, test_size=0.1, validate_size=0.2):
    """split data into dev and test set
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


if __name__ == "__main__":
    remove_nan(generate_febrl_data())
