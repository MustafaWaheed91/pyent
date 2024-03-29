from typing import Optional, List, Dict

import pandas as pd


def generate_textual_features(X: pd.DataFrame, lr_dict: Optional[Dict[str,List[str]]] = None, ) -> pd.DataFrame:
    """collect text attributes into two sentences
        :params X (pd.DataFrame): 
            a dataframe for features
        :param lr_dict (dict): 
            a dict with keys sentence_l and sentence_r and values being a list of column names (default: None)

        :returns pandas.DataFrame:

    .. Example Usage:
    ----- example argument ------
    lr_dict = {
        "sentence_l": ["given_name_l", "surname_l", "street_number_l"],
        "sentence_r": ["given_name_r", "surname_r", "street_number_r"]
    }
    """
    X_ = X.copy()
    lr_dict = {
        "sentence_l": ["given_name_l", "surname_l", "street_number_l", "address_1_l", "address_2_l", "postcode_l"], 
        "sentence_r": ["given_name_r", "surname_r", "street_number_r", "address_1_r", "address_2_r", "postcode_r"]
    } if lr_dict is None else lr_dict
    X_[f"{list(lr_dict.keys())[1]}"] = X_.loc[:, list(lr_dict.values())[1]].applymap(str).agg(" ".join, axis=1)
    X_[f"{list(lr_dict.keys())[0]}"] = X_.loc[:, list(lr_dict.values())[0]].applymap(str).agg(" ".join, axis=1)
    return X_.loc[:, list(lr_dict.keys())]
