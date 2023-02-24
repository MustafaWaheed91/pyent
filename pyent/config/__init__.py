import os
import yaml
import json

import pyent


def get_config():
    """reads yaml config file and returns dict of contents
    """
    try:
        cfg_base_path = os.path.dirname(pyent.config.__file__)
        cfg_file_path = os.path.join(cfg_base_path, "config.yaml")

        with open(cfg_file_path, 'r') as fp:
            params = yaml.safe_load(fp)

        return params
    except Exception as e:
        print(e)
        raise
