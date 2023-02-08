import pandas as pd
from logger import create_logger
import json

log = create_logger("rw")


def write_fitted_data(
    data: pd.DataFrame or dict, data_path: str, feature_name: str, file_type: str
) -> None:
    log.info("Saving fit data for feature {}...".format(feature_name))
    if file_type == "json":
        with open(data_path, "w") as f:
            json.dump(data, f)

    if file_type == "csv":
        data.to_csv(data_path)
    log.info("Saving fit data for feature {} complete...".format(feature_name))


def read_fitted_data(
    data_path: str, feature_name: str, file_type: str
) -> pd.DataFrame or dict:
    log.info("Loading fit data for feature {}...".format(feature_name))
    data = None
    if file_type == "json":
        with open(data_path) as f:
            data = json.load(f)

    elif file_type == "csv":
        data = pd.read_csv(data_path, index_col=0)

    log.info("Loading fit data for feature {} complete...".format(feature_name))
    return data
