import os
import pandas as pd
import numpy as np
from logger import create_logger

log = create_logger("Check_json_log")

"""
Unit test of config.jsons and confirming numeric data.
"""


def config_checker(config: dict):
    FEATURE_TYPES = ["categorical", "discrete", "continuous"]
    EXPECTED_MISSING = ["nan"]
    IMPUTATION_KEYS = ["type", "method"]
    BINNING_KEYS = ["type", "ascending"]
    EXPECTED_VALUES_KEYS = ["min", "max"]
    OUTLIER_REMOVAL_KEYS = ["method", "min", "max"]
    TRANSFORMATION_KEYS = ["method", "groupby", "target_field"]

    for feature_name, feature_values in config.items():
        if feature_values["type"] not in FEATURE_TYPES:
            log.error(
                "{} has invalid feature type {} (Options: {})".format(
                    feature_name, feature_values["type"], FEATURE_TYPES
                )
            )
            raise ValueError(
                "{} has invalid feature type {} (Options: {})".format(
                    feature_name, feature_values["type"], FEATURE_TYPES
                )
            )
        if feature_values["expected_missing"] not in EXPECTED_MISSING:
            log.error(
                "{} has invalid expected missing values {} (Options: {})".format(
                    feature_name, feature_values["type"], EXPECTED_MISSING
                )
            )
            raise ValueError(
                "{} has invalid expected missing values {} (Options: {})".format(
                    feature_name, feature_values["type"], EXPECTED_MISSING
                )
            )
        if type(feature_values["imputation"]) is dict:
            for k in IMPUTATION_KEYS:
                if k not in feature_values["imputation"]:
                    log.error(
                        "{} key is missing in imputation setting for feature {}. Expects {}".format(
                            k, feature_name, IMPUTATION_KEYS
                        )
                    )
                    raise ValueError(
                        "{} key is missing in imputation setting for feature {}. Expects {}".format(
                            k, feature_name, IMPUTATION_KEYS
                        )
                    )
        elif type(feature_values["imputation"]) is int:
            if feature_values["imputation"] != 0:
                log.error(
                    "{} is not a valid imputation option".format(
                        str(feature_values["imputation"])
                    )
                )
                raise ValueError(
                    "{} is not a valid imputation option".format(
                        str(feature_values["imputation"])
                    )
                )

        if feature_values["flag_imputed"] not in [0, 1]:
            log.error(
                "Flag imputed setting is invalid for {} Options: {}.".format(
                    feature_name, "[0, 1]"
                )
            )
            raise ValueError(
                "Flag imputed setting is invalid for {} Options: {}.".format(
                    feature_name, "[0, 1]"
                )
            )

        if type(feature_values["expected_values"]) is dict:
            for k in EXPECTED_VALUES_KEYS:
                if k not in feature_values["expected_values"]:
                    log.error(
                        "{} key is missing in expected values for feature {}. Expects {}".format(
                            k, feature_name, EXPECTED_VALUES_KEYS
                        )
                    )
                    raise ValueError(
                        "{} key is missing in expected values for feature {}. Expects {}".format(
                            k, feature_name, EXPECTED_VALUES_KEYS
                        )
                    )
                elif type(feature_values["expected_values"][k]) is not int:
                    log.error(
                        "Expected value key {} is not an integer for feature {}".format(
                            k, feature_name
                        )
                    )
                    raise ValueError(
                        "Expected value key {} is not an integer for feature {}".format(
                            k, feature_name
                        )
                    )

        if type(feature_values["expected_values"]) is str:
            if not os.path.isfile(feature_values["expected_values"]):
                log.error(
                    "{} is not a valid file path for expected values for feature {}".format(
                        feature_values["expected_values"], feature_name
                    )
                )
                raise FileNotFoundError(
                    "{} is not a valid file path for expected values for feature {}".format(
                        feature_values["expected_values"], feature_name
                    )
                )

        if type(feature_values["expected_values"]) is list:
            if len(feature_values["expected_values"]) < 2:
                log.error(
                    "{} error: Requires at least 2 expected values, found {}".format(
                        feature_name, str(len(feature_values["expected_values"]))
                    )
                )
                raise ValueError(
                    "{} error: Requires at least 2 expected values, found {}".format(
                        feature_name, str(len(feature_values["expected_values"]))
                    )
                )

        if type(feature_values["outlier_removal"]) is dict:
            for k in OUTLIER_REMOVAL_KEYS:
                if k not in feature_values["outlier_removal"]:
                    log.error(
                        "{} key is missing in outlier removal for feature {}. Expects {}".format(
                            k, feature_name, OUTLIER_REMOVAL_KEYS
                        )
                    )
                    raise ValueError(
                        "{} key is missing in outlier removal for feature {}. Expects {}".format(
                            k, feature_name, OUTLIER_REMOVAL_KEYS
                        )
                    )
        elif type(feature_values["outlier_removal"]) is int:
            if feature_values["outlier_removal"] != 0:
                log.error(
                    "{} is not a valid outlier removal option for feature {}".format(
                        str(feature_values["outlier_removal"]), feature_name
                    )
                )
                raise ValueError(
                    "{} is not a valid outlier removal option for feature {}".format(
                        str(feature_values["outlier_removal"]), feature_name
                    )
                )

        if type(feature_values["transformation"]) is dict:
            for k in TRANSFORMATION_KEYS:
                if k not in feature_values["transformation"]:
                    log.error(
                        "{} key is missing in transformation for feature {}. Expects {}".format(
                            k, feature_name, TRANSFORMATION_KEYS
                        )
                    )
                    raise ValueError(
                        "{} key is missing in transformation for feature {}. Expects {}".format(
                            k, feature_name, TRANSFORMATION_KEYS
                        )
                    )
        elif type(feature_values["transformation"]) is int:
            if feature_values["transformation"] != 0:
                log.error(
                    "{} is not a valid transformation option for feature {}".format(
                        str(feature_values["transformation"]), feature_name
                    )
                )
                raise ValueError(
                    "{} is not a valid transformation option for feature {}".format(
                        str(feature_values["transformation"]), feature_name
                    )
                )

        if type(feature_values["binning"]) is dict:
            for k in BINNING_KEYS:
                if k not in feature_values["binning"]:
                    log.error(
                        "{} key is missing in binning for feature {}. Expects {}".format(
                            k, feature_name, BINNING_KEYS
                        )
                    )
                    raise ValueError(
                        "{} key is missing in binning for feature {}. Expects {}".format(
                            k, feature_name, BINNING_KEYS
                        )
                    )
        elif type(feature_values["binning"]) is int:
            if feature_values["binning"] != 0:
                log.error(
                    "{} is not a valid binning option for feature {}".format(
                        str(feature_values["binning"]), feature_name
                    )
                )
                raise ValueError(
                    "{} is not a valid binning option for feature {}".format(
                        str(feature_values["binning"]), feature_name
                    )
                )
        elif type(feature_values["binning"]) is list:
            if len(feature_values["binning"]) < 3:
                log.error(
                    "{} error: Binning by list requires at least 3 data points, found {}".format(
                        feature_name, str(len(feature_values["binning"]))
                    )
                )
                raise ValueError(
                    "{} error: Binning by list requires at least 3 data points, found {}".format(
                        feature_name, str(len(feature_values["binning"]))
                    )
                )


def check_fields_exist(found_fields: list, expected_fields: list):
    for field in expected_fields:
        if field not in found_fields:
            log.error("{} field could not be found in data input fields".format(field))
            raise KeyError(
                "{} field could not be found in data input fields".format(field)
            )

    log.info("Field name check for complete...")


def check_expected_values(
    field_values: list, expected_values: list, field_name: str, operation: str
):
    remain = list(set(field_values) - set(expected_values))
    remain = [x for x in remain if str(x) != str(float(np.nan))]
    if len(remain) > 0:
        log.error(
            "{} contains {} unexpected values following {} operation".format(
                field_name, str(len(remain)), operation
            )
        )
        raise ValueError(
            "{} contains {} unexpected values following {} operation".format(
                field_name, str(len(remain)), operation
            )
        )
    else:
        log.info(
            "Expected values check for feature {} following operation {} complete...".format(
                field_name, operation
            )
        )


def check_nans(
    data: pd.Series, field_name: str, operation: str, raise_flag: bool = True
):
    if data.isna().sum() > 0:
        if raise_flag:
            log.error(
                "{} contains nans following operation {}.".format(field_name, operation)
            )
            raise ValueError(
                "{} contains nans following operation {}.".format(field_name, operation)
            )
    else:
        log.info(
            "NaN check for feature {} following operation {} complete...".format(
                field_name, operation
            )
        )


def check_numeric(
    data: pd.Series, field_name: str, operation: str, raise_flag: bool = True
):
    if not pd.api.types.is_numeric_dtype(data):
        if raise_flag:
            log.error(
                "{} contains non-numeric values following operation {}.".format(
                    field_name, operation
                )
            )
            raise ValueError(
                "{} contains non-numeric values following operation {}.".format(
                    field_name, operation
                )
            )

    else:
        log.info(
            "Numeric check for feature {} following operation {} complete...".format(
                field_name, operation
            )
        )
