import pandas as pd
from enums import DataProc
from checks import check_expected_values
from read_write import read_fitted_data, write_fitted_data
from logger import create_logger
import numpy as np


class Imputer(object):
    """
    Methods for imputing features according to json recipe

    Parameters
    ----------
    method: dict
        Imputer methods and attributes.
    data: pd.DataFrame
        Data to transform
    flag: bool
        If True, creates flag field for imputed observations

    Examples
     ----------
     >>> data = pd.DataFrame(np.random.randint(1,100, 1000), columns=['My_feature'])
     >>> methods = {"type": "replace", "method": "medium"}
     >>> my_imputer = Imputer(method=methods, data=data, flag=True)
     >>> my_imputer.run()
    """

    def __init__(self, method: dict, data: pd.DataFrame, flag: bool):
        self.method = method
        self.data = data
        self.flag = flag
        self.feature_name = data.columns[0]
        self.results = pd.DataFrame()
        self.log = create_logger(name="Imputer")

    def run(self):
        self.log.info(
            "Running {} impute for feature {}...".format(
                self.method[DataProc.METHOD.value], self.feature_name
            )
        )
        if self.flag:
            self.data[
                "{}_{}".format(self.feature_name, DataProc.IMPUTATION_FLAG_SUFFIX.value)
            ] = 0
            self.data.loc[
                self.data[self.feature_name].isna(),
                "{}_{}".format(
                    self.feature_name, DataProc.IMPUTATION_FLAG_SUFFIX.value
                ),
            ] = 1
        if self.method[DataProc.TYPE.value] == DataProc.REPLACE.value:
            self.replace_imputer()
        if self.method[DataProc.TYPE.value] == DataProc.AGGREGATE.value:
            if self.method[DataProc.FIT.value] == 1:
                self.imputed_values = {}
                self.fit_aggregate_imputer()
                write_fitted_data(
                    data=self.imputed_values,
                    data_path=DataProc.PATH.value,
                    feature_name=self.feature_name,
                    file_type="json",
                )
            if self.method[DataProc.FIT.value] == 0:
                self.imputed_values = read_fitted_data(
                    data_path=DataProc.PATH.value,
                    feature_name=self.feature_name,
                    file_type="json",
                )
                self.transform_aggregate_imputer()
            check_expected_values(
                field_values=self.results[self.feature_name].unique(),
                expected_values=self.data[self.feature_name],
                field_name=self.feature_name,
                operation="Imputer",
            )
        self.log.info(
            "{} impute for feature {} complete...".format(
                self.method[DataProc.METHOD.value], self.feature_name
            )
        )

    def replace_imputer(self):
        self.log.info(
            "Running replace imputer for feature {}...".format(self.feature_name)
        )
        self.results[self.feature_name] = self.data[self.feature_name].fillna(
            value=self.method[DataProc.METHOD.value]
        )

    def fit_aggregate_imputer(self):
        self.log.info(
            "Running fit transform for aggregate imputer for feature {}...".format(
                self.feature_name
            )
        )
        if self.method[DataProc.METHOD.value] == DataProc.MEDIAN.value:
            self.imputed_values[DataProc.MEDIAN.value] = self.data[
                self.feature_name
            ].median()
            self.results[self.feature_name] = self.data[self.feature_name].fillna(
                value=self.imputed_values[DataProc.MEDIAN.value]
            )

        elif self.method[DataProc.METHOD.value] == DataProc.MODE.value:
            self.imputed_values[DataProc.MODE.value] = (
                self.data[self.feature_name].mode().values[0]
            )
            self.results[self.feature_name] = self.data[self.feature_name].fillna(
                value=self.imputed_values[DataProc.MODE.value]
            )
        self.log.info(
            "Fit and transform for aggregate imputer for feature {} complete...".format(
                self.feature_name
            )
        )

    def transform_aggregate_imputer(self):
        self.log.info(
            "Running transform for aggregate imputer for feature {}...".format(
                self.feature_name
            )
        )
        if self.method[DataProc.METHOD.value] == DataProc.MEDIAN.value:
            self.results[self.feature_name] = self.data[self.feature_name].fillna(
                value=self.imputed_values[DataProc.MEDIAN.value]
            )
        elif self.method[DataProc.METHOD.value] == DataProc.MODE.value:
            self.results[self.feature_name] = self.data[self.feature_name].fillna(
                value=self.imputed_values[DataProc.MODE.value]
            )
        self.log.info(
            "Transform for aggregate imputer for feature {} complete...".format(
                self.feature_name
            )
        )
