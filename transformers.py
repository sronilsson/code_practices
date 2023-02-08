import pandas as pd
from enums import DataProc
from checks import check_nans, check_numeric
import numpy as np
from read_write import read_fitted_data, write_fitted_data
from logger import create_logger


class Transformers(object):
    """
    Methods for transforming features according to json recipe

    Parameters
    ----------
    method: dict
        Transformation methods and attributes.
    data: pd.DataFrame
        Data to be transformed

    Examples
    ----------
    >>> data = pd.DataFrame(np.random.randint(1,100, 1000), columns=['My_feature'])
    >>> methods = {"method": "z_transform", "target_field": 0, "groupby": "Feature_5", "fit": 0, "path": "lookups/Feature_4_transformation.json"}
    >>> my_transformer = Transformers(methods=methods, data=data)
    >>> my_transformer.run()
    """

    def __init__(self, method: dict, data: pd.DataFrame):
        self.method = method
        self.data = data
        self.feature_name = data.columns[0]
        self.results = pd.DataFrame()
        self.log = create_logger(name="Transformer")

    def run(self):
        self.log.info(
            "Running {} transformation for feature {}...".format(
                self.method[DataProc.METHOD.value], self.feature_name
            )
        )
        if self.method[DataProc.METHOD.value] == DataProc.Z_TRANSFORM.value:
            if self.method[DataProc.FIT.value] == 1:
                self.transformed_values = {}
                self.z_value_fit()
                write_fitted_data(
                    data=self.transformed_values,
                    data_path=DataProc.PATH.value,
                    feature_name=self.feature_name,
                    file_type="csv",
                )
            if self.method[DataProc.FIT.value] == 0:
                self.transformed_values = read_fitted_data(
                    DataProc.PATH.value, feature_name=self.feature_name, file_type="csv"
                )
                self.z_value_transform()

        if self.method[DataProc.METHOD.value] == DataProc.MEAN.value:
            if self.method[DataProc.FIT.value] == 1:
                self.transformed_values = {}
                self.field_mean_fit()
                write_fitted_data(
                    data=self.transformed_values,
                    data_path=DataProc.PATH.value,
                    feature_name=self.feature_name,
                    file_type="json",
                )
            if self.method[DataProc.FIT.value] == 0:
                self.transformed_values = read_fitted_data(
                    DataProc.PATH.value,
                    feature_name=self.feature_name,
                    file_type="json",
                )
                self.field_mean_transform()

            check_nans(
                data=self.results[self.feature_name],
                field_name=self.feature_name,
                operation="Transform",
                raise_flag=False,
            )

            check_numeric(
                data=self.results[self.feature_name],
                field_name=self.feature_name,
                operation="Transform",
            )
            self.log.info(
                "{} transformation for feature {} complete...".format(
                    self.method[DataProc.METHOD.value], self.feature_name
                )
            )

    def z_value_fit(self):
        self.log.info(
            "Performing z value fit_transform for feature {}...".format(
                self.feature_name
            )
        )
        self.transformed_values = (
            self.data.groupby([DataProc.GROUPBY.value])[self.feature_name]
            .agg(["mean", "std"])
            .reset_index()
        )
        self.data = pd.merge(
            self.data, self.transformed_values, on=DataProc.GROUPBY.value, how="left"
        )
        self.data.columns = [self.feature_name, DataProc.GROUPBY.value, "mean", "std"]
        self.results[self.feature_name] = (
            self.data[self.feature_name] - self.data["mean"]
        ) / self.data["std"]
        self.log.info(
            "Z value fit_transform for feature {} complete...".format(self.feature_name)
        )

    def z_value_transform(self):
        self.log.info(
            "Performing z value transform for feature {}...".format(self.feature_name)
        )
        self.data = pd.merge(
            self.data, self.transformed_values, on=DataProc.GROUPBY.value, how="left"
        )
        self.data.columns = [self.feature_name, DataProc.GROUPBY.value, "mean", "std"]
        self.results[self.feature_name] = (
            self.data[self.feature_name] - self.data["mean"]
        ) / self.data["std"]
        self.log.info(
            "Z value transform for feature {} complete...".format(self.feature_name)
        )

    def field_mean_fit(self):
        self.log.info(
            "Performing field mean fit for feature {}...".format(self.feature_name)
        )
        self.transformed_values[DataProc.FIELD_MEAN_TRANSFORMER.value] = (
            self.data.groupby([DataProc.GROUPBY.value])[DataProc.TARGET_FIELD.value]
            .mean()
            .to_dict()
        )
        self.data[self.feature_name] = (
            self.data.groupby([DataProc.GROUPBY.value])[DataProc.TARGET_FIELD.value]
            .mean()
            .reset_index(drop=True)
        )
        self.results[self.feature_name] = self.data[self.feature_name]
        self.log.info(
            "Field mean fit for feature {} complete...".format(self.feature_name)
        )

    def field_mean_transform(self):
        self.log.info(
            "Performing field mean transform for feature {}...".format(
                self.feature_name
            )
        )
        self.results[self.feature_name] = self.data[self.feature_name].map(
            self.transformed_values[DataProc.FIELD_MEAN_TRANSFORMER.value]
        )
        self.log.info(
            "Field mean transform for feature {} complete...".format(self.feature_name)
        )
