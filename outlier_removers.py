import pandas as pd
from enums import DataProc
from checks import check_nans, check_numeric
from read_write import read_fitted_data, write_fitted_data
from logger import create_logger


class OutlierRemover(object):
    """
    Methods for removing features outliers according to json recipe

    Parameters
    ----------
    method: dict
        Outlier removal methods and attributes
    data: pd.DataFrame
        Data to perform outlier removal on

     Examples
     ----------
     >>> data = pd.DataFrame(np.random.randint(1,100, 1000), columns=['My_feature'])
     >>> methods = {"method": "percentile", "min": 0.05, "max": 0.95, "fit": 0, "path": "lookups/Feature_4_outlier.json"}
     >>> outlier_remover = OutlierRemover(method=methods, data=data)
     >>> outlier_remover.run()
    """

    def __init__(self, method: dict, data: pd.DataFrame):
        self.method = method
        self.data = data
        self.feature_name = data.columns[0]
        self.results = pd.DataFrame()
        self.log = create_logger(name="Outlier_remover")

    def run(self):
        self.log.info(
            "Running {} outlier removal for feature {}...".format(
                self.method[DataProc.METHOD.value], self.feature_name
            )
        )
        if self.method[DataProc.METHOD.value] == DataProc.PERCENTILE.value:
            if self.method[DataProc.FIT.value] == 1:
                self.imputed_values = {}
                self.percentile_remover_fit()
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
                self.percentile_remover_transform()

            check_nans(
                data=self.results[self.feature_name],
                field_name=self.feature_name,
                operation="Outlier Remover",
            )
            check_numeric(
                data=self.results[self.feature_name],
                field_name=self.feature_name,
                operation="Outlier Remover",
            )
        self.log.info(
            "{} outlier removal for feature {} complete...".format(
                self.method[DataProc.METHOD.value], self.feature_name
            )
        )

    def percentile_remover_fit(self):
        self.log.info(
            "Running percentile remover fit transform for feature {}...".format(
                self.feature_name
            )
        )
        self.imputed_values[DataProc.LOWER_PCT.value] = self.data[
            self.feature_name
        ].quantile(self.method[DataProc.MIN.value])
        self.imputed_values[DataProc.UPPER_PCT.value] = self.data[
            self.feature_name
        ].quantile(self.method[DataProc.MAX.value])
        self.results[self.feature_name] = self.data[self.feature_name].clip(
            lower=self.imputed_values[DataProc.LOWER_PCT.value],
            upper=self.imputed_values[DataProc.UPPER_PCT.value],
        )
        self.log.info(
            "Percentile remover fit transform for feature {} complete...".format(
                self.feature_name
            )
        )

    def percentile_remover_transform(self):
        self.log.info(
            "Running percentile remover transform for feature {}...".format(
                self.feature_name
            )
        )
        self.results[self.feature_name] = self.data[self.feature_name].clip(
            lower=self.imputed_values[DataProc.LOWER_PCT.value],
            upper=self.imputed_values[DataProc.UPPER_PCT.value],
        )
        self.log.info(
            "Percentile remover transform for feature {} complete...".format(
                self.feature_name
            )
        )
