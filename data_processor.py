import pandas as pd
from imputers import Imputer
from outlier_removers import OutlierRemover
from transformers import Transformers
from binners import Binners
from checks import config_checker, check_fields_exist, check_expected_values
from enums import DataProc
from copy import deepcopy
import json
import os


class DataProcessor(object):
    """
    Main class for data processing according to json recipe

    Parameters
    ----------
    config_path: str
        path to config json format

    Examples
    ----------
    >>> data_processor = DataProcessor(config_path='configs/config.json')
    >>> data_processor.read_data(data_path='data/data_1000_train.csv')
    >>> results = data_processor.transform()
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = json.load(f)
        config_checker(config=self.config)

    def read_data(self, data_path: str):
        if not os.path.isfile(data_path):
            raise FileNotFoundError("{} is not a valid file path".format(data_path))
        self.data_df = pd.read_csv(data_path, index_col=0)
        check_fields_exist(
            found_fields=self.data_df.columns, expected_fields=self.config.keys()
        )

    def transform(self):
        self.results_lst = []
        for name, methods in self.config.items():
            feature_data = deepcopy(self.data_df[name])
            if type(methods[DataProc.EXPECTED_VALUES.value]) is dict:
                expected_values = list(
                    range(
                        methods[DataProc.EXPECTED_VALUES.value][DataProc.MIN.value],
                        methods[DataProc.EXPECTED_VALUES.value][DataProc.MAX.value] + 1,
                    )
                )
                expected_values = [float(x) for x in expected_values]
            if type(methods[DataProc.EXPECTED_VALUES.value]) is list:
                expected_values = methods[DataProc.EXPECTED_VALUES.value]
            if type(methods[DataProc.EXPECTED_VALUES.value]) is str:
                expected_values = list(
                    pd.read_csv(methods[DataProc.EXPECTED_VALUES.value], header=None)[0]
                )
            check_expected_values(
                field_values=feature_data.unique(),
                expected_values=expected_values,
                field_name=name,
                operation="READ IN",
            )

            if methods[DataProc.IMPUTATION.value] != 0:
                imputer = Imputer(
                    method=methods[DataProc.IMPUTATION.value],
                    data=pd.DataFrame(feature_data),
                    flag=bool(methods[DataProc.FLAG_IMPUTED.value]),
                )
                imputer.run()
                feature_data = imputer.results

            if methods[DataProc.OUTLIER_REMOVAL.value] != 0:
                outlier_remover = OutlierRemover(
                    method=methods[DataProc.OUTLIER_REMOVAL.value],
                    data=pd.DataFrame(feature_data),
                )
                outlier_remover.run()
                feature_data = outlier_remover.results

            if methods[DataProc.TRANSFORMATION.value] != 0:
                transform_df = pd.DataFrame(feature_data)
                if methods[DataProc.TRANSFORMATION.value][DataProc.GROUPBY.value] != 0:
                    transform_df[DataProc.GROUPBY.value] = self.data_df[
                        methods[DataProc.TRANSFORMATION.value][DataProc.GROUPBY.value]
                    ]
                if (
                    methods[DataProc.TRANSFORMATION.value][DataProc.TARGET_FIELD.value]
                    != 0
                ):
                    transform_df[DataProc.TARGET_FIELD.value] = self.data_df[
                        methods[DataProc.TRANSFORMATION.value][
                            DataProc.TARGET_FIELD.value
                        ]
                    ]
                transformer = Transformers(
                    method=methods[DataProc.TRANSFORMATION.value], data=transform_df
                )
                transformer.run()
                feature_data = transformer.results

            if methods[DataProc.BINNING.value] != 0:
                binner = Binners(
                    method=methods[DataProc.BINNING.value],
                    data=feature_data,
                    expected_values=expected_values,
                )
                binner.run()
                feature_data = binner.data
            self.results_lst.append(feature_data)

        return pd.concat(self.results_lst, axis=1)
