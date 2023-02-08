import pandas as pd
from enums import DataProc
from logger import create_logger
from read_write import read_fitted_data, write_fitted_data
import numpy as np


class Binners(object):
    """
     Methods for binning features according to json recipe

     Parameters
     ----------
     method: dict or list
         Binner methods and attributes.
     data: pd.DataFrame
         Data to be transformed
    expected_values: List
        Expected values in data or None

     Notes
     ----------
     Expected values required when method[type] == label_encoding, else None.

     Examples
     ----------
     >>> data = pd.DataFrame(np.random.randint(1,100, 1000), columns=['My_feature'])
     >>> methods = [0, 18, 25, 45, 65, 100]
     >>> my_binner = Binners(method=methods, data=data)
     >>> my_binner.run()
    """

    def __init__(self, method: dict, data: pd.DataFrame, expected_values: list = None):
        self.method = method
        self.data = data
        self.feature_name = data.columns[0]
        self.label_encoders = {}
        self.expected_values = expected_values
        self.results = pd.DataFrame()
        self.log = create_logger(name="Binner")

    def run(self):
        self.log.info("Running binning for feature {}...".format(self.feature_name))
        if type(self.method) is list:
            self.lst_binner()
        if type(self.method) is dict:
            if self.method[DataProc.TYPE.value] == DataProc.LABEL_ENCODING.value:
                if self.method[DataProc.FIT.value] == 1:
                    self.map = {}
                    self.lbl_encoder_fit()
                    write_fitted_data(
                        data=self.map,
                        data_path=DataProc.PATH.value,
                        feature_name=self.feature_name,
                        file_type="json",
                    )
                if self.method[DataProc.FIT.value] == 0:
                    self.map = read_fitted_data(
                        data_path=DataProc.PATH.value,
                        feature_name=self.feature_name,
                        file_type="json",
                    )
                    self.lbl_encoder_transform()
        self.log.info("Binning for feature {} complete...".format(self.feature_name))

    def lst_binner(self):
        self.log.info(
            "Running range binner fit transform for feature {}...".format(
                self.feature_name
            )
        )
        bin_lbl = list(range(len(self.method) - 1))
        self.data[self.feature_name] = pd.cut(
            self.data[self.feature_name], bins=self.method, labels=bin_lbl
        )

    def lbl_encoder_fit(self):
        self.log.info(
            "Running label encoder fit transform for feature {}...".format(
                self.feature_name
            )
        )
        if self.method[DataProc.ASCENDING.value] == 0:
            for feat_cnt, feature_value in enumerate(reversed(self.expected_values)):
                self.map[feature_value] = feat_cnt
        else:
            for feat_cnt, feature_value in enumerate(self.expected_values):
                self.map[feature_value] = feat_cnt

        self.results[self.feature_name] = self.data[self.feature_name].map(self.map)
        self.log.info(
            "Label encoder fit transform for feature {} complete...".format(
                self.feature_name
            )
        )

    def lbl_encoder_transform(self):
        self.log.info(
            "Running label encoder transform for feature {}...".format(
                self.feature_name
            )
        )
        self.results[self.feature_name] = self.data[self.feature_name].map(self.map)
        self.log.info(
            "Label encoder transform for feature {} complete...".format(
                self.feature_name
            )
        )


#
# data = pd.DataFrame(np.random.randint(1,100, 1000), columns=['My_feature'])
# methods = [0, 18, 25, 45, 65, 100]
# my_binner = Binners(method=methods, data=data)
# my_binner.run()
