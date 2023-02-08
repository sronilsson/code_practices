import pandas as pd
import pytest
from data_processor import DataProcessor


@pytest.mark.parametrize(
    "config_path, data_path",
    [
        ("configs/config_train.json", "data/data_1000_train.csv"),
        ("configs/config_test.json", "data/data_1000_test.csv"),
    ],
)
def test_fit_and_transforms(config_path, data_path):
    data_processor = DataProcessor(config_path=config_path)
    data_processor.read_data(data_path=data_path)
    results = data_processor.transform()
    assert type(results) is pd.DataFrame
    assert results.isnull().values.any()
