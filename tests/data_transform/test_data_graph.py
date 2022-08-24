import pandas as pd
import pytest
from src.data_transform.data_graph import DataGraph

class TestData:
    # def setup_method(self):
    #     path = "/user/jemonet/home/Documents/Thèse/Code/Graphes/figure_lib/data_tests/simple_data.csv"
    #     datagraph = DataGraph(path)

    #     return datagraph

    # @pytest.fixture
    # def dataframe_for_test(self):
    #     data = {"calories": [420, 380, 390],"duration": [50, 40, 45]}
    #     idx = pd.Index(["day1", "day2", "day3"],name="Time")
    #     df_test = pd.DataFrame(data, index = idx)
        
    #     return df_test

    def test_should_inherited_from_dataframe(self, dataframe_for_test):
        data = {"calories": [420, 380, 390],"duration": [50, 40, 45]}
        idx = pd.Index(["day1", "day2", "day3"],name="Time")
        df_test = pd.DataFrame(data, index = idx)

        path = "/user/jemonet/home/Documents/Thèse/Code/Graphes/figure_lib/data_tests/simple_data.csv"
        datagraph = DataGraph(path)
        
        assert datagraph.equals(df_test.reset_index(level=0))

