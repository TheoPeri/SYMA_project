import pandas as pd
from zipfile import ZipFile
import os


class GetData(object):
    """
    we have 1 file.
    - dressipi_recsys2022.zip - the folder contain all the datas:
        - candidate_items.csv
        - item_features.csv
        - test_final_sessions.csv
        - test_leaderboard_sessions.csv
        - train_purchases.csv
        - train_sessions.csv
    """

    def __init__(self, split_percent = 99):
        self.ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
        self.DATA_DIR = self.ROOT_DIR + "/data/"
        
        # unzip
        self._unzipper("dressipi_recsys2022")
        
        # get dataframes
        self.list_candidate_items = self._get_df("candidate_items")
        self.df_item_features = self._get_df("item_features")
        self.df_train_purchases = self._get_df("train_purchases")
        self.df_train_sessions = self._get_df("train_sessions")
        self.df_test_leaderboard_sessions = self._get_df("test_leaderboard_sessions")
        self.df_test_final_sessions = self._get_df("test_final_sessions")
        
        # basic type changes
        self.list_candidate_items = self.list_candidate_items.astype(int)
        self.list_candidate_items = sorted(self.list_candidate_items["item_id"].tolist())
        
        self.df_item_features = self.df_item_features.astype(int)
        
        for df in [self.df_train_purchases, self.df_train_sessions, self.df_test_leaderboard_sessions, self.df_test_final_sessions]:
            df[["session_id", "item_id"]] = df[["session_id", "item_id"]].astype(int)
            df['date'] = pd.to_datetime(df['date'])
            
        # Get train test dfs
        shape_train = self.df_train_purchases.shape[0]
        
        self.train_purchases = self.df_train_purchases.head(int(shape_train*split_percent/100))
        self.test_purchases = self.df_train_purchases.tail(int(shape_train*(100 - split_percent)/100))
        
        self.train_sessions = self.df_train_sessions.set_index('session_id').loc[self.train_purchases["session_id"].tolist()].reset_index()
        self.test_sessions = self.df_train_sessions.set_index('session_id').loc[self.test_purchases["session_id"].tolist()].reset_index()

    def _get_df(self, file_base):
        path = self.DATA_DIR + "__use__/dressipi_recsys2022/" + file_base + ".csv"

        return pd.read_csv(path)

    def _unzipper(self, file_path):
        path = self.DATA_DIR
        with ZipFile(path + file_path + ".zip", "r") as zipped_file:
            zipped_file.extractall(path + "__use__")