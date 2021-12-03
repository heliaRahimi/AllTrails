import glob
import os
import pandas as pd
import ast
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from itertools import chain


class AllTrails(object):
    """
    load all data into this class
    """

    def __init__(self, csv_dir):
        # load all raw data as DFs #
        self.csv_dir = csv_dir
        self.files = glob.glob(os.path.join(csv_dir, "*.csv"))
        self.datasets = {
            os.path.basename(p).split(".")[0]: pd.read_csv(p) for p in self.files
        }

    # All sub-datasets stored in this manner to ensure that only copies are used throughout the app #
    @property
    def lat_lon_trail_id(self):
        return self._lat_lon_trail_id().copy(deep=False)

    @property
    def type_length_elev_rating(self):
        return self._type_length_elev_rating().copy(deep=False)

    @property
    def trail_descriptions(self):
        return self._trail_descriptions().copy(deep=False)

    @property
    def main_map_data(self):
        return self._main_map_data().copy(deep=False)

    @property
    def cluster_map_data(self):
        return self._cluster_map_data().copy(deep=False)

    @property
    def sentiment_analysis_data(self):
        return self._sentiment_analysis_data().copy(deep=False)

    @property
    def key_words(self):
        return self._key_words().copy(deep=False)

    @property
    def num_reviews(self):
        return self._num_reviews().copy(deep=False)

    @property
    def tag_dummies(self):
        return self._tag_dummies().copy(deep=False)

    def _lat_lon_trail_id(self) -> pd.DataFrame:
        """
        get the lat lon pairs and associated name pairs for each hiking trail location
        :return: DF containing lat lon info
        """
        # get coords and trail id
        trail_id_coords = self.datasets["meta_reviews"].copy(deep=False)[
            ["coords", "trail_id"]
        ]
        # convert strings to dicts
        trail_id_coords.loc[:, "coords"] = trail_id_coords["coords"].apply(
            lambda x: ast.literal_eval(x)
        )
        # get lat and lon
        trail_id_coords["latitude"] = trail_id_coords.loc[:, "coords"].apply(
            lambda x: str(x["latitude"])
        )
        trail_id_coords["longitude"] = trail_id_coords.loc[:, "coords"].apply(
            lambda x: str(x["longitude"])
        )
        return trail_id_coords.drop(columns=["coords"])

    def _trail_descriptions(self) -> pd.DataFrame:
        """
        get trail descriptions
        :return: DF containing trail descriptions
        """
        return self.datasets["meta_reviews"][["trail_id", "description"]]

    def _type_length_elev_rating(self) -> pd.DataFrame:
        """
        get type length and elevation fields
        :return: DF containing type, length, elev and rating
        """
        # load type length and elevation
        data = self.datasets["meta_reviews"].loc[:, ["trail_id", "length_elev_type"]]
        data.loc[:, "length_elev_type"] = data["length_elev_type"].apply(
            lambda x: ast.literal_eval(x)
        )
        type_length_elev_rating = pd.DataFrame()
        type_length_elev_rating.loc[:, "trail_id"] = data["trail_id"]
        # in form "['6.0 mi', '738 ft', 'Loop']"
        type_length_elev_rating.loc[:, "length"] = data["length_elev_type"].apply(
            lambda x: float(x[0].split(" ")[0])
        )
        type_length_elev_rating.loc[:, "elevation"] = data["length_elev_type"].apply(
            lambda x: int(x[1].split(" ")[0].replace(",", ""))
        )
        type_length_elev_rating.loc[:, "type"] = data["length_elev_type"].apply(
            lambda x: x[2]
        )
        # process ratings
        stars = self.datasets["star_reviews"]
        type_length_elev_rating.loc[:, "rating"] = stars["rating"].apply(
            lambda x: ast.literal_eval(x)
        )
        # add avg rating
        type_length_elev_rating.loc[:, "avg_rating"] = type_length_elev_rating[
            "rating"
        ].apply(
            lambda x: np.mean(
                [
                    int(i.split(" ")[0])
                    for i in x
                    if i.split(" ")[0] != "NaN" and int(i.split(" ")[0])
                ]
            )
        )
        type_length_elev_rating.loc[:, "normalized_length"] = (
            type_length_elev_rating["length"] - type_length_elev_rating["length"].min()
        ) / (
            type_length_elev_rating["length"].max()
            - type_length_elev_rating["length"].min()
        )
        type_length_elev_rating.loc[:, "normalized_elevation"] = (
            type_length_elev_rating["elevation"]
            - type_length_elev_rating["elevation"].min()
        ) / (
            type_length_elev_rating["elevation"].max()
            - type_length_elev_rating["elevation"].min()
        )
        return type_length_elev_rating

    def _main_map_data(self):
        """
        concat all datasets together for main map dataset
        :return: DF containing concatonated data for general purpose functionality
        """
        # dset 1
        lat_lon_ind = self.lat_lon_trail_id.copy(deep=False)
        lat_lon_ind.index = lat_lon_ind["trail_id"]
        lat_lon_ind = lat_lon_ind.drop(columns="trail_id")
        # dset 2
        type_length_elev_rating = self.type_length_elev_rating.copy(deep=False)
        type_length_elev_rating.index = type_length_elev_rating["trail_id"]
        type_length_elev_rating = type_length_elev_rating.drop(columns="trail_id")
        # concated
        main_map_data = pd.concat(
            [type_length_elev_rating, lat_lon_ind], axis=1
        ).reset_index()
        # # drop outliers #
        # main_map_data = main_map_data[(np.abs(stats.zscore(main_map_data)) < 3).all(axis=1)]
        return main_map_data

    def _cluster_map_data(self):
        """
        get clustered data
        :return:  DF containing cluster data for map
        """
        data = self.main_map_data.copy()
        cluster_data = self.datasets["clusters"][["clusters", "trail_id"]]
        # get same entries as main data
        # cluster_data = cluster_data[cluster_data["trail_id"].isin(data["trail_id"])]
        cluster_data.index = cluster_data["trail_id"]
        data.index = data["trail_id"]
        data = pd.concat([cluster_data, data], axis=1)
        data = data.drop(columns=["trail_id"]).reset_index()
        return data

    def _sentiment_analysis_data(self):
        """
        get sentiment analysis data
        :return:  DF containing sentiment analysis data
        """
        sentiment_data = self.datasets["sentiment"]
        sentiment_data.index = sentiment_data["trail_id"]
        sentiment_data = sentiment_data.drop(columns="trail_id")
        # add latitude and longitude for map plot
        lat_lon_ind = self.lat_lon_trail_id.copy(deep=False)
        lat_lon_ind.index = lat_lon_ind["trail_id"]
        lat_lon_ind = lat_lon_ind.drop(columns="trail_id")
        sentiment_data = pd.concat([sentiment_data, lat_lon_ind], axis=1).reset_index()

        return sentiment_data

    def _key_words(self):
        """
        get keywords data
        :return:  DF containing key words
        """
        keywords = self.datasets["key_words"].copy(deep=False)
        keywords.loc[:, "list"] = keywords["key_words"].apply(
            lambda x: list(chain(*ast.literal_eval(x)))
        )
        keywords.loc[:, "str"] = keywords["list"].apply(lambda x: " ".join(x))
        keywords.loc[:, "set"] = keywords["list"].apply(lambda x: set(x))
        return keywords

    def _num_reviews(self):
        """
        get number of reviews
        :return: DF containing reviews
        """
        reviews = pd.DataFrame()
        reviews["ratings"] = self.type_length_elev_rating["rating"].apply(
            lambda x: len(x)
        )
        reviews["trail_id"] = self.type_length_elev_rating["trail_id"]
        reviews["written_reviews"] = self.datasets["written_reviews"]["reviews"].apply(
            lambda x: len(ast.literal_eval(x))
        )
        reviews["ratings_normalized"] = (
            reviews["ratings"] - reviews["ratings"].min()
        ) / (reviews["ratings"].max() - reviews["ratings"].min())
        reviews["written_normalized"] = (
            reviews["written_reviews"] - reviews["written_reviews"].min()
        ) / (reviews["written_reviews"].max() - reviews["written_reviews"].min())
        return reviews

    def _tag_dummies(self):
        """
        get tags as dummy variables
        :return: DF containing tag dummies
        """
        tags = self.datasets["clusters"].copy(deep=False)
        tags = tags.drop(columns=["type", "length", "elevation"])
        tags["trail_name"] = tags["trail_id"].apply(
            lambda x: " ".join(x.split("/")[-1].split("-"))
        )
        return tags


if __name__ == "__main__":
    AllTrails(
        csv_dir=r"C:\Users\NoahB\Desktop\School\first year MCSC (2021-2022)\CS6612\group_proj\GimmeAllTheTrails\data\csv"
    )
