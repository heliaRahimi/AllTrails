import glob
import os
import pandas as pd
import ast
import numpy as np
import random
from itertools import chain

class AllTrails(object):
    """
    load all data into this class
    """
    def __init__(self, csv_dir):
        self.csv_dir = csv_dir
        self.files = glob.glob(os.path.join(csv_dir, "*.csv"))
        self.datasets = {
            os.path.basename(p).split(".")[0]:pd.read_csv(p) for p in self.files
        }
        self.lat_lon_trail_id = self._lat_lon_trail_id()
        self.type_length_elev_rating = self._type_length_elev_rating()
        self.trail_descriptions = self._trail_descriptions()
        self.main_map_data = self._main_map_data()
        self.cluster_map_data = self._cluster_map_data()
        self.sentiment_analysis_data = self._sentiment_analysis_data()
        self.key_words = self._key_words()
        self.num_reviews = self._num_reviews()
        self.tag_dummies = self._tag_dummies()

    def _lat_lon_trail_id(self) -> pd.DataFrame:
        """
        get the lat lon pairs and associated name pairs for each hiking trail location
        :return:
        """
        # get coords and trail id
        trail_id_coords = self.datasets["meta_reviews"][["coords", "trail_id"]]
        # convert strings to dicts
        trail_id_coords["coords"] = trail_id_coords["coords"].apply(lambda x: ast.literal_eval(x))
        # get lat and lon
        trail_id_coords["latitude"] = trail_id_coords["coords"].apply(lambda x: str(x["latitude"]))
        trail_id_coords["longitude"] = trail_id_coords["coords"].apply(lambda x: str(x["longitude"]))
        return trail_id_coords.drop(columns=["coords"])

    def _trail_descriptions(self) -> pd.DataFrame:
        """
        get trail descriptions
        :return:
        """
        return self.datasets["meta_reviews"][["trail_id", "description"]]

    def _type_length_elev_rating(self) -> pd.DataFrame:
        """
        get type length and elevation fields
        :return:
        """
        # load type length and elevation
        data = self.datasets["meta_reviews"][["trail_id", "length_elev_type"]]
        data["length_elev_type"] = data["length_elev_type"].apply(lambda x: ast.literal_eval(x))
        type_length_elev_rating = pd.DataFrame()
        type_length_elev_rating["trail_id"] = data["trail_id"]
        # in form "['6.0 mi', '738 ft', 'Loop']"
        type_length_elev_rating["length"] = data["length_elev_type"].apply(lambda x: float(x[0].split(" ")[0]))
        type_length_elev_rating["elevation"] = data["length_elev_type"].apply(lambda x: int(x[1].split(" ")[0].replace(",", "")))
        type_length_elev_rating["type"] = data["length_elev_type"].apply(lambda x: x[2])
        # process ratings
        stars = self.datasets["star_reviews"]
        type_length_elev_rating["rating"] = stars["rating"].apply(lambda x: ast.literal_eval(x))
        # add avg rating
        type_length_elev_rating["avg_rating"] = type_length_elev_rating["rating"].apply(lambda x: np.mean([int(i.split(" ")[0]) for i in x if i.split(" ")[0] != "NaN"]))
        type_length_elev_rating["normalized_length"] = (type_length_elev_rating["length"] - type_length_elev_rating["length"].min()) /(type_length_elev_rating["length"].max()-type_length_elev_rating["length"].min())
        type_length_elev_rating["normalized_elevation"] = (type_length_elev_rating["elevation"] - type_length_elev_rating["elevation"].min()) /(type_length_elev_rating["elevation"].max()-type_length_elev_rating["elevation"].min())
        return type_length_elev_rating

    def _main_map_data(self):
        """
        concat all datasets together for main map dataset
        :return:
        """
        # dset 1
        lat_lon_ind = self.lat_lon_trail_id.copy(deep=False)
        lat_lon_ind.index = lat_lon_ind["trail_id"]
        lat_lon_ind = lat_lon_ind.drop(columns = "trail_id")
        # dset 2
        type_length_elev_rating = self.type_length_elev_rating.copy(deep=False)
        type_length_elev_rating.index = type_length_elev_rating["trail_id"]
        type_length_elev_rating = type_length_elev_rating.drop(columns="trail_id")

        return pd.concat([type_length_elev_rating, lat_lon_ind], axis=1).reset_index()

    def _cluster_map_data(self):
        data = self.main_map_data.copy()
        cluster_data = self.datasets["clusters"][["clusters", "trail_id"]]
        cluster_data.index = cluster_data["trail_id"]
        data.index = data["trail_id"]
        data = pd.concat([cluster_data, data], axis=1)
        data = data.drop(columns=["trail_id"]).reset_index()
        return data

    def _sentiment_analysis_data(self):
        sentiment_data = self.datasets["sentiment"]
        return sentiment_data

    def _key_words(self):
        keywords = self.datasets["key_words"].copy(deep=False)
        keywords["list"] =  keywords["key_words"].apply(lambda x: list(chain(*ast.literal_eval(x))))
        keywords["str"] = keywords["list"].apply(lambda x: " ".join(x))
        keywords["set"] = keywords["list"].apply(lambda x: set(x))
        return keywords

    def _num_reviews(self):
        reviews = pd.DataFrame()
        reviews["ratings"] = self.type_length_elev_rating["rating"].apply(lambda x: len(x))
        reviews["trail_id"] = self.type_length_elev_rating["trail_id"]
        reviews["written_reviews"] = self.datasets["written_reviews"]["reviews"].apply(lambda x: len(ast.literal_eval(x)))
        reviews["ratings_normalized"] = (reviews["ratings"] - reviews["ratings"].min()) /(reviews["ratings"].max()-reviews["ratings"].min())
        reviews["written_normalized"] = (reviews["written_reviews"] - reviews["written_reviews"].min()) /(reviews["written_reviews"].max()-reviews["written_reviews"].min())
        return reviews

    def _tag_dummies(self):
        tags = self.datasets["clusters"].copy(deep=False)
        tags = tags.drop(columns = ["type", "length", "elevation"])
        tags["trail_name"] = tags["trail_id"].apply(lambda x: " ".join(x.split("/")[-1].split("-")))
        return tags





if __name__ == "__main__":
    data = AllTrails(r"C:\Users\NoahB\Desktop\School\first year MCSC (2021-2022)\CS6612\group_proj\GimmeAllTheTrails\data\csv")
    print(data.main_map_data)