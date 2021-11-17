import glob
import os
import pandas as pd
import ast

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
        trail_id_coords["latitude"] = trail_id_coords["coords"].apply(lambda x: float(x["latitude"]))
        trail_id_coords["longitude"] = trail_id_coords["coords"].apply(lambda x: float(x["longitude"]))
        return trail_id_coords.drop(columns=["coords"])


if __name__ == "__main__":
    data = AllTrails(r"/blah/GimmeAllTheTrails/data\csv")
    print(data)