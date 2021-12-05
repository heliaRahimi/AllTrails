# GimmeAllTheTrails

Visualizing the sentiment of trail reviews of NS on a popularly used hiking trail app.

# Dashboard

View of all trails in NS

![GLOBAL](img/Global.png)


View of single trail

![LOCAL](img/Local.png)

# Installation
The project can be installed as a python package

```
conda create --name GimmeAllTheTrails python=3.9
cd /path/to/GimmeAllTheTrails
pip install -e .
```

# Running App

Once project is installed, app can be run from terminal/cmd line:
```
GimmeAllTheTrails --csv_dir /path/to/GimmeAllTheTrails/data/csv
```

# Datasets

All data is stored in data/csv the datasets are derived from the original raw data which can be found in
data/raw.

# Machine Learning Components

This project has both clustering and sentiment analysis ML components to it. The results of the
algorithms are stored in data/csv/custering.csv and data/csv/sentiment.csv. 

## Reproducing ML results
### Clustering
 To reproduce clustering results simply rerun the clustering.py script:

```
cd /path/to/GimmeAllTheTrails
python GimmeAllTheTrails/clustering.py
```

### Sentiment Analysis
To reproduce sentiment analysis, an additional package, NLTK, must be installed: instructions for this can
be found at https://www.nltk.org/data.html

Once installed, simply rerun sentiment.py script:
```
cd /path/to/GimmeAllTheTrails
python GimmeAllTheTrails/sentiment.py
```

# Collecting New Data

If you would like to further experiment with this project by adding more trails from other regions in the world, you can use 
the tools present in GimmeAllTheTrails/utils/scraper.py
