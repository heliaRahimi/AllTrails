from pathlib import Path
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def normalize_column(df_column: pd.Series) -> pd.Series:
    """
    This method should recalculate all values of a numeric column and normalise it between 0 and 1.
    :param df_column: Dataset's column
    :return: The column normalized
    """
    # convert all values to positive values
    min = df_column.min()
    df_column = df_column.apply(lambda x: x + abs(min))
    max_val = df_column.max()
    df_column = df_column.apply(lambda x: x/max_val)
    return df_column


def create_df(csv_f):
    # data cleaning
    dataset = pd.read_csv(csv_f)
    len_elev_type = dataset['length_elev_type']
    new_params = [''] * len(len_elev_type)
    t_length = [''] * len(len_elev_type)
    t_elevation = [''] * len(len_elev_type)
    t_type = [''] * len(len_elev_type)

    for i in range(0, len(len_elev_type)):
        new_params[i] = len_elev_type[i].replace("'", '').replace("[", "").replace("]", "")
        t_length[i], t_elevation[i], t_type[i] = new_params[i].split(", ")
        t_length[i] = t_length[i].replace(' mi', '')
        t_length[i] = float(t_length[i])
        t_elevation[i] = t_elevation[i].replace(' ft', '').replace(',', '.')
        t_elevation[i] = float(t_elevation[i])

    t_tags = dataset['tags']
    t_tags_list = ['']*len(t_tags)
    for i in range(0, len(t_tags)):
        t_tags_list[i] = t_tags[i].replace(" ","").replace("[", "").replace("]", "").replace(",", "").replace("''"," ")\
            .replace("'","")
        t_tags_list[i] = t_tags_list[i].split()

    trail_id = dataset['trail_id']
    df = pd.DataFrame({'trail_id': trail_id, 'length': t_length, 'elevation': t_elevation, 'type': t_type,
                       'tags': t_tags_list})
    return df

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def make_cluster_csv(reviews_csv_dir, outfile):
    """
    applies clustering algo to data,
    :param reviews_csv_dir: location of meta_reviews.csv file
    :param outfile: location to output resultant csv file
    :return: None
    """
    trail_dataset = create_df(
        csv_f=reviews_csv_dir)
    # generating one hot encoder for type column
    ohe = OneHotEncoder(handle_unknown='ignore')
    encoded_columns = ohe.fit_transform(trail_dataset['type'].values.reshape(-1, 1)).toarray()
    df = trail_dataset.copy()
    ohe_column_names = ['Out & back', 'Loop', 'Point to point']
    df[ohe_column_names] = pd.DataFrame(encoded_columns)
    df = df.drop(columns=['type'])

    # explode the tags column so that we have different rows with same index for each tag
    trail_data_expo = trail_dataset['tags'].explode()
    # all unique tags
    unique_tags = trail_data_expo.unique()
    # the frequency of all tags
    frequency = trail_data_expo.value_counts()
    # dataframe of all tags and their frequencies
    df_frequent_tags = pd.DataFrame({ 'tag': frequency.index, 'frequency': frequency.values })

    """
    -----------------------------------Plot the frequency of all tags------------------------------------------- 
    """
    # fig1 = px.bar(df_frequent_tags, x="tag", y='frequency', title="frequency of tags")
    # fig1.show()
    """
    ------------------------------------------------------------------------------------------------------------ 
    """
    # most frequent tag. The most frequent tag is repeated around 590 times, any number greater than 590 and smaller than
    # 713 (total number of rows) means no tags are selected. 0 means all tags are selected. use frequency plot to see
    # the distribution of tags

    freq = 0
    df_most_freq = df_frequent_tags[~(df_frequent_tags['frequency'] <= freq)]
    most_freq_tag = df_most_freq['tag']

    # add new columns representing the tags
    for i in range(0, len(df_most_freq)):
        index_list = trail_data_expo.index[trail_data_expo == most_freq_tag[0]].tolist()
        for c in range(0, len(df)):
            if c in index_list:
                df.loc[c, most_freq_tag[i]] = 1
            else:
                df.loc[c, most_freq_tag[i]] = 0

    # normalize length and elevation columns
    processed_data = df.drop(columns=['trail_id', 'tags'])
    processed_data['length'] = normalize_column(df['length'])
    processed_data['elevation'] = normalize_column(df['elevation'])

    """
    ------------------------------------------------Clustering---------------------------------------------------------
    """

    # KMeans clustering
    model = KMeans(n_clusters=4, init='k-means++')
    model.fit(processed_data)
    labels = model.predict(processed_data)
    score = metrics.silhouette_score(processed_data, labels, metric='euclidean')

    # clustering using KMedoids
    # model = KMedoids(n_clusters=6,  init='k-medoids++').fit(processed_data)
    # labels = model.predict(processed_data)

    # clustering using Agglomerative clustering
    # model = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='ward')
    # labels = model.fit_predict(processed_data)

    """
    ---------------------------------------------Plot dendrogram of hierarchical clustering-----------------------------
    """
    # model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    # model = model.fit(processed_data)
    # plt.title("Hierarchical Clustering Dendrogram")
    # # plot the top three levels of the dendrogram
    # plot_dendrogram(model, truncate_mode="level", p=3)
    # plt.xlabel("Number of points in node (or index of point if no parenthesis) using tags frequency>400")
    # plt.show()
    """
    --------------------------------------------------------------------------------------------------------------------
    """
    #
    # #Naming labels
    # cluster_labels = [0] * len(labels)
    # for i in range(0, len(labels)):
    #     if labels[i] == 0:
    #         cluster_labels[i] = 'cluster 0'
    #     elif labels[i] == 1:
    #         cluster_labels[i] = 'cluster 1'
    #     elif labels[i] == 2:
    #         cluster_labels[i] = 'cluster 2'
    #     elif labels[i] == 3:
    #         cluster_labels[i] = 'cluster 3'
    #     elif labels[i] == 4:
    #         cluster_labels[i] = 'cluster 4'
    #     elif labels[i] == 5:
    #         cluster_labels[i] = 'cluster 5'
    #     # elif labels[i] == 6:
    #     #     cluster_labels[i] = 'cluster 6'

    """
    -------------------------------------------------------------------------------------------------------------
    """
    #
    # to better plot the data I have transformed the one hot encoder of column = type to the original column
    data_copy = processed_data.copy(deep=True)
    decoded_column = ohe.inverse_transform(data_copy[ohe_column_names].values).squeeze()
    join_data = pd.DataFrame(decoded_column,
                             columns=['type'])  # Creating a data frame of the decoded column
    data_copy = data_copy.drop(ohe_column_names, axis=1)  # Dropping the original encoded columns
    data_copy = pd.concat([data_copy, join_data], axis=1)  # Concatinating the original column

    data_copy['clusters'] = labels


    """
    ------------------------------------------------------Final dataframe----------------------------------------------
    """
    final_dataframe = data_copy.copy()
    final_dataframe['trail_id'] = trail_dataset['trail_id']
    final_dataframe.to_csv(
        outfile)



if __name__ == "__main__":
    make_cluster_csv("data/csv/meta_reviews.csv",
                     "data/csv/clusters.csv")
