from pandas import read_csv as read
import pandas as pd
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn import metrics


def csv_reader():
    """
      Read a csv file
      """
    f = open(u'Данные для задания 3 и 4.csv')
    data = read(f, sep=',', encoding='utf-8')
    f.close()
    clean_data(data)


def clean_data(datalist):
    x = datalist.values[::, 0:14]
    #y = datalist.values[::, 14:]
    df = pd.DataFrame(x)
    df = df.replace("?", float('nan'))
    x = df.fillna(0)  # замена ? на 0
    replace_mas1 = ["notpresent", "yes", "good"]
    replace_mas0 = ["present", "no", "poor"]
    for text in replace_mas1:  #замена notpresent", "yes", "good" на 1
        x = x.replace(text, 1)
    for text in replace_mas0:
        x = x.replace(text, 0)
    cluster(x)


def cluster(x):
    kmeans = KMeans(n_clusters=3, random_state=0).fit(x) #кластеризация методом к средних
    kmeans.fit_predict(x)
    score = metrics.silhouette_score(x, kmeans.labels_)
    print(kmeans.labels_)
    print(score)


if __name__ == "__main__":
    with open("Данные для задания 3 и 4.csv", "r") as f_obj:
        csv_reader()
