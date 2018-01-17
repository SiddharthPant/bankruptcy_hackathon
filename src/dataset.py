import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pandas import DataFrame, read_csv
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
from textwrap import wrap


class Dataset(object):
    def __init__(self, folder_path='../data/', filetype='*.arff', test_size=0.2, random_state=9):
        self.df = read_csv('../data/key_values.csv', index_col=0, skiprows=0)
        self.test_size = test_size
        self.random_state = random_state
        self.path_list = glob.glob(folder_path + filetype)
        self.data = [DataFrame(loadarff(path)[0]) for path in self.path_list]
        self.totals = [datum.shape[0] for datum in self.data]
        self.replace_cols()
        self.xy = self.process_data()
        self.train, self.val, self.test = self.split_data()
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = self.concat()

    def concat(self):
        x_train = pd.concat([df[0] for df in self.train])
        y_train = pd.concat([df[1] for df in self.train])
        x_val = pd.concat([df[0] for df in self.val])
        y_val = pd.concat([df[1] for df in self.val])
        x_test = pd.concat([df[0] for df in self.test])
        y_test = pd.concat([df[1] for df in self.test])
        return x_train, y_train, x_val, y_val, x_test, y_test

    def process_data(self):
        xy = []
        for datum in self.data:
            x, y = datum.iloc[:, :-1], datum['class']
            x = x.drop(self.df.loc[['Attr37', 'Attr21']]['description'].values, axis=1)
            xy.append((x, y))
        return xy

    def replace_cols(self):
        for datum in self.data:
            datum.columns = self.df.description

    def split_data(self):
        train_test_data = self.splitter(self.xy)
        train, test = get_train_test_list(train_test_data)

        train_val_data = self.splitter(train)
        train, val = get_train_test_list(train_val_data)

        return train, val, test

    def splitter(self, data):
        return [
            train_test_split(datum[0], datum[1], test_size=self.test_size, random_state=self.random_state)
            for datum in data
        ]


def get_train_test_list(data):
    return (
        impute([(datum[0], datum[2]) for datum in data]),
        impute([(datum[1], datum[3]) for datum in data])
    )

def jointer(data, index):
    data_list = [df[index] for df in data]
    return pd.concat(data_list)



def impute(data):
    return [(x.fillna(x.median()), y) for x, y in data]


def get_counts(datum, gt):
    null_counts = datum.isnull().sum().sort_values(ascending=False)
    return null_counts[null_counts > gt]


def get_null_counts(data, gt=100):
    return [get_counts(datum, gt) for datum in data]


def listplot(data, size):
    pass


def plot(data, size):
    data = (data / size) * 100
    # plt.figure(figsize=(5, 6))
    ax = sns.barplot(data.values, data.index)
    ax.set_ylabel('')
    labels = data.index
    labels = ['\n'.join(wrap(l, 30)) for l in labels]
    ax.set_yticklabels(labels)
    plt.show()

# data = Dataset()
# df = data.data
# null_counts = get_null_counts(data.data, gt = 100)
# for nc, size in zip(null_counts, data.totals):
#     plot(nc, size)
