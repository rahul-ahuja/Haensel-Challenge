import matplotlib.pyplot  as plt
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm



def read_data(filename):
    '''reading data csv file by pandas
       Arguments
       ---------
       filename: csv file name
    '''
    data = pd.read_csv(filename, header=None)
    return data


def plot_labels(labels):
    '''Plotting class distribution histogram

       Arguments
       ---------
       labels: class labels
    '''
    lab = np.array(labels).flatten()
    labels_count = np.unique(lab, return_counts = True)
    plt.figure(figsize=(16,10), dpi=75)
    plt.bar(range(len(labels_count[0])), labels_count[1], align='center', alpha=0.5)
    plt.title('number of different labels in the data')
    plt.xlabel('labels')
    plt.ylabel('quantity of each label')
    plt.xticks(range(len(labels_count[0])), labels_count[0])
    plt.show()

def normalize(continuous_data):
    '''Performing scaling to the continuous
    data by normalization

       Arguments
       ---------
       continuous_data: features with continuous values
    '''
    quant_features = np.array(continuous_data.columns)
    for each in quant_features:
        mean, std = continuous_data[each].mean(), continuous_data[each].std()
        continuous_data.loc[:, each] = (continuous_data[each] - mean)/std
        return np.array(continuous_data)


def one_hot_encoding(data_list):
    '''Applies one-hot encoding to the categorical data.

       Arguments
       ---------
       data_list: categorical data

    '''
    encoder = LabelBinarizer()
    encoder.fit(data_list)
    cols = encoder.transform(data_list)
    return pd.DataFrame(cols).astype('float32')

def get_batches(X, y, batch_size):
    '''Create a generator that returns batch data X, labels y of size
       batch_size.

       Arguments
       ---------
       X: feature data
       batch_size: Batch size, the number of samples per batch
       y: class labels in the batch
    '''
    X, y = shuffle(X, y)
    num_examples = len(X)
    for offset in tqdm(range(0, num_examples, batch_size)):
        end = offset + batch_size
        batch_x, batch_y = X[offset:end], y[offset:end]
        yield batch_x, batch_y