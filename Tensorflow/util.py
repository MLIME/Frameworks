import os
import time
import matplotlib.pyplot as plt
import pickle
import numpy as np
from pandas_ml import ConfusionMatrix


def get_log_path():
    log_basedir = './graphs'
    run_label = time.strftime('%d-%m-%Y_%H-%M-%S')  # e.g. 12-11-2016_18-20-45
    return os.path.join(log_basedir, run_label)


def plot9images(images, cls_true, img_shape, cls_pred=None):
    """
    Function to show 9 images with their respective classes.
    If cls_pred is an array, you can see the image and the prediction.

    :type images: np array
    :type cls_true: np array
    :type img_shape: np array
    :type cls_prediction: None or np array
    """
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def one_hot(labels, num_labels=10):
    """
    Function transorm the labels array to be one array of one-hot vectors.

    :type labels: np array
    :type num_labels: int
    :rtype: np array
    """
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return labels


def get_batch(data, labels, batch_size):
    """
    Given one dataset data and an array of labels label,
    this function returns a batch of size batch_size

    :type data: np array
    :type labels: np array
    :type batch_size: int
    :rtype: tuple of np arrays
    """
    random_indices = np.random.randint(data.shape[0], size=batch_size)
    return data[random_indices], labels[random_indices]


def get_data():
    """
    Function to get all the datasets from the pickle in

    https://www.dropbox.com/s/t5l172b417p9pf1/notMNIST.zip?dl=0

    it is require that the pickle is downloaded.
    The scrip download.sh should be run before call this function

    :rtype train_dataset: np arrays
    :rtype train_labels: np arrays
    :rtype valid_dataset: np arrays
    :rtype train_labels: np arrays
    :rtype valid_labels: np arrays
    :rtype test_dataset: np arrays
    :rtype test_labels: np arrays
    """
    currentdir = os.path.dirname(__file__)
    filepath = os.path.join(currentdir, "data")
    filepath = os.path.join(filepath, "notMNIST.pickle")
    assert os.path.exists(filepath)
    with open(filepath, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = one_hot(save['train_labels'])
        valid_dataset = save['valid_dataset']
        valid_labels = one_hot(save['valid_labels'])
        test_dataset = save['test_dataset']
        test_labels = one_hot(save['test_labels'])
        del save
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels


def plot15images(images, cls_true, img_shape, cls_pred=None):
    """
    Function to show 15 images with their respective classes.
    If cls_pred is an array, you can see the image and the prediction.

    :type images: np array
    :type cls_true: np array
    :type img_shape: np array
    :type cls_prediction: None or np array
    """
    assert len(images) == len(cls_true) == 15
    fig, axes = plt.subplots(3, 5, figsize=(11, 11))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def plotconfusion(truth, predictions):
    """
    Function to plot the confusion fuction between the
    truth and predictions array.

    :type truth: np array
    :type predictions: np array
    """
    cm = ConfusionMatrix(truth, predictions)
    plt.figure(figsize=(10, 10))
    cm.plot(backend='seaborn')
    plt.show()