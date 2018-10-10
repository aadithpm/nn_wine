#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

def euclidean_distance(a,b):
    diff = a - b
    return np.sqrt(np.dot(diff, diff))

def load_data(csv_filename):
    """ 
    Returns a numpy ndarray in which each row repersents
    a wine and each column represents a measurement. There should be 11
    columns (the "quality" column cannot be used for classificaiton).
    """
    
    wines = np.genfromtxt(csv_filename, delimiter = ';', skip_header = 1)
    return wines[:, :-1]

    """
    # More verbose: (note: import csv beforehand)
    f = open(csv_filename, 'r')
    data_file = csv.reader(f, delimiter = ';')
    wine_data = np.array(list(data_file))[1:]
    wine_data = wine_data[:, :-1]
    wine_data = wine_data.astype(np.float)
    return wine_data
    """

    
def split_data(dataset, ratio = 0.9):
    """
    Return a (train, test) tuple of numpy ndarrays. 
    The ratio parameter determines how much of the data should be used for 
    training. For example, 0.9 means that the training portion should contain
    90% of the data. You do not have to randomize the rows. Make sure that 
    there is no overlap. 
    """
    split = round(dataset.shape[0] * 0.9)
    training_set = dataset[:split, :]
    testing_set = dataset[split:, :]
    return (training_set, testing_set)
    
def compute_centroid(data):
    """
    Returns a 1D array (a vector), representing the centroid of the data
    set. 
    """
    return np.mean(data, axis = 0)
    
def experiment(ww_train, rw_train, ww_test, rw_test):
    """
    Train a model on the training data by creating a centroid for each class.
    Then test the model on the test data. Prints the number of total 
    predictions and correct predictions. Returns the accuracy. 
    """
    ww_centroid = compute_centroid(ww_train)
    rw_centroid = compute_centroid(rw_train)
    
    ww_test_labels = []
    rw_test_labels = []
    # Predictions for white wine
    for row in ww_test:        
        distance_1 = euclidean_distance(row, ww_centroid)
        distance_2 = euclidean_distance(row, rw_centroid)
        if distance_1 <= distance_2:
            ww_test_labels.append('white')
        else:
            ww_test_labels.append('red')
    
    for row in rw_test:       
        distance_1 = euclidean_distance(row, ww_centroid)
        distance_2 = euclidean_distance(row, rw_centroid)
        if distance_1 <= distance_2:
            rw_test_labels.append('white')
        else:
            rw_test_labels.append('red')
    
    accuracy = (ww_test_labels.count('white') / len(ww_test_labels)) + (rw_test_labels.count('red') / len(rw_test_labels))
    accuracy = accuracy / 2
    # accuracy = accuracy * 100 (%)
    
    """
    # Uncomment for values:
    print("Predicting red/white wine based on chemical properties..")
    print("Number of predictions:", len(ww_test_labels) + len(rw_test_labels))
    print("Number of correct predictions:", ww_test_labels.count('white') + rw_test_labels.count('red'))
    print("Number of incorrect predictions:", ww_test_labels.count('red') + rw_test_labels.count('white'))
    print("Accuracy:", accuracy)
    """
    
    return accuracy

def learning_curve(ww_training, rw_training, ww_test, rw_test):
    """
    Perform a series of experiments to compute and plot a learning curve.
    """
    np.random.shuffle(ww_training)
    np.random.shuffle(rw_training)
    accuracies = []
    
    for i in range(1, ww_training.shape[0]):
        accuracies.append(experiment(ww_training[:i,:], rw_training[:i,:], ww_test, rw_test))
        
    return accuracies

def cross_validation(ww_data, rw_data, k):
    """
    Perform k-fold crossvalidation on the data and print the accuracy for each
    fold. 
    """
    ww_datasets = np.array_split(ww_data, k)
    rw_datasets = np.array_split(rw_data, k)
    
    accuracies = []
    
    for i in range(k):
        ww_test = ww_datasets[i]
        rw_test = rw_datasets[i]
        ww_train = ww_datasets[:i] + ww_datasets[i:]
        ww_train = np.vstack(ww_train)
        rw_train = rw_datasets[:i] + rw_datasets[i:]
        rw_train = np.vstack(rw_train)
        accuracies.append(experiment(ww_train, rw_train, ww_test, rw_test))
        
    return sum(accuracies) / len(accuracies)

if __name__ == "__main__":
    
    ww_data = load_data('whitewine.csv')
    rw_data = load_data('redwine.csv')

    ww_train, ww_test = split_data(ww_data, 0.9)
    rw_train, rw_test = split_data(rw_data, 0.9)
    
    accuracy = experiment(ww_train, rw_train, ww_test, rw_test)
    print("Accuracy of the model:", accuracy)
    
    
    accuracies = learning_curve(ww_train, rw_train, ww_test, rw_test)
    x = [i for i in range(1, ww_train.shape[0])]
    y = accuracies
    xnew = np.linspace(min(x), max(x), 300)
    smooth = interp1d(x, y, kind = 'cubic')
    ynew = smooth(xnew)
    plt.plot(xnew, ynew)
    plt.title('Learning curve')
    plt.xlabel('Number of training examples')
    plt.ylabel('Accuracy')
    
    # Changing figure size
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 11.0
    fig_size[1] = 5.0
    plt.rcParams["figure.figsize"] = fig_size
    
    plt.ylim([min(y), max(y)])
    plt.show()
    
    k = 10
    acc = cross_validation(ww_data, rw_data, k)
    print("{}-fold cross-validation accuracy: {}".format(k, acc))
    