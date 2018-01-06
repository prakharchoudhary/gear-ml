"""
Applying Decision Trees to the Bank Note Dataset
"""
import numpy as np
from ..tree import DecisionTreeClassifier
from random import seed
from random import randrange
from csv import reader


def load_csv(filename):
    '''
    Load a CSV file
    '''
    file = open(filename, "rb")
    lines = reader(file)
    dataset = list(lines)
    return dataset


def str_column_to_float(dataset, column):
    '''
    Convert string column to float
    '''
    for row in dataset:
        row[column] = float(row[column].strip())


def cross_validation_split(dataset, n_folds):
    '''
    Split a dataset into k folds
    '''
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def accuracy_metric(actual, predicted):
    '''
    Calculate accuracy percentage
    '''
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    '''
    Evaluate an algorithm using a cross validation split

    Params
    ------
    dataset: All samples
    algorithm: Algorithm class
    '''
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        # predicted = algorithm(train_set, test_set, *args)
        tree = algorithm(train, max_depth=max_depth,
                         min_size=min_size)
        predictions = list()
        for row in test_set:
            prediction = tree.predict(tree, row)
            predictions.append(prediction)

        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predictions)
        scores.append(accuracy)
    return scores

if __name__ == '__main__':
    seed(1)
    filename = ''
    dataset = load_csv(filename)
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    n_folds = 5
    max_depth = 5
    min_size = 10
    scores = evaluate_algorithm(dataset, algorithm,
                                n_folds, max_depth, min_size)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
