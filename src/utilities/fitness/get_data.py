import io
from os import path

import numpy as np
import pandas as pd

from algorithm.parameters import params
from scipy.io import arff


def get_Xy_train_test_separate(train_filename, test_filename, skip_header=0):
    """
    Read in training and testing data files, and split each into X
    (all columns up to last) and y (last column). The data files should
    contain one row per training example.
    
    :param train_filename: The file name of the training dataset.
    :param test_filename: The file name of the testing dataset.
    :param skip_header: The number of header lines to skip.
    :return: Parsed numpy arrays of training and testing input (x) and
    output (y) data.
    """

    if params['DATASET_DELIMITER']:
        # Dataset delimiter has been explicitly specified.
        delimiter = params['DATASET_DELIMITER']

    else:
        # Try to auto-detect the field separator (i.e. delimiter).
        f = open(train_filename)
        for line in f:
            if line.startswith("#") or len(line) < 2:
                # Skip excessively short lines or commented out lines.
                continue

            else:
                # Set the delimiter.
                if "\t" in line:
                    delimiter = "\t"
                    break
                elif "," in line:
                    delimiter = ","
                    break
                elif ";" in line:
                    delimiter = ";"
                    break
                elif ":" in line:
                    delimiter = ":"
                    break
                else:
                    print(
                        "Warning (in utilities.fitness.get_data.get_Xy_train_test_separate)\n"
                        "Warning: Dataset delimiter not found. "
                        "Defaulting to whitespace delimiter.")
                    delimiter = " "
                    break
        f.close()

    # Read in all training data.
    train_Xy = np.genfromtxt(train_filename, skip_header=skip_header,
                             delimiter=delimiter)

    try:
        # Separate out input (X) and output (y) data.
        train_X = train_Xy[:, :-1] # all columns but last
        train_y = train_Xy[:, -1]  # last column

    except IndexError:
        s = "utilities.fitness.get_data.get_Xy_train_test_separate\n" \
            "Error: specified delimiter '%s' incorrectly parses training " \
            "data." % delimiter
        raise Exception(s)

    if test_filename:
        # Read in all testing data.
        test_Xy = np.genfromtxt(test_filename, skip_header=skip_header,
                                delimiter=delimiter)

        # Separate out input (X) and output (y) data.
        test_X = test_Xy[:, :-1] # all columns but last
        test_y = test_Xy[:, -1]  # last column

    else:
        test_X, test_y = None, None

    return train_X, train_y, test_X, test_y

def read_arff(file):
    """
    Read an arff dataset and returns the input variables and output ones

    :param file: Path to the file to be read
    :return: The parsed data contained in the dataset file in a tuple (input, output)
    """
    try:
        data, metadata = arff.arffread.loadarff(file)
    except UnicodeEncodeError:
        with open(file, 'r') as f:
            content = ''.join(f.readlines())
            content = content.replace('á', 'a')
            content = content.replace('é', 'e')
            content = content.replace('í', 'i')
            content = content.replace('ó', 'o')
            content = content.replace('ú', 'u')
            content = content.replace('ñ', 'n')
            with io.StringIO(content) as f2:
                data, metadata = arff.loadarff(f2)

    data = pd.DataFrame(data)
    data = \
        data.apply(lambda x: x.str.decode('utf-8') if x.dtype == 'object' else x)
    data[data == '?'] = np.nan
    num_in_features = data.shape[1] - 1
    input_data = data.iloc[:, :num_in_features]
    output = data.iloc[:, num_in_features]

    return input_data, output

def get_data(train, test):
    """
    Return the training and test data for the current experiment.
    
    :param train: The desired training dataset.
    :param test: The desired testing dataset.
    :return: The parsed data contained in the dataset files.
    """

    # Get the path to the training dataset.
    train_set = path.join("..", "datasets", train)

    if test:
        # Get the path to the testing dataset.
        test_set = path.join("..", "datasets", test)

    else:
        # There is no testing dataset used.
        test_set = None

    if train_set.endswith('.arff'):
        test_in, test_out = None, None
        training_in, training_out = read_arff(train_set)

        if test_set is not None:
            test_in, test_out = read_arff(test_set)
    else:
        # Read in the training and testing datasets from the specified files.
        training_in, training_out, test_in, \
        test_out = get_Xy_train_test_separate(train_set, test_set, skip_header=1)

    return training_in, training_out, test_in, test_out
