import io
from os import path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from algorithm.parameters import params
from scipy.io import arff

def impute_missing_values(train_set, test_set):

    def get_impute_values(a_series):
        if (a_series.dtype == 'object' or a_series.dtype == 'category' or a_series.dtype == 'bool'):
            return a_series.mode()
        else:
            return a_series.median()

    imputing_values = train_set.apply(get_impute_values)
    train_nullvalues = train_set.isnull()
    train_set = train_set.copy()
    if type(imputing_values) == pd.DataFrame:
        train_set.fillna(imputing_values.iloc[0,:], inplace=True)
    else:
        train_set.fillna(imputing_values, inplace=True)
    labels=list(train_set.columns) + [i+'_missing' for i in train_nullvalues.columns]
    train_set = pd.concat([train_set, train_nullvalues], axis=1)
    train_set.columns = labels

    if test_set is not None:
        test_nullvalues = test_set.isnull()
        test_set = test_set.copy()
        if type(imputing_values) == pd.DataFrame:
            test_set.fillna(imputing_values.iloc[0,:], inplace=True)
        else:
            test_set.fillna(imputing_values, inplace=True)
        test_set = pd.concat([test_set, test_nullvalues], axis=1,keys=labels)
        test_set.columns = labels

    #Drop columns with just one value
    for col in train_set.columns:
        if len(train_set[col].unique()) == 1:
            train_set.drop(col, inplace=True, axis=1)

            if test_set is not None:
                test_set.drop(col, inplace=True, axis=1)

    return train_set, test_set


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
                data_metadata = arff.loadarff(f2)

    data,metadata = data_metadata
    # arff.dump(data_metadata, f)
    data = pd.DataFrame(data)
    data = \
        data.apply(lambda x: x.str.decode('utf-8') if x.dtype == 'object' else x)
    data[data == '?'] = np.nan
    num_in_features = data.shape[1] - 1
    input_data = data.iloc[:, :num_in_features]
    output = data.iloc[:, num_in_features]

    return input_data, output, metadata

def get_data(train, test):
    """
    Return the training and test data for the current experiment.
    
    :param train: The desired training dataset.
    :param test: The desired testing dataset. If filename, then read it; FIXME the rest of the comment might be wrong. I think I moved to using a single integer
		if tuple or list [i,j], then j means the number of folds and i means the index of the test fold
    :return: The parsed data contained in the dataset files.
    """

    # Get the path to the training dataset.
    train_set = path.join("..", "datasets", train)

    if test and isinstance(test, str) and path.isfile(path.join("..","datasets",test)):
        # Get the path to the testing dataset.
        test_set = path.join("..", "datasets", test)

    else:
        # There is no testing dataset used.
        test_set = None

    if train_set.endswith('.arff'):
        test_in, test_out = None, None
        training_in, training_out, metadata = read_arff(train_set)

        if test_set is None and isinstance(test, str):
            try:
                test = eval(test)
            except:
                test = None

        if 'CROSS_VALIDATION' in params and params['CROSS_VALIDATION'] and isinstance(test, int):
            random_state = None

            assert 'CROSS_VALIDATION_SEED' in params

            if 'CROSS_VALIDATION_SEED' in params:
                random_state = params['CROSS_VALIDATION_SEED']

            folds_generator = StratifiedKFold(n_splits=params['CROSS_VALIDATION'], shuffle=True, random_state=random_state)
            for _, (train_index, test_index) in zip(range(test + 1), folds_generator.split(training_in, training_out)):
                pass
            test_in, test_out = training_in.iloc[test_index], training_out[test_index]
            training_in, training_out = training_in.iloc[train_index], training_out[train_index]
        elif test_set is not None:
            test_in, test_out, metadata = read_arff(test_set)

        if params.get('IMPUTE_MISSING', False):
            training_in, test_in = impute_missing_values(training_in, test_in)

    else:
        # Read in the training and testing datasets from the specified files.
        training_in, training_out, test_in, \
        test_out = get_Xy_train_test_separate(train_set, test_set, skip_header=1)

    return training_in, training_out, test_in, test_out, metadata
