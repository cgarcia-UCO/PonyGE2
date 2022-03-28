import warnings

import numpy as np
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import precision_score as sklearn_precision_score
from sklearn.metrics import recall_score as sklearn_recall_score


def mae(y, yhat):
    """
    Calculate mean absolute error between inputs.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The mean absolute error.
    """

    return np.mean(np.abs(y - yhat))


# Set maximise attribute for mae error metric.
mae.maximise = False


def rmse(y, yhat):
    """
    Calculate root mean square error between inputs.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The root mean square error.
    """

    return np.sqrt(np.mean(np.square(y - yhat)))


# Set maximise attribute for rmse error metric.
rmse.maximise = False


def mse(y, yhat):
    """
    Calculate mean square error between inputs.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The mean square error.
    """

    return np.mean(np.square(y - yhat))


# Set maximise attribute for mse error metric.
mse.maximise = False


def hinge(y, yhat):
    """
    Hinge loss is a suitable loss function for classification.  Here y is
    the true values (-1 and 1) and yhat is the "raw" output of the individual,
    ie a real value. The classifier will use sign(yhat) as its prediction.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The hinge loss.
    """

    # Deal with possibility of {-1, 1} or {0, 1} class label convention
    y_vals = set(y)
    # convert from {0, 1} to {-1, 1}
    if 0 in y_vals:
        y[y == 0] = -1

    # Our definition of hinge loss cannot be used for multi-class
    assert len(y_vals) == 2

    # NB not np.max. maximum does element-wise max.  Also we use the
    # mean hinge loss rather than sum so that the result doesn't
    # depend on the size of the dataset.
    return np.mean(np.maximum(0, 1 - y * yhat))


# Set maximise attribute for hinge error metric.
hinge.maximise = False


def f1_score(y, yhat):
    """
    The F_1 score is a metric for classification which tries to balance
    precision and recall, ie both true positives and true negatives.
    For F_1 score higher is better.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The f1 score.
    """

    # if phen is a constant, eg 0.001 (doesn't refer to x), then yhat
    # will be a constant. that will break f1_score. so convert to a
    # constant array.
    if not isinstance(yhat, np.ndarray) or len(yhat.shape) < 1:
        yhat = np.ones_like(y) * yhat

    # Deal with possibility of {-1, 1} or {0, 1} class label
    # convention.  FIXME: would it be better to canonicalise the
    # convention elsewhere and/or create user parameter to control it?
    # See https://github.com/PonyGE/PonyGE2/issues/113.
    y_vals = set(y)
    # convert from {-1, 1} to {0, 1}
    if -1 in y_vals:
        y[y == -1] = 0

    # We binarize with a threshold, so this cannot be used for multi-class
    assert len(y_vals) == 2

    # convert real values to boolean {0, 1} with a zero threshold
    # Only if they are numbers (not string like 'Yes'/'No' o 'Positive'/'Negative'
    if issubclass(yhat.dtype.type, np.number):
        yhat = (yhat > 0)

    with warnings.catch_warnings():
        # if we predict the same value for all samples (trivial
        # individuals will do so as described above) then f-score is
        # undefined, and sklearn will give a runtime warning and
        # return 0. We can ignore that warning and happily return 0.
        warnings.simplefilter("ignore")
        return sklearn_f1_score(y, yhat, average="weighted")


# Set maximise attribute for f1_score error metric.
f1_score.maximise = True


def Hamming_error(y, yhat):
    """
    The number of mismatches between y and yhat. Suitable
    for Boolean problems and if-else classifier problems.
    Assumes both y and yhat are binary or integer-valued.
    """
    return np.sum(y != yhat)


Hamming_error.maximise = False


def precision_score(y, yhat):
    """
    The precision is the ratio tp / (tp + fp) where tp is 
    the number of true positives and fp the number of false 
    positives. The precision is intuitively the ability of 
    the classifier not to label as positive a sample that 
    is negative. The best value is 1 and the worst value is 0.
    See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return The precision score.
    """

    # if phen is a constant, eg 0.001 (doesn't refer to x), then yhat
    # will be a constant. that will break precision_score. so convert to a
    # constant array.
    if not isinstance(yhat, np.ndarray) or len(yhat.shape) < 1:
        yhat = np.ones_like(y) * yhat

    # Deal with possibility of {-1, 1} or {0, 1} class label
    # convention.  FIXME: would it be better to canonicalise the
    # convention elsewhere and/or create user parameter to control it?
    # See https://github.com/PonyGE/PonyGE2/issues/113.
    y_vals = set(y)

    # convert from {-1, 1} to {0, 1}
    if -1 in y_vals:
        y[y == -1] = 0

    # We binarize with a threshold, so this cannot be used for multi-class
    assert len(y_vals) == 2

    # convert real values to boolean {0, 1} with a zero threshold
    # Only if they are numbers (not string like 'Yes'/'No' o 'Positive'/'Negative'
    if issubclass(yhat.dtype.type, np.number):
        yhat = (yhat > 0)

    with warnings.catch_warnings():
        # if we predict the same value for all samples (trivial
        # individuals will do so as described above) then precision is
        # undefined, and sklearn will give a runtime warning and
        # return 0. We can ignore that warning and happily return 0.
        warnings.simplefilter("ignore")
        return sklearn_precision_score(y, yhat, average="binary", pos_label='Si')


# Set maximise attribute for precision_score error metric.
precision_score.maximise = True


def recall_score(y, yhat):
    """
    The recall is the ratio tp / (tp + fn) where tp is
    the number of true positives and fn the number of false
    negatives. The recall is intuitively the ability of
    the classifier to find all the positive samples. The best
    value is 1 and the worst value is 0.
    See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return The recall score.
    """

    # if phen is a constant, eg 0.001 (doesn't refer to x), then yhat
    # will be a constant. that will break recall_score. so convert to a
    # constant array.
    if not isinstance(yhat, np.ndarray) or len(yhat.shape) < 1:
        yhat = np.ones_like(y) * yhat

    # Deal with possibility of {-1, 1} or {0, 1} class label
    # convention.  FIXME: would it be better to canonicalise the
    # convention elsewhere and/or create user parameter to control it?
    # See https://github.com/PonyGE/PonyGE2/issues/113.
    y_vals = set(y)
    # convert from {-1, 1} to {0, 1}
    if -1 in y_vals:
        y[y == -1] = 0

    # We binarize with a threshold, so this cannot be used for multi-class
    assert len(y_vals) == 2

    # convert real values to boolean {0, 1} with a zero threshold
    # Only if they are numbers (not string like 'Yes'/'No' o 'Positive'/'Negative'
    if issubclass(yhat.dtype.type, np.number):
        yhat = (yhat > 0)

    with warnings.catch_warnings():
        # if we predict the same value for all samples (trivial
        # individuals will do so as described above) then recall is
        # undefined, and sklearn will give a runtime warning and
        # return 0. We can ignore that warning and happily return 0.
        warnings.simplefilter("ignore")
        return sklearn_recall_score(y, yhat, average="binary", pos_label='Si')


# Set maximise attribute for recall_score error metric.
recall_score.maximise = True


def precision_and_recall_score(y, yhat):
    """
    Compute the product (precision · recall)
    """
    return precision_score(y, yhat) * recall_score(y, yhat)


# Set maximise attribute for precision_and_recall_score error metric.
precision_and_recall_score.maximise = True

def accuracy(y, yhat):
    return (np.sum(y == yhat) / len(y))

accuracy.maximise = True