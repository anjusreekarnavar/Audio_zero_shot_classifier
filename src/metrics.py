import numpy as np


def reciprocal_rank(prediction, truth):
    """
    Reciprocal rank of the first relevant class.

    Parameters
    ----------
    prediction : array-like
        Predicted abundance vector.

    truth : array-like
        Ground-truth abundance vector.

    Returns
    -------
    float
        Reciprocal rank.
    """
    prediction = np.asarray(prediction)
    truth = np.asarray(truth)

    # Classes with non-zero ground-truth abundance are relevant.
    relevant = truth > 0

    # Sort classes from largest predicted abundance to smallest.
    ranking = np.argsort(-prediction)

    for rank, class_index in enumerate(ranking, start=1):
        if relevant[class_index]:
            return 1.0 / rank

    return 0.0


def recall_at_k(prediction, truth, k):
    """
    Fraction of true classes appearing among the top-k predictions.
    """
    prediction = np.asarray(prediction)
    truth = np.asarray(truth)

    relevant = np.flatnonzero(truth > 0)

    if len(relevant) == 0:
        return 0.0

    top_k = np.argsort(-prediction)[:k]

    retrieved = np.intersect1d(top_k, relevant).size

    return retrieved / len(relevant)


def total_variation(prediction, truth):
    """
    Total Variation distance between predicted and true abundance vectors.

    Both vectors should be non-negative and sum to one.
    """
    prediction = np.asarray(prediction, dtype=float)
    truth = np.asarray(truth, dtype=float)

    return 0.5 * np.abs(prediction - truth).sum()
