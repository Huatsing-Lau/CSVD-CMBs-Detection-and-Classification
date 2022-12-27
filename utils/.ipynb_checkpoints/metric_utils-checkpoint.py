import pandas as pd
import numpy as np


def bootstrap_score(func, y_true, y_pred, classes, bootstraps = 100, fold_size = 1000):
    '''
    boottrap的方式计算指标，返回一个数组。
    输入：
        func: callable score function
        y_true不是onehot
    '''
    statistics = np.zeros((len(classes), bootstraps))

    for k,c in enumerate(classes):
        df = pd.DataFrame(columns=['y_true', 'y_pred'])
        df.loc[:, 'y_true'] = y_true
        df.loc[:, 'y_pred'] = y_pred
        df_pos = df[df.y_true == 1]
        df_neg = df[df.y_true == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            # 类别等比采样
            pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)
            
            y_sample = np.concatenate([pos_sample.y_true.values, neg_sample.y_true.values])
            pred_sample = np.concatenate([pos_sample.y_pred.values, neg_sample.y_pred.values])
            score = func(y_sample, pred_sample)
            statistics[k][i] = score
    return statistics



# +
"""Multiclass"""

__author__ = "Guillaume Genthial"

import numpy as np
# from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix
from sklearn.metrics import confusion_matrix

def precision(labels, predictions, num_classes, pos_indices=None,
              weights=None, average='micro'):
    """Multi-class precision metric for Tensorflow
    Parameters
    ----------
    labels : Array of np.int32 or np.int64
        The true labels
    predictions : Array of np.int32 or np.int64
        The predictions, same shape as labels
    num_classes : int
        The number of classes
    pos_indices : list of int, optional
        The indices of the positive classes, default is all
    weights : Array of np.int32, optional
        Mask, must be of compatible shape with labels
    average : str, optional
        'micro': counts the total number of true positives, false
            positives, and false negatives for the classes in
            `pos_indices` and infer the metric from it.
        'macro': will compute the metric separately for each class in
            `pos_indices` and average. Will not account for class
            imbalance.
        'weighted': will compute the metric separately for each class in
            `pos_indices` and perform a weighted average by the total
            number of true labels for each class.
    Returns
    -------
    tuple of (scalar float Tensor, update_op)
    """
    import pdb
    pdb.set_trace()
    cm = confusion_matrix(y_true=labels, y_pred=predictions, labels=pos_indices, sample_weight=weights)
#     cm, op = _streaming_confusion_matrix(
#         labels, predictions, num_classes, weights)
    pr, _, _ = metrics_from_confusion_matrix(cm, pos_indices, average=average)
#     op, _, _ = metrics_from_confusion_matrix(
#         op, pos_indices, average=average)
    return pr#(pr, op)


def recall(labels, predictions, pos_indices=None, weights=None,
           average='micro'):
    """Multi-class recall metric for Tensorflow
    Parameters
    ----------
    labels : Array of np.int32 or np.int64
        The true labels
    predictions : Array of np.int32 or np.int64
        The predictions, same shape as labels
    num_classes : int
        The number of classes
    pos_indices : list of int, optional
        The indices of the positive classes, default is all
    weights : Array of np.int32, optional
        Mask, must be of compatible shape with labels
    average : str, optional
        'micro': counts the total number of true positives, false
            positives, and false negatives for the classes in
            `pos_indices` and infer the metric from it.
        'macro': will compute the metric separately for each class in
            `pos_indices` and average. Will not account for class
            imbalance.
        'weighted': will compute the metric separately for each class in
            `pos_indices` and perform a weighted average by the total
            number of true labels for each class.
    Returns
    -------
    tuple of (scalar float Tensor, update_op)
    """
    cm = confusion_matrix(y_true=labels, y_pred=predictions, labels=pos_indices, sample_weight=weights)
#     cm, op = _streaming_confusion_matrix(
#         labels, predictions, num_classes, weights)
    _, re, _ = metrics_from_confusion_matrix(
        cm, pos_indices, average=average)
    _, op, _ = metrics_from_confusion_matrix(
        op, pos_indices, average=average)
    return (re, op)


def f1(labels, predictions, num_classes, pos_indices=None, weights=None,
       average='micro'):
    return fbeta(labels, predictions, num_classes, pos_indices, weights,
                 average)


def fbeta(labels, predictions, num_classes, pos_indices=None, weights=None,
          average='micro', beta=1):
    """Multi-class fbeta metric for Tensorflow
    Parameters
    ----------
    labels : Array of np.int32 or np.int64
        The true labels
    predictions : Array of np.int32 or np.int64
        The predictions, same shape as labels
    num_classes : int
        The number of classes
    pos_indices : list of int, optional
        The indices of the positive classes, default is all
    weights : Array of np.int32, optional
        Mask, must be of compatible shape with labels
    average : str, optional
        'micro': counts the total number of true positives, false
            positives, and false negatives for the classes in
            `pos_indices` and infer the metric from it.
        'macro': will compute the metric separately for each class in
            `pos_indices` and average. Will not account for class
            imbalance.
        'weighted': will compute the metric separately for each class in
            `pos_indices` and perform a weighted average by the total
            number of true labels for each class.
    beta : int, optional
        Weight of precision in harmonic mean
    Returns
    -------
    tuple of (scalar float Tensor, update_op)
    """
    cm = confusion_matrix(y_true=labels, y_pred=predictions, labels=pos_indices, sample_weight=weights)
#     cm, op = _streaming_confusion_matrix(
#         labels, predictions, num_classes, weights)
    _, _, fbeta = metrics_from_confusion_matrix(
        cm, pos_indices, average=average, beta=beta)
    _, _, op = metrics_from_confusion_matrix(
        op, pos_indices, average=average, beta=beta)
    return (fbeta, op)


def safe_div(numerator, denominator):
    """Safe division, return 0 if denominator is 0"""
    numerator, denominator = numerator.astype(np.float32), denominator.astype(np.float32)
    zeros = np.zeros_like(numerator, dtype=numerator.dtype)
    denominator_is_zero = np.equal(denominator, zeros)
    return np.where(denominator_is_zero, zeros, numerator / denominator)


def pr_re_fbeta(cm, pos_indices, beta=1):
    """Uses a confusion matrix to compute precision, recall and fbeta"""
    num_classes = cm.shape[0]
    neg_indices = [i for i in range(num_classes) if i not in pos_indices]
    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[neg_indices, neg_indices] = 0
    diag_sum = np.mean(np.diag(cm * cm_mask))

    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[:, neg_indices] = 0
    tot_pred = np.mean(cm * cm_mask)

    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[neg_indices, :] = 0
    tot_gold = np.mean(cm * cm_mask)

    pr = safe_div(diag_sum, tot_pred)
    re = safe_div(diag_sum, tot_gold)
    fbeta = safe_div((1. + beta**2) * pr * re, beta**2 * pr + re)

    return pr, re, fbeta


def metrics_from_confusion_matrix(cm, pos_indices=None, average='micro',
                                  beta=1):
    """Precision, Recall and F1 from the confusion matrix
    Parameters
    ----------
    cm : Array of type np.int32, of shape (num_classes, num_classes)
        The streaming confusion matrix.
    pos_indices : list of int, optional
        The indices of the positive classes
    beta : int, optional
        Weight of precision in harmonic mean
    average : str, optional
        'micro', 'macro' or 'weighted'
    """
    num_classes = cm.shape[0]
    if pos_indices is None:
        pos_indices = [i for i in range(num_classes)]

    if average == 'micro':
        return pr_re_fbeta(cm, pos_indices, beta)
    elif average in {'macro', 'weighted'}:
        precisions, recalls, fbetas, n_golds = [], [], [], []
        for idx in pos_indices:
            pr, re, fbeta = pr_re_fbeta(cm, [idx], beta)
            precisions.append(pr)
            recalls.append(re)
            fbetas.append(fbeta)
            cm_mask = np.zeros([num_classes, num_classes])
            cm_mask[idx, :] = 1
            n_golds.append( np.mean(cm * cm_mask, dtype=np.float32) )

        if average == 'macro':
            pr = np.mean(precisions)
            re = np.mean(recalls)
            fbeta = np.mean(fbetas)
            return pr, re, fbeta
        if average == 'weighted':
            n_gold = np.mean(n_golds)
            pr_sum = sum(p * n for p, n in zip(precisions, n_golds))
            pr = safe_div(pr_sum, n_gold)
            re_sum = sum(r * n for r, n in zip(recalls, n_golds))
            re = safe_div(re_sum, n_gold)
            fbeta_sum = sum(f * n for f, n in zip(fbetas, n_golds))
            fbeta = safe_div(fbeta_sum, n_gold)
            return pr, re, fbeta

    else:
        raise NotImplementedError()

        
def _streaming_confusion_matrix(labels, predictions, num_classes, weights=None):
    """
    Calculate a streaming confusion matrix.

    Calculates a confusion matrix. For estimation over a stream of data,
    the function creates an  `update_op` operation.

    Args:
    labels: A `Tensor` of ground truth labels with shape [batch size] and of
        type `int32` or `int64`. The tensor will be flattened if its rank > 1.
    predictions: A `Tensor` of prediction results for semantic labels, whose
        shape is [batch size] and type `int32` or `int64`. The tensor will be
        flattened if its rank > 1.
    num_classes: The possible number of labels the prediction task can
        have. This value must be provided, since a confusion matrix of
        dimension = [num_classes, num_classes] will be allocated.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
        be either `1`, or the same as the corresponding `labels` dimension).

    Returns:
    total_cm: A `Tensor` representing the confusion matrix.
    update_op: An operation that increments the confusion matrix.
    """
    # Local variable to accumulate the predictions in the confusion matrix.
    total_cm = metric_variable(
        [num_classes, num_classes], dtypes.float64, name='total_confusion_matrix')

    # Cast the type to int64 required by confusion_matrix_ops.
    predictions = math_ops.cast(predictions, dtypes.int64)
    labels = math_ops.cast(labels, dtypes.int64)
    num_classes = math_ops.cast(num_classes, dtypes.int64)

    # Flatten the input if its rank > 1.
    if predictions.get_shape().ndims > 1:
        predictions = array_ops.reshape(predictions, [-1])

    if labels.get_shape().ndims > 1:
        labels = array_ops.reshape(labels, [-1])

    if (weights is not None) and (weights.get_shape().ndims > 1):
        weights = array_ops.reshape(weights, [-1])

    # Accumulate the prediction to current confusion matrix.
    current_cm = confusion_matrix.confusion_matrix(
        labels, predictions, num_classes, weights=weights, dtype=dtypes.float64)
    update_op = state_ops.assign_add(total_cm, current_cm)
    return total_cm, update_op

