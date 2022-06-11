import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, r2_score


def plot_pdp(estimator, estimator_cd, X, var_idx, var_thr=10, var_name = "feature", ylim=None, n_split=100):
    """ the function is only valid for one variable explanation.
    :Args:
        - estimator, model
        - estimator_cd (str), either classification or regression
        - X (numpy.array), input features
        - var_idx, the target feature index
        - var_thr (int), the threshold to decide whether a feature is ordinal or numerical variable
            For example, if we set var_thr = 10, then if there is 10 unique values or less from the variable
            It will be defined as an ordinal variable
        - ylim (tuple), to set up the y limits of the plot
        - n_split (int), define the resolution of feature if var_cd is "number"
    :Return:
        - plt object
    """
    X_pdp = dict()
    y_pdp = dict()
    var_unique = np.unique(X[:,var_idx]) # target feature's unique values
    
    # some testing 
    assert isinstance(var_thr,int), 'var_thr has to be an integer'
    assert estimator_cd in ('classification','regression'), 'doesn\'t support your estimator type'
    
    # generate the pdp's x-axis
    if len(var_unique) > var_thr:
        plot_cd = 'line'
        ntile = np.linspace(start = 0, stop = 1, num = n_split + 1)
        var_label = [np.quantile(X[:,var_idx], i) for i in ntile]
    else:
        plot_cd = 'bar'
        var_label = var_unique
    
    # generate pdp's y-axis
    for i in range(len(var_label)):
        X_pdp[var_label[i]] = X.copy()
        X_pdp[var_label[i]][:,var_idx] = var_label[i]
        if estimator_cd == 'regression':
            y_pdp[var_label[i]] = np.mean(
                estimator.predict(X_pdp[var_label[i]])
                )
        elif estimator_cd == 'classification':
            y_pdp[var_label[i]] = np.mean(
                # get class = 1 probability
                estimator.predict_proba(X_pdp[var_label[i]]).reshape(-1,2)[:,1]
                # get the actual prediction value like 0 or 1
                # estimator.predict(X_pdp[var_label[i]])
                )
    
    # plotting
    f,ax = plt.subplots(figsize=(7,3))
    if ylim is not None:
        plt.ylim(*ylim)
    if plot_cd == 'bar':
        ax.bar(x = list(y_pdp.keys()), height = list(y_pdp.values()), width = 0.5, align = 'center', color = "#2492ff")
        for k,v in y_pdp.items():
            ax.text(k, v, np.round(v,3), horizontalalignment='center')
    elif plot_cd == 'line':
        ax.plot(list(y_pdp.keys()), list(y_pdp.values()), '.-', color = "#2492ff")
    ax.set_title(f"Partial Dependency Plot - {var_name}")
    ax.set_xlabel(var_name)
    ax.set_ylabel("Average Predicted Value")
    
    return plt


def plot_ale(estimator, estimator_cd, X, var_idx, var_cd="numerical", var_name = "feature", ylim=None, n_split=100, is_centre=True):
    """ the function is only valid for one variable explanation.
    NOTE: we can improve the function performance based on how we bucketise X. We don't necessarily need to replicate X multiple times.
    :Args:
        - estimator, model
        - estimator_cd (str), either classification or regression
        - X (numpy.array), input features. Categorical variables require to be converted to numerical variables.
        - var_idx, the target feature index
        - var_thr (str), "categorical" or "numerical"
        - ylim (tuple), to set up the y limits of the plot
        - n_split (int), define the resolution of feature if var_cd is "number"
        - is_centre (bool), a flag to decide whether to plot main effect with or without average effect
    :Return:
        - plt object
    """

    def replace_value(X, var_idx, value_to_replace):
        # replace values of a column of X numpy array
        X[:,var_idx] = value_to_replace 
        return X
    
    # some testing 
    assert estimator_cd in ('classification','regression'), 'doesn\'t support your estimator type'
    assert var_cd in ('numerical','categorical'), 'doesn\'t support your variable type'

    # empty dictionaries
    X_ale = dict()
    y_ale = dict()
    y_ale_acc = dict()
    y_ale_output = dict()

    # split data into buckets, to get ready for local effect calculation for each bucket
    # categorial or numerical
    
    if var_cd == 'numerical':
        ntile = np.linspace(start = 0, stop = 1, num = n_split + 1)
        var_label = [np.quantile(X[:,var_idx], i) for i in ntile]
        var_parts = {(i+1):(var_label[i], var_label[i+1]) for i in range(n_split)}
        delta = np.mean([v[1] - v[0] for v in var_parts.values()]) # averaged gap between upper and lower bound
        var_parts[0] = (var_label[0] - delta, var_label[0]) 
        assert len(var_label) == len(np.unique(var_label)), 'cant generate ntile given duplicated values'
    else:
        var_unique = np.unique(X[:,var_idx]) # target feature's unique values
        var_label = np.sort(var_unique)
        var_parts = {(i+1):(var_label[i], var_label[i+1]) for i in range(len(var_label) - 1)}
        delta = 1
        var_parts[0] = (var_label[0] - delta, var_label[0]) 
    
    # calculate the local effect for each bucket
    for i in range(len(var_parts)):
        mask = var_parts[i]
        X_ale[i] = X[(X[:,var_idx]>mask[0]) & (X[:,var_idx]<=mask[1]),:]
        if estimator_cd == 'regression':
            y_ale[i] = np.sum(\
                estimator.predict(replace_value(X_ale[i], var_idx, mask[1])) \
                - estimator.predict(replace_value(X_ale[i], var_idx, mask[0]))) \
                / X_ale[i].shape[0]
        else:
            y_ale[i] = np.sum(\
                estimator.predict(replace_value(X_ale[i], var_idx, mask[1])) - \
                estimator.predict(replace_value(X_ale[i], var_idx, mask[0])) \
            ) / X_ale[i].shape[0]
        
        # accumulated result
        y_ale_acc[i] = np.sum([v for v in y_ale.values()])
    
    # centralise result or not
    bias = 1 / X.shape[0] * np.dot(np.array([item.shape[0] for item in X_ale.values()]), np.array(list(y_ale_acc.values())))
    # we chose upper bound as our x-axis
    if is_centre:
        y_ale_output = {var_parts[k][1]:(v - bias) for k,v in y_ale_acc.items()}
    else:
        y_ale_output = {var_parts[k][1]:v for k,v in y_ale_acc.items()}

    # plotting
    f,ax = plt.subplots(figsize=(7,3))
    if ylim is not None:
        plt.ylim(*ylim)

    ax.plot(list(y_ale_output.keys()), list(y_ale_output.values()), '.-', color = "#2492ff")
    ax.set_title(f"Accumulated Local Effect - {var_name}")
    ax.set_xlabel(var_name)
    ax.set_ylabel("Effect")
    
    return plt