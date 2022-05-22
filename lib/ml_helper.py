from operator import is_
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, r2_score

# evaluation
def calc_metrics(model, X, y):
    """ it is to calculate basic metrics for binary classification
    :Args:
        - model (sklearn compatible model object), the model to calculate performance
        - X (numpy array), the input tensor / matrix
        - y (numpy array), the response vector
    :Returns:
        - a dictionary, with accuracy, precision, recall and f1_score
    """
    tn, fp, fn, tp = confusion_matrix(y, model.predict(X)).ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + tn + fn + fp)
    f1_score = 2 * recall * precision / (recall + precision)
    return {'accuracy':accuracy, 'precision':precision, 'recall':recall, 'f1_score':f1_score}


# plot learning curve
# reference: explainable_ml/s4.1_bike_rental.ipynb
def plot_learning_curve(estimator_name, estimator, X, y, ylim=None, cv=None, n_jobs=1,  scoring=None, train_sizes = np.linspace(.1, 1.0, 5)):
    """ the function is to plot learning curve between train and cv datasets. 
        the objective is to visualise fitting condition of training process
    : Args:
        - estimator_name (string), the model name
        - estimator (model), the model object in sklearn that "fits" and "predicts" data
        - X (numpy array), X_train data
        - y (numpy array), y_train data
        - ylim (tuple), decides the y axis upper and lower limits
        - cv (int, cross-validation generator or an iterable), it's to determin cross-validation splitting strategy
        - n_jobs (int), num of threads to work on calculation
        - scoring (string or callable), A str or a scorer callable object/function with signature scorer(estimator, X, y)
        - train_sizes (numpy's linspace object), to indicate the dots to be plotted in learning curve
    : Returns:
        - matplotlib plot with learning curve.
    """
    f, ax = plt.subplots()
    if ylim is not None:
        plt.ylim(*ylim)
    if scoring is None:
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    else:
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)
    
    # calculate confidence interval 
    train_scores_mean = np.mean(train_scores, axis=1)   
    test_scores_mean = np.mean(test_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill_between.html
    # plotting
    ax.fill_between(train_sizes, test_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="#ff9124")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",label="Cross-validation score")
    ax.set_title(f"{estimator_name} Learning Curve", fontsize=14)
    ax.set_xlabel('Training size')
    ax.set_ylabel('Score')
    ax.grid(True)
    ax.legend(loc="best")
    return plt

def plot_actual_and_pred(estimator, X, y):
    """ the function is to plot regression's actual and predicted value
        so that we can see how predicted values spread across the plane
    :Args:
        - estimator, the regressor (from scikit-learn or lightgbm or other libs)
        - X (numpy.array), the inputs
        - y (numpy.array), the responses
    :Return:
        - plot object for visualisation
    """
    f,ax = plt.subplots(figsize=(5,5))
    ax.scatter(y, estimator.predict(X).reshape(-1,1), color="#2492ff", alpha=0.8, marker=".")
    ax.plot(y,y, color="#ff9124", alpha=1, linestyle='-.')
    ax.set_xlim(xmin=0, xmax=8500)
    ax.set_ylim(ymin=0, ymax=8500)
    ax.set_xlabel("actual y")
    ax.set_ylabel("predicted y")
    ax.set_title(f"actual vs predicted response, R^2 {np.round(r2_score(y, estimator.predict(X).reshape(-1,1)),2)}")
    ax.grid(True)
    return plt

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
                estimator.predict_proba(replace_value(X_ale[i], var_idx, mask[1])).reshape(-1,2)[:,1] \
                - estimator.predict_proba(replace_value(X_ale[i], var_idx, mask[0])).reshape(-1,2)[:,1]) \
                / X_ale[i].shape[0]
        
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