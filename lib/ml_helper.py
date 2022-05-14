import matplotlib.pyplot as plt
import numpy as np
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
    