import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, r2_score
from scipy.stats import chi2_contingency


#################################################################
# data pre-processing
def generate_dict(df):
    """ it is to create a data dictionary based on type of columns
        the metrics inlcude row counts, unique values for categorical variables, data types,
        mean, max, min, std, etc
    :Args:
        - df (pandas.DataFrame), the target dataframe to generate data dictionary
    :Returns:
        - dict_df (pandas.DataFrame), the stats of the target dataframe
    """
    dict_df = pd.DataFrame(df.isnull().sum()).rename(columns={0:"n_of_nulls"})
    dict_df['row_counts'] = df.count(axis=0)
    dict_df['unique_values'] = df.nunique(axis=0)
    dict_df['dtypes'] = df.dtypes
    dict_df['min_value'] = df.min(numeric_only=True)
    dict_df['mean'] = df.mean(numeric_only=True)
    dict_df['max_value'] = df.max(numeric_only=True)
    dict_df = dict_df.merge(df.describe().transpose()[["std", "25%", "50%", "75%"]]\
        , how='left', left_index=True, right_index=True)
    # extract unique items from dataframe
    unique_items = dict()
    for item in dict_df[(dict_df["unique_values"] <= 10) & (dict_df["dtypes"]=='object')].index.to_list():
        try:
            unique_items[item] = ", ".join(df[item].unique().tolist())
        except Exception as e:
            print(f"item '{item}':\n{e}")
    dict_df['unique_items'] = pd.Series(unique_items)
    return dict_df


#################################################################
# feature selection
def chisquare_test(df, col_x, col_y, thres=0.05, significant_flag=False):
    """ the function is to test the col_x on col_y whether there is no difference on col_x for the two populations
        can only be used between categorical variables col_x on col_y
    :args: 
        - df, dataframe
        - col_x, str, the column we want to prove that has no significance to col_y
        - col_y, str, response column
        - thres, float, the confidence interval
        - significant_flag, bool, do we want to return decision of statistical significance or not
    :return:
        - bool, can we reject our null hypothesis that col_x has no impact on col_y
    """
    try:
        test_df = df.groupby(by=[col_x,col_y])[col_y].count().unstack(col_y).fillna(0)
        chi2, p, dof, expected = chi2_contingency(test_df.to_numpy())
        # if significant_flag is true, just return bool else return p-value
        if significant_flag:
            return True if p <= thres else False
        else:
            return p
    except Exception as e:
        print(e)


#################################################################
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


#################################################################
# misc
def get_decile(target, n_split = 10):
    """ the function is to split a numpy 1d array into buckets by n_split
    :Args:
        - target (1D numpy.array), the target numerical numpy array to split
        - n_split (int), how many splits we want from the numpy array
    :Return:
        - outcome (list), the split
    """
    outcome = list()    # placeholder
    ntile = np.linspace(start = 0, stop = 1, num = n_split + 1)
    var_label = [np.quantile(target, i) for i in ntile]
    var_parts = {(i+1):(var_label[i], var_label[i+1]) for i in range(n_split)}
    delta = np.mean([v[1] - v[0] for v in var_parts.values()]) 
    var_parts[0] = (var_label[0] - delta, var_label[0]) 
    for item in target:
        for k,v in var_parts.items():
            if (item > v[0]) & (item <= v[1]):
                outcome.append(k)
                continue
    return outcome
