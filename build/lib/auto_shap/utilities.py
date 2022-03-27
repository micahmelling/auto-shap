import multiprocessing as mp
import os
from functools import partial

import numpy as np
import pandas as pd

from auto_shap.config import (AMBIGUOUS_REGRESSION_MODEL_NAMES,
                              LINEAR_MODEL_NAMES, TREE_MODEL_NAMES)


def make_directories_if_not_exists(directories_list: list):
    """
    Makes directories in the current working directory if they do not exist.

    :param directories_list: list of directories to create
    """
    for directory in directories_list:
        if not os.path.exists(directory):
            os.makedirs(directory)


def determine_model_qualities(model: callable, linear_model: bool = None, tree_model: bool = None,
                              boosting_model: bool = None, calibrated_model: bool = None,
                              regression_model: bool = False) -> tuple:
    """
    Determines the qualities of a model. That is, if not defined, determines if the model is one of the following in a
    Boolean nature:
    - linear model
    - tree model
    - boosting model
    - calibrated model
    - regression modeling

    If the model is a CalibratedClassifierCV, the first base_estimator is used to determine model qualities.

    :param model: fitted model
    :param linear_model: Boolean of whether model is a linear model, which would employ Linear SHAP
    :param tree_model: Boolean of whether model is a tree-based model, which would employ Tree SHAP
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :param calibrated_model: Boolean of whether or not the model is a CalibratedClassifierCV
    :param regression_model: Boolean of whether or not this is a regression model; if not, it's assumed this is a
    classification model
    :returns: Booleans for calibrated_model, linear_model, tree_model, boosting_model, regression_model
    """
    if not calibrated_model:
        calibrated_model = determine_if_name_in_object('calibrated', model)
    if calibrated_model:
        model = model.calibrated_classifiers_[0].base_estimator
    if not linear_model:
        linear_model = determine_if_any_name_in_object(LINEAR_MODEL_NAMES, model)
    if not tree_model:
        tree_model = determine_if_any_name_in_object(TREE_MODEL_NAMES, model)
    if not boosting_model:
        boosting_model = determine_if_any_name_in_object(['boost', 'gbm'], model)
    if not regression_model:
        regression_model = determine_if_regression_model(AMBIGUOUS_REGRESSION_MODEL_NAMES, model)
    return calibrated_model, linear_model, tree_model, boosting_model, regression_model


def run_shap_explainer(x_df: pd.DataFrame, explainer: callable, boosting_model: bool,
                       regression_model: bool, linear_model: bool) -> np.array:
    """
    Runs the SHAP explainer on a dataframe of predictors (i.e. x_df).

    :param x_df: x dataframe
    :param explainer: SHAP explainer object
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :param regression_model: Boolean of whether or not the explainer is for a regression model; if False, it is assumed
    we are dealing with a classification model
    :param linear_model: Boolean of whether or not the explainer is for a linear model
    """
    if boosting_model or regression_model or linear_model:
        shap_values = explainer.shap_values(x_df)
        if len(np.shape(shap_values)) == 3:
            return shap_values[1]
        else:
            return shap_values
    else:
        return explainer.shap_values(x_df)[1]


def set_n_jobs(n_jobs: int, x_df: pd.DataFrame) -> int:
    """
    Sets the number of n_jobs, processes to run in parallel. If n_jobs is not specified, the max number of CPUs is
    used. If n_jobs is set to a higher amount than the number of observations in x_df, n_jobs is rebalanced to match
    the length of x_df.

    :param n_jobs: number of jobs to run in parallel
    :param x_df: x dataframe
    :return: number of jobs to run in parallel, using the above logic
    """
    if not n_jobs:
        n_jobs = mp.cpu_count()
    if n_jobs > len(x_df):
        n_jobs = len(x_df)
    return n_jobs


def run_shap_explainer_with_n_jobs(x_df: pd.DataFrame, explainer: callable, boosting_model: bool,
                                   regression_model: bool, linear_model: bool, n_jobs: int = None) -> np.array:
    """
    Splits x_df into evenly-split partitions based on the n_jobs parameter. If n_jobs is not specified, the max number
    of CPUs is used. If n_jobs is set to a higher amount than the number of observations in x_df, n_jobs is rebalanced
    to match the length of x_df. Then, the SHAP explainer object is run in parallel on each subset of x_df. The results
    are combined into a single object. If n_jobs is set to 1, then no parallel processing is used.

    :param x_df: x dataframe
    :param explainer: SHAP explainer object
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :param regression_model: Boolean of whether or not the explainer is for a regression model; if False, it is assumed
    we are dealing with a classification model
    :param linear_model: Boolean of whether or not the explainer is for a linear model
    :param n_jobs: number of cores to use when processing
    """
    if n_jobs != 1:
        n_jobs = set_n_jobs(n_jobs, x_df)
        array_split = np.array_split(x_df, n_jobs)
        shap_fn = partial(run_shap_explainer, explainer=explainer, boosting_model=boosting_model,
                          regression_model=regression_model, linear_model=linear_model)
        with mp.Pool(processes=n_jobs) as pool:
            result = pool.map(shap_fn, array_split)
        result = np.concatenate(result)
        return result
    else:
        return run_shap_explainer(x_df, explainer, boosting_model, regression_model, linear_model)


def make_shap_df(shap_values: np.array, x_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a numpy array of SHAP values into a dataframe, using the column names from x_df.

    :param shap_values: array of SHAP values
    :param x_df: x dataframe
    :return: dataframe of SHAP values
    """
    return pd.DataFrame(shap_values, columns=list(x_df))


def determine_if_name_in_object(name: str, py_object: object) -> bool:
    """
    Determine if a name is in a Python object.

    :param name: name to search for in py_object
    :param py_object: Python object
    """
    object_str = str(type(py_object)).lower()
    if name in object_str:
        return True
    else:
        return False


def determine_if_any_name_in_object(names_list: list, py_object: object) -> bool:
    """
    Determines if any name in names_list is also in the name of py_object, an arbitrary Python object.

    :param names_list: list of names to search fow
    :param py_object: an arbitrary Python object
    :return: boolean classification of if a match was found
    """
    match = False
    for name in names_list:
        result = determine_if_name_in_object(name, py_object)
        if result:
            match = True
            break
    return match


def determine_if_regression_model(ambiguous_regression_models: list, model: callable) -> bool:
    """
    Determines if model is a regression model.

    :param ambiguous_regression_models: list of non-obvious regression model names
    :param model: fitted model
    :return: Boolean of whether or not the model is a regression
    """
    regression = determine_if_any_name_in_object(['regress'] + ambiguous_regression_models, model)
    classification = determine_if_name_in_object('classifier', model)
    log_reg_check = determine_if_name_in_object('logistic', model)
    if regression and not log_reg_check and not classification:
        return True
    else:
        return False


def save_expected_value(expected_value: float, save_path: os.path):
    """
    Saves the SHAP expected value to a txt file.

    :param expected_value: expected value
    :param save_path: path in which to save the file
    """
    with open(os.path.join(save_path, 'shap_expected_value.txt'), 'w') as f:
        f.write(str(expected_value))
