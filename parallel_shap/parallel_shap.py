import shap
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import multiprocessing as mp

from functools import partial
from statistics import mean


# move to helpers
def make_directories_if_not_exists(directories_list: list):
    """
    Makes directories in the current working directory if they do not exist.

    :param directories_list: list of directories to create
    """
    for directory in directories_list:
        if not os.path.exists(directory):
            os.makedirs(directory)


# move to helpers
def run_shap_explainer(x_df: pd.DataFrame, explainer: callable, boosting_model: bool,
                       regression_model: bool) -> np.array:
    """
    Runs the SHAP explainer on a dataframe of predictors (i.e. x_df).

    :param x_df: x dataframe
    :param explainer: SHAP explainer object
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :param regression_model: Boolean of whether or not the explainer is for a regression model; if False, it is assumed
    we are dealing with a classification model
    """
    if boosting_model or regression_model:
        return explainer.shap_values(x_df)
    else:
        return explainer.shap_values(x_df)[1]


# move to helpers
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


# move to helpers
def run_parallel_shap_explainer(x_df: pd.DataFrame, explainer: callable, boosting_model: bool, regression_model: bool,
                                n_jobs: int = None) -> np.array:
    """
    Splits x_df into evenly-split partitions based on the n_jobs parameter. If n_jobs is not specified, the max number
    of CPUs is used. If n_jobs is set to a higher amount than the number of observations in x_df, n_jobs is rebalanced
    to match the length of x_df.Then, the SHAP explainer object is run in parallel on each subset of x_df. The results
    are combined into a single object.

    :param x_df: x dataframe
    :param explainer: SHAP explainer object
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :param regression_model: Boolean of whether or not the explainer is for a regression model; if False, it is assumed
    we are dealing with a classification model
    :param n_jobs: number of cores to use when processing
    """
    n_jobs = set_n_jobs(n_jobs, x_df)
    array_split = np.array_split(x_df, n_jobs)
    shap_fn = partial(run_shap_explainer, explainer=explainer, boosting_model=boosting_model,
                      regression_model=regression_model)
    with mp.Pool(processes=n_jobs) as pool:
        result = pool.map(shap_fn, array_split)
    result = np.concatenate(result)
    return result


def get_shap_expected_value(explainer: callable, boosting_model: bool) -> float:
    """
    Extracts a SHAP Explainer's expected value. For a classifier, this is the mean positive predicted probability.
    For a regression model, this is the mean predicted value.

    :param explainer: SHAP explainer object
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :returns: SHAP Explainer's expected value
    """
    if boosting_model:
        expected_value = explainer.expected_value[0]
    else:
        try:
            expected_value = explainer.expected_value[1]
        except IndexError:
            expected_value = explainer.expected_value[0]
    return expected_value


def generate_shap_global_values(shap_values: np.array, x_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the global SHAP values for every feature.

    :param shap_values: numpy array of shap values
    :param x_df: x_df dataframe
    :returns: global SHAP values for every feature
    """
    shap_values = np.abs(shap_values).mean(0)
    df = pd.DataFrame(list(zip(x_df.columns, shap_values)), columns=['feature', 'shap_value'])
    df.sort_values(by=['shap_value'], ascending=False, inplace=True)
    return df


# move to helpers
def make_shap_df(shap_values, x_df):
    """

    :param shap_values:
    :param x_df:
    :return:
    """
    return pd.DataFrame(shap_values, columns=list(x_df))


def produce_shap_output_with_kernel_explainer(model, x_df, boosting_model, regression_model, return_df=True):
    explainer = shap.KernelExplainer(model.predict_proba, x_df)
    shap_values = run_parallel_shap_explainer(x_df, explainer, boosting_model, regression_model)
    shap_expected_value = get_shap_expected_value(explainer, boosting_model)
    global_shap_df = generate_shap_global_values(shap_values, x_df)
    if return_df:
        shap_values_df = make_shap_df(shap_values, x_df)
        return shap_values_df, shap_expected_value, global_shap_df
    else:
        return shap_values, shap_expected_value, global_shap_df


def produce_shap_output_with_tree_explainer(model, x_df, boosting_model, regression_model, return_df=True):
    explainer = shap.TreeExplainer(model)
    shap_values = run_parallel_shap_explainer(x_df, explainer, boosting_model, regression_model)
    shap_expected_value = get_shap_expected_value(explainer, boosting_model)
    global_shap_df = generate_shap_global_values(shap_values, x_df)
    if return_df:
        shap_values_df = make_shap_df(shap_values, x_df)
        return shap_values_df, shap_expected_value, global_shap_df
    else:
        return shap_values, shap_expected_value, global_shap_df


def produce_shap_output_with_linear_explainer(model, x_df, regression_model, return_df=True):
    explainer = shap.LinearExplainer(model)
    shap_values = shap_values = run_parallel_shap_explainer(x_df, explainer, False, regression_model)
    shap_expected_value = get_shap_expected_value(explainer, False)
    global_shap_df = generate_shap_global_values(shap_values, x_df)
    if return_df:
        shap_values_df = make_shap_df(shap_values, x_df)
        return shap_values_df, shap_expected_value, global_shap_df
    else:
        return shap_values, shap_expected_value, global_shap_df


def produce_shap_output_for_calibrated_classifier(model, x_df, boosting_model, linear_model):
    shap_values_list = []
    shap_expected_list = []
    for calibrated_classifier in model.calibrated_classifiers_:
        if linear_model:
            shap_values, shap_expected_value = produce_shap_output_with_linear_explainer(
                calibrated_classifier.base_estimator, x_df, regression_model=False, return_df=False
            )
        else:
            shap_values, shap_expected_value = produce_shap_output_with_tree_explainer(
                model, x_df, boosting_model, regression_model=False, return_df=False
            )
        shap_values_list.append(shap_values)
        shap_expected_list.append(shap_expected_value)
    shap_values = np.array(shap_values_list).sum(axis=0) / len(shap_values_list)
    shap_expected_value = mean(shap_expected_list)
    global_shap_df = generate_shap_global_values(shap_values, x_df)
    shap_values_df = make_shap_df(shap_values, x_df)
    return shap_values_df, shap_expected_value, global_shap_df


def produce_raw_shap_values(model, x_df, use_kernel, linear_model, tree_model, calibrated_model, boosting_model,
                            regression_model):
    """
    Produces the raw shap values for every observation in the test set. A dataframe of the shap values is saved locally
    as a csv. The shap expected value is extracted and save locally in a csv.

    :param model: fitted model
    :param x_df: x dataframe
    :param calibrated_model: boolean of whether or not the model is a CalibratedClassifierCV; the default is False
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :param use_kernel: Boolean of whether or not to use Kernel SHAP, which mostly makes sense when we are using a
    CalibratedClassifierCV
    :param tree_model:
    :param linear_model:
    :returns: pandas dataframe
    """
    if use_kernel:
        return produce_shap_output_with_kernel_explainer(model, x_df, boosting_model, regression_model)
    if linear_model:
        return produce_shap_output_with_linear_explainer(model, x_df, regression_model)
    if tree_model:
        return produce_shap_output_with_tree_explainer(model, x_df, boosting_model, regression_model)
    if calibrated_model:
        return produce_shap_output_for_calibrated_classifier(model, x_df, boosting_model, linear_model)


def generate_shap_summary_plot(shap_values, x_df, plot_type, save_path, file_prefix=None):
    """
    Generates a plot of shap values and saves it locally.

    :param shap_values: numpy array of shap values produced for x_df
    :param x_df: x dataframe
    :param file_prefix
    :param plot_type: the type of plot we want to generate; generally, either dot or bar
    """
    shap.summary_plot(shap_values, x_df, plot_type=plot_type, show=False)
    if not file_prefix:
        file_name = f'shap_values_{plot_type}.png'
    else:
        file_name = f'{file_prefix}_shap_values_{plot_type}.png'
    plt.savefig(os.path.join(save_path, file_name), bbox_inches='tight')
    plt.clf()


# move to helpers
def determine_if_name_in_object(name, py_object):
    """
    Determine if a name is in a Python object.

    :param py_object: Python object
    :param name: name to search for in py_object
    """
    object_str = str((type(py_object))).lower()
    if name in object_str:
        return True
    else:
        return False


# move to helpers
def determine_if_any_name_in_object(names_list, py_object):
    """

    :param names_list:
    :param py_object:
    :return:
    """
    for name in names_list:
        result = determine_if_name_in_object(name, py_object)
        if result:
            return result


# move to helpers
def determine_if_regression_model(model):
    regression = determine_if_name_in_object('regression', model)
    log_reg_check = determine_if_name_in_object('logistic', model)
    if regression and not log_reg_check:
        return True
    else:
        return False


def generate_shap_values(model, x_df, linear_model=None, tree_model=None, boosting_model=None,
                         calibrated_model=None, regression_model=False, use_kernel=False):
    if not linear_model:
        linear_model = determine_if_name_in_object(['logisticregression', 'linearregression', 'elasticnet', 'ridge',
                                                    'lasso'], model)
    if not tree_model:
        tree_model = determine_if_name_in_object(['randomforest', 'extratrees', 'decisiontree', 'boost'], model)
    if not boosting_model:
        boosting_model = determine_if_name_in_object('boost', model)
    if not calibrated_model:
        calibrated_model = determine_if_name_in_object('calibrated', model)
    if not regression_model:
        regression_model = determine_if_regression_model(model)
    shap_values_df, shap_expected_value, global_shap_df = produce_raw_shap_values(
        model, x_df, use_kernel, linear_model, tree_model, calibrated_model, boosting_model, regression_model
    )
    return shap_values_df, shap_expected_value, global_shap_df


def produce_shap_values_and_summary_plots(model, x_df, model_uid, save_path, linear_model, tree_model, boosting_model,
                                          calibrated_model, regression_model, use_kernel):
    """
    Produces SHAP values for x_df and writes associated diagnostics locally.

    :param model: model with predict method
    :param x_df: x dataframe
    :param model_uid: model uid
    :param use_kernel: Boolean of whether or not to use Kernel SHAP, which mostly makes sense when we are using a
    CalibratedClassifierCV
    :param save_path: path in which to save output; subdirectories titled 'files' and 'plots' will also be created,
    and appropriate output will be saved in each
    :param boosting_model:
    :param calibrated_model:
    :param linear_model:
    """
    make_directories_if_not_exists(
        [
            save_path,
            os.path.join(save_path, 'files'),
            os.path.join(save_path, 'plots'),
        ]
    )
    local_shap_df = generate_shap_values(model, x_df, linear_model, tree_model, boosting_model, calibrated_model,
                                         regression_model, use_kernel)
    generate_shap_global_values(local_shap_df, x_df)
    generate_shap_summary_plot(local_shap_df, x_df, model_uid, 'dot')
    generate_shap_summary_plot(local_shap_df, x_df, model_uid, 'bar')


# TODO: use type hints
# TODO: docstrings
# TODO: helpers file
# TODO: add config file
# TODO: error handling where needed
# TODO: use kwargs when appropriate
# TODO: build out repo
# TODO: tests
# TODO: documentation (readme)
