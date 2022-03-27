import os
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from auto_shap.utilities import (determine_model_qualities,
                                 make_directories_if_not_exists, make_shap_df,
                                 run_shap_explainer_with_n_jobs,
                                 save_expected_value)


def get_shap_expected_value(explainer: callable, boosting_model: bool, linear_model) -> float:
    """
    Extracts a SHAP Explainer's expected value. For a classifier, this is the mean positive predicted probability.
    For a regression model, this is the mean predicted value.

    :param explainer: SHAP explainer object
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :param linear_model: Boolean of whether or not the explainer is for a linear model
    :returns: SHAP Explainer's expected value
    """
    if linear_model:
        expected_value = explainer.expected_value
    elif boosting_model:
        try:
            expected_value = explainer.expected_value[0]
        except TypeError:
            expected_value = explainer.expected_value
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


def produce_shap_output_with_kernel_explainer(model: callable, x_df: pd.DataFrame, boosting_model: bool,
                                              regression_model: bool, linear_model: bool,
                                              return_df: bool = True, n_jobs: int = None) -> tuple:
    """
    Using the Kernel Explainer, produces SHAP values and associated output: SHAP values for every row in x_df, the
    expected value from the SHAP explainer, and a dataframe of global SHAP values. Runs the SHAP explainer in parallel
    to increase speed.

    :param model: fitted model
    :param x_df: x dataframe
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :param regression_model: Boolean of whether or not this is a regression model; if not, it's assumed this is a
    classification model
    :param return_df: Boolean of whether to return a dataframe; if False, an array is returned
    :param linear_model: Boolean of whether or not the explainer is for a linear model
    :param n_jobs: number of cores to use when processing
    :return: tuple with three components: SHAP values (dataframe or array), the expected value from the SHAP explainer,
    and a dataframe of global SHAP values
    """
    if regression_model:
        explainer = shap.KernelExplainer(model.predict, x_df)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, x_df)
    shap_values = run_shap_explainer_with_n_jobs(x_df, explainer, boosting_model, regression_model, linear_model, n_jobs)
    shap_expected_value = get_shap_expected_value(explainer, boosting_model, linear_model)
    global_shap_df = generate_shap_global_values(shap_values, x_df)
    if return_df:
        shap_values_df = make_shap_df(shap_values, x_df)
        return shap_values_df, shap_expected_value, global_shap_df
    else:
        return shap_values, shap_expected_value, global_shap_df


def produce_shap_output_with_tree_explainer(model: callable, x_df: pd.DataFrame, boosting_model: bool,
                                            regression_model: bool, linear_model: bool,
                                            return_df: bool = True, n_jobs: int = None) -> tuple:
    """
    Using the Tree Explainer, produces SHAP values and associated output: SHAP values for every row in x_df, the
    expected value from the SHAP explainer, and a dataframe of global SHAP values. Runs the SHAP explainer in parallel
    to increase speed.

    :param model: fitted model
    :param x_df: x dataframe
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :param regression_model: Boolean of whether or not this is a regression model; if not, it's assumed this is a
    classification model
    :param linear_model: Boolean of whether or not the explainer is for a linear model
    :param return_df: Boolean of whether to return a dataframe; if False, an array is returned
    :param n_jobs: number of cores to use when processing
    :return: tuple with three components: SHAP values (dataframe or array), the expected value from the SHAP explainer,
    and a dataframe of global SHAP values
    """
    explainer = shap.TreeExplainer(model)
    shap_values = run_shap_explainer_with_n_jobs(x_df, explainer, boosting_model, regression_model, False, n_jobs)
    shap_expected_value = get_shap_expected_value(explainer, boosting_model, linear_model)
    global_shap_df = generate_shap_global_values(shap_values, x_df)
    if return_df:
        shap_values_df = make_shap_df(shap_values, x_df)
        return shap_values_df, shap_expected_value, global_shap_df
    else:
        return shap_values, shap_expected_value, global_shap_df


def produce_shap_output_with_linear_explainer(model: callable, x_df: pd.DataFrame, regression_model: bool,
                                              linear_model: bool, return_df: bool = True, n_jobs: int = None) -> tuple:
    """
    Using the Linear Explainer, produces SHAP values and associated output: SHAP values for every row in x_df, the
    expected value from the SHAP explainer, and a dataframe of global SHAP values. Runs the SHAP explainer in parallel
    to increase speed.

    :param model: fitted model
    :param x_df: x dataframe
    :param regression_model: Boolean of whether or not this is a regression model; if not, it's assumed this is a
    classification model
    :param linear_model: Boolean of whether or not the explainer is for a linear model
    :param return_df: Boolean of whether to return a dataframe; if False, an array is returned
    :param n_jobs: number of cores to use when processing
    :return: tuple with three components: SHAP values (dataframe or array), the expected value from the SHAP explainer,
    and a dataframe of global SHAP values
    """
    explainer = shap.LinearExplainer(model, x_df)
    shap_values = run_shap_explainer_with_n_jobs(x_df, explainer, False, regression_model, True, n_jobs)
    shap_expected_value = get_shap_expected_value(explainer, False, linear_model)
    global_shap_df = generate_shap_global_values(shap_values, x_df)
    if return_df:
        shap_values_df = make_shap_df(shap_values, x_df)
        return shap_values_df, shap_expected_value, global_shap_df
    else:
        return shap_values, shap_expected_value, global_shap_df


def produce_shap_output_for_calibrated_classifier(model: callable, x_df: pd.DataFrame, boosting_model: bool,
                                                  linear_model: bool, n_jobs: int = None) -> tuple:
    """
    Produces SHAP values for a CalibratedClassifierCV. This process will extract the SHAP values for every base
    estimator in the calibration ensemble. The SHAP values will then be averaged. For details on the
    CalibratedClassifierCV, please go to the following link
    https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html. Since we are
    extracting only the SHAP values for the base estimator, we will miss some detail since we are not using the full
    calibrator pair. Therefore, while these SHAP values will still be instructive, they will not be perfectly precise.
    For more precision, we would need to use the Kernel Explainer. The main benefit of the approach in this function is
    computational as the Kernel Explainer can be quite slow.

    The following output is generated: SHAP values for every row in x_df, the expected value from the SHAP explainer,
    and a dataframe of global SHAP values. Runs the SHAP explainer in parallel to increase speed.

    :param model: fitted model
    :param x_df: x dataframe
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :param linear_model: Boolean of whether or not this is a linear model; if not, it is assumed this is a tree-based
    model
    :param n_jobs: number of cores to use when processing
    :return: tuple with three components: SHAP values (dataframe or array), the expected value from the SHAP explainer,
    and a dataframe of global SHAP values
    """
    shap_values_list = []
    shap_expected_list = []
    for calibrated_classifier in model.calibrated_classifiers_:
        if linear_model:
            shap_values, shap_expected_value, _ = produce_shap_output_with_linear_explainer(
                calibrated_classifier.base_estimator, x_df, regression_model=False, linear_model=True,
                return_df=False, n_jobs=n_jobs
            )
        else:
            shap_values, shap_expected_value, _ = produce_shap_output_with_tree_explainer(
                calibrated_classifier.base_estimator, x_df, boosting_model, regression_model=False,
                linear_model=False, return_df=False, n_jobs=n_jobs
            )
        shap_values_list.append(shap_values)
        shap_expected_list.append(shap_expected_value)
    shap_values = np.array(shap_values_list).sum(axis=0) / len(shap_values_list)
    shap_expected_value = mean(shap_expected_list)
    global_shap_df = generate_shap_global_values(shap_values, x_df)
    shap_values_df = make_shap_df(shap_values, x_df)
    return shap_values_df, shap_expected_value, global_shap_df


def produce_raw_shap_values(model: callable, x_df: pd.DataFrame, use_kernel: bool, linear_model: bool, tree_model: bool,
                            calibrated_model: bool, boosting_model: bool, regression_model: bool,
                            n_jobs: int = None) -> tuple:
    """
    Produces SHAP output for every observation in x_df: SHAP values for every row in x_df, the expected value from the
    SHAP explainer, and a dataframe of global SHAP values. Runs the SHAP explainer in parallel to increase speed.
    The appropriate explainer is used based on the boolean function arguments.

    :param model: fitted model
    :param x_df: x dataframe
    :param use_kernel: Boolean of whether or not to use Kernel SHAP
    :param linear_model: Boolean of whether model is a linear model, which would employ Linear SHAP
    :param tree_model: Boolean of whether model is a tree-based model, which would employ Tree SHAP
    :param calibrated_model: Boolean of whether or not the model is a CalibratedClassifierCV
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :param regression_model: Boolean of whether or not this is a regression model; if not, it's assumed this is a
    classification model
    :param n_jobs: number of cores to use when processing
    :return: tuple with three components: dataframe of SHAP values for every row in x_df, the expected value from the
    SHAP explainer, and a dataframe of global SHAP values
    """
    if use_kernel:
        return produce_shap_output_with_kernel_explainer(model, x_df, boosting_model, regression_model, linear_model,
                                                         n_jobs=n_jobs)
    else:
        if calibrated_model:
            return produce_shap_output_for_calibrated_classifier(model, x_df, boosting_model, linear_model,
                                                                 n_jobs=n_jobs)
        else:
            if tree_model:
                return produce_shap_output_with_tree_explainer(model, x_df, boosting_model, regression_model, False,
                                                       n_jobs=n_jobs)
            elif linear_model:
                return produce_shap_output_with_linear_explainer(model, x_df, regression_model, True, n_jobs=n_jobs)


def generate_shap_summary_plot(shap_values: np.array, x_df: pd.DataFrame, plot_type: str, save_path: str,
                               file_prefix: str = None):
    """
    Generates a summary plot of SHAP values and saves it locally.

    :param shap_values: numpy array of shap values produced for x_df
    :param x_df: x dataframe
    :param plot_type: the type of plot we want to generate; generally, either 'dot' or 'bar'
    :param save_path: path in which to save the plot
    :param file_prefix: prefix to add to the file name
    """
    shap.summary_plot(shap_values.values, x_df, plot_type=plot_type, show=False)
    if not file_prefix:
        file_name = f'shap_values_{plot_type}.png'
    else:
        file_name = f'{file_prefix}_shap_values_{plot_type}.png'
    plt.savefig(os.path.join(save_path, 'plots', file_name), bbox_inches='tight')
    plt.clf()


def generate_shap_values(model: callable, x_df: pd.DataFrame, linear_model: bool = None, tree_model: bool = None,
                         boosting_model: bool = None, calibrated_model: bool = None, regression_model: bool = False,
                         use_kernel: bool = False, n_jobs: int = None) -> tuple:
    """
    Generates SHAP values for the provided model and x_df. Three pieces of output are generated: dataframe of SHAP
    values for every row in x_df, the expected value from the SHAP explainer, and a dataframe of global SHAP values.
    The function will attempt to use the best explainer for the model (e.g. linear vs. tree). It will also attempt
    to automatically detect if the model is a boosting model, which can require some special handling. Likewise, it
    will work to automatically detect if the model is for classification or regression, which again requires some
    subtle handling between the two. Additionally, if the model is a CalibratedClassifierCV, it will detect and
    handle appropriately. Lastly, the user can specify any of these booleans if needed or useful. Related, the user
    can specify if they want to use Kernel SHAP, which is appropriate in certain cases but is computationally expensive.
    If using Kernel SHAP, it's recommended to use a sample of x_df.

    :param model: fitted model
    :param x_df: x dataframe
    :param linear_model: Boolean of whether model is a linear model, which would employ Linear SHAP
    :param tree_model: Boolean of whether model is a tree-based model, which would employ Tree SHAP
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :param calibrated_model: Boolean of whether or not the model is a CalibratedClassifierCV
    :param regression_model: Boolean of whether or not this is a regression model; if not, it's assumed this is a
    classification model
    :param use_kernel: Boolean of whether or not to use Kernel SHAP
    :param n_jobs: number of cores to use when processing
    :return: tuple with three components: dataframe of SHAP values for every row in x_df, the expected value from the
    SHAP explainer, and a dataframe of global SHAP values
    """
    calibrated_model, linear_model, tree_model, boosting_model, regression_model = determine_model_qualities(
        model, linear_model, tree_model, boosting_model, calibrated_model, regression_model
    )
    shap_values_df, shap_expected_value, global_shap_df = produce_raw_shap_values(
        model, x_df, use_kernel, linear_model, tree_model, calibrated_model, boosting_model, regression_model, n_jobs
    )
    return shap_values_df, shap_expected_value, global_shap_df


def produce_shap_values_and_summary_plots(model: callable, x_df: pd.DataFrame, save_path: str,
                                          linear_model: bool = None,  tree_model: bool = None,
                                          boosting_model: bool = None, calibrated_model: bool = None,
                                          regression_model: bool = None, use_kernel: bool = None,
                                          file_prefix: str = None, n_jobs: int = None):
    """
    Produces SHAP values for x_df and writes associated diagnostics locally. The following output is saved in
    save_path in appropriate subdirectories: SHAP values for every row in x_df, global SHAP values for every feature
    in x_df, the expected value of the SHAP explainer, bar plot of global SHAP features, and dot plot of SHAP values for
    top features.

    The function will attempt to use the best explainer for the model (e.g. linear vs. tree). It will also attempt
    to automatically detect if the model is a boosting model, which can require some special handling. Likewise, it
    will work to automatically detect if the model is for classification or regression, which again requires some
    subtle handling between the two. Additionally, if the model is a CalibratedClassifierCV, it will detect and
    handle appropriately. Lastly, the user can specify any of these booleans if needed or useful. Related, the user
    can specify if they want to use Kernel SHAP, which is appropriate in certain cases but is computationally expensive.
    If using Kernel SHAP, it's recommended to use a sample of x_df.

    :param model: fitted model
    :param x_df: x dataframe
    :param save_path: path in which to save the plot
    :param linear_model: Boolean of whether model is a linear model, which would employ Linear SHAP
    :param tree_model: Boolean of whether model is a tree-based model, which would employ Tree SHAP
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :param calibrated_model: Boolean of whether or not the model is a CalibratedClassifierCV
    :param regression_model: Boolean of whether or not this is a regression model; if not, it's assumed this is a
    classification model
    :param use_kernel: Boolean of whether or not to use Kernel SHAP
    :return: tuple with three components: dataframe of SHAP values for every row in x_df, the expected value from the
    SHAP explainer, and a dataframe of global SHAP values
    :param file_prefix: prefix to add to the file name
    :param n_jobs: number of cores to use when processing
    """
    save_file_path = os.path.join(save_path, 'files')
    save_plots_path = os.path.join(save_path, 'plots')
    make_directories_if_not_exists(
        [
            save_path,
            save_file_path,
            save_plots_path,
        ]
    )
    shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(
        model, x_df, linear_model, tree_model, boosting_model, calibrated_model, regression_model, use_kernel, n_jobs
    )
    shap_values_df.to_csv(os.path.join(save_file_path, 'local_shap_values.csv'), index=False)
    global_shap_df.to_csv(os.path.join(save_file_path, 'global_shap_values.csv'), index=False)
    save_expected_value(shap_expected_value, save_file_path)
    generate_shap_summary_plot(shap_values_df, x_df, 'bar', save_path, file_prefix)
    generate_shap_summary_plot(shap_values_df, x_df, 'dot', save_path, file_prefix)
