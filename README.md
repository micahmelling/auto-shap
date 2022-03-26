# auto-shap
The auto-shap library is your best friend when calculating SHAP values!

[SHAP](https://christophm.github.io/interpretable-ml-book/shap.html) is a
state-of-the-art technique for explaining model predictions.
Model explanation can be valuable in many regards. For one, understanding
how a model devised a prediction can engender trust. Conversely, it could
inform us if our model is using features in a nonsensical way. Likewise,
feature importance scores can be useful for external presentations. For
further details on SHAP values and their underlying mathematical properties,
see the hyperlink at the beginning of this paragraph.

The Python [SHAP library](https://shap.readthedocs.io/en/latest/index.html)
is often the go-to source for computing SHAP values. It's handy and can
explain virtually any model we would like. However, we must be aware of the
following when using the library.

* The correct type of explainer class must be used. For example, if we
have a Random Forest model, we should use the TreeExplainer.
* Our code for implementing SHAP values will be slightly different when we
have a regression model instead of a classifier.
* SHAP cannot natively handel scikit-learn's CalibratedClassifierCV.
* Boosting models have distinct behavior when it comes to SHAP values.

Likewise, the native SHAP library does not take advantage of multiprocessing.
The auto-shap library will run SHAP calculations in parallel to speed them
up!

At a high level, the library will automatically detect the type of model
that has been trained (regressor vs. classifier, boosting model vs. other
model, etc) and apply the correct handling. If your model is not accurately
identified, it's easy to specify how it should be handled.

## Installation
The easiest way to install the library is with pip.

```buildoutcfg
$ pip3 install auto-shap
```
## Quick Example
Once installed, SHAP values can be calculated as follows.

```buildoutcfg
$ python3
>>> from auto_shap.auto_shap import generate_shap_values
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.ensemble import ExtraTreesClassifier
>>> x, y = load_breast_cancer(return_X_y=True, as_frame=True)
>>> model = ExtraTreesClassifier()
>>> model.fit(x, y)
>>> shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(model, x_df)
```

There you have it!
* A dataframe of SHAP values for every row in x.
* The expected values of the SHAP explainer (in the above, the average
predicted positive probability)
* A dataframe of the global SHAP values covering every feature in x.

What's more, you can change to a completely new model without changing any
of the auto-shap code.

```buildoutcfg
$ python3
>>> from auto_shap.auto_shap import generate_shap_values
>>> from sklearn.datasets import load_diabetes
>>> from sklearn.ensemble import GradientBoostingRegressor
>>> x, y = load_diabetes(return_X_y=True, as_frame=True)
>>> model = GradientBoostingRegressor()
>>> model.fit(x, y)
>>> shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(model, x_df)
```
auto-shap detected this was a boosted regressor and handled appropriately.

## Saving Output
The library also provides a helper function for saving output files and plots to a
local directory.

```buildoutcfg
$ python3
>>> from auto_shap.auto_shap import produce_shap_values_and_summary_plots
>>> from sklearn.datasets import load_diabetes
>>> from sklearn.ensemble import GradientBoostingRegressor
>>> x, y = load_diabetes(return_X_y=True, as_frame=True)
>>> model = GradientBoostingRegressor()
>>> model.fit(x, y)
>>> produce_shap_values_and_summary_plots(model=model, x_df=x_df, save_path='shap_output')
```
The above code will save three files into a files subdirectory in the specified
shap_output directory.
* A csv of SHAP values for every row in x_df.
* A txt file containing expected values of the SHAP explainer.
* A csv of the global SHAP values covering every feature in x_df.

Likewise, two plots will be saved into a plots subdirectory.
* A bar plot of the top global SHAP values.
* A dot plot of SHAP values to show the influence of features across observations
in x_df.

## Multiprocessing Support
By default, the maximum number of cores are used to calculate SHAP values in
parallel. To manually set the number of cores to use, you can do the following.

```buildoutcfg
>>> generate_shap_values(model, x_df, n_jobs=4)
```

## Overriding Auto-Detection
Using generate_shap_values or produce_shap_values_and_summary_plots will leverage
auto-detection of certain model characteristics. Those are as follows, which are
all controlled with Booleans:
* linear_model
* tree_model
* boosting_model
* calibrated_model
* regression_model

Though auto-shap will handle most common models, it is not yet tuned to handle
every possible type of model. In some cases, then, you can have to manually set
one or more of the above booleans in the function calls. At present and at minimum,
auto-shap will work with the following models.
* XGBClassifier
* XGBRegressor
* CatBoostClassifier
* CatBoostRegressor
* LGBMClassifier
* LGBMRegressor
* ExtraTreesClassifier
* ExtraTreesRegressor
* GradientBoostingClassifier
* GradientBoostingRegressor
* RandomForestClassifier
* RandomForestRegressor
* ElasticNet
* Lasso
* LinearRegression
* LogisticRegression
* Ridge
* DecisionTreeClassifier
* DecisionTreeRegressor

## CalibratedClassifierCV
The auto-shap library provides support for scikit-learn's CalibratedClassifierCV.
This implementation will extract the SHAP values for every base estimator in the
calibration ensemble. The SHAP values will then be averaged. For details on the
CalibratedClassifierCV, please go to the
[documentation](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html).
Since we are extracting only the SHAP values for the base estimator, we will miss
some detail since we are not using the full calibrator pair. Therefore, while
these SHAP values will still be instructive, they will not be perfectly precise.
For more precision, we would need to use the Kernel Explainer. The main benefit of
the approach in this function is computational as the Kernel Explainer can be
quite slow.

## Other Potentially Useful Functionality
The generate_shap_values function relies on a few underlying functions that can
be accessed directly and have the corresponding arguments and datatypes.

```buildoutcfg
get_shap_expected_value(explainer: callable, boosting_model: bool, linear_model) -> float

generate_shap_global_values(shap_values: np.array, x_df: pd.DataFrame) -> pd.DataFrame

produce_shap_output_with_kernel_explainer(model: callable, x_df: pd.DataFrame, boosting_model: bool,
                                          regression_model: bool, linear_model: bool,
                                          return_df: bool = True, n_jobs: int = None) -> tuple

produce_shap_output_with_tree_explainer(model: callable, x_df: pd.DataFrame, boosting_model: bool,
                                        regression_model: bool, linear_model: bool,
                                        return_df: bool = True, n_jobs: int = None) -> tuple

produce_shap_output_with_linear_explainer(model: callable, x_df: pd.DataFrame, regression_model: bool,
                                          inear_model: bool, return_df: bool = True, n_jobs: int = None) -> tuple

produce_shap_output_for_calibrated_classifier(model: callable, x_df: pd.DataFrame, boosting_model: bool,
                                              linear_model: bool, n_jobs: int = None) -> tuple

produce_raw_shap_values(model: callable, x_df: pd.DataFrame, use_kernel: bool, linear_model: bool, tree_model: bool,
                        calibrated_model: bool, boosting_model: bool, regression_model: bool,
                        n_jobs: int = None) -> tuple


generate_shap_summary_plot(shap_values: np.array, x_df: pd.DataFrame, plot_type: str, save_path: str,
                           file_prefix: str = None)
```

## The End
Enjoy explaining your models with auto-shap! Feel free to report any issues.
