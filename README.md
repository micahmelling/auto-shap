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

## Producing Plots

## Multiprocessing Support

## Overriding Auto-Detection

## Using the Repository
