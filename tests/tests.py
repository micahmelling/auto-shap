import os

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor, \
    ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from parallel_shap.parallel_shap import produce_shap_values_and_summary_plots, generate_shap_values


def train_simple_classification_model(model):
    x, y = load_breast_cancer(return_X_y=True, as_frame=True)
    model.fit(x, y)
    x = x.head(50)
    return model, x


def train_simple_regression_model(model):
    x, y = load_diabetes(return_X_y=True, as_frame=True)
    model.fit(x, y)
    x = x.head(50)
    return model, x


def test_classification_tree_model():
    model, x_df = train_simple_classification_model(RandomForestClassifier())
    shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(model, x_df)
    assert len(shap_values_df) == len(x_df)
    assert len(global_shap_df) == len(list(x_df))
    assert isinstance(shap_expected_value, float)


def test_classification_linear_model():
    model, x_df = train_simple_classification_model(LogisticRegression())
    shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(model, x_df)
    assert len(shap_values_df) == len(x_df)
    assert len(global_shap_df) == len(list(x_df))
    assert isinstance(shap_expected_value, float)


def test_classification_gradient_boosting_model():
    model, x_df = train_simple_classification_model(GradientBoostingClassifier())
    shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(model, x_df)
    assert len(shap_values_df) == len(x_df)
    assert len(global_shap_df) == len(list(x_df))
    assert isinstance(shap_expected_value, float)


def test_classification_xgboost_model():
    model, x_df = train_simple_classification_model(XGBClassifier(n_jobs=1))
    shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(model, x_df)
    assert len(shap_values_df) == len(x_df)
    assert len(global_shap_df) == len(list(x_df))
    assert isinstance(shap_expected_value, float)


def test_classification_lightgbm_model():
    model, x_df = train_simple_classification_model(LGBMClassifier())
    shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(model, x_df)
    assert len(shap_values_df) == len(x_df)
    assert len(global_shap_df) == len(list(x_df))
    assert isinstance(shap_expected_value, float)


def test_classification_catboost_model():
    model, x_df = train_simple_classification_model(CatBoostClassifier())
    shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(model, x_df)
    assert len(shap_values_df) == len(x_df)
    assert len(global_shap_df) == len(list(x_df))
    assert isinstance(shap_expected_value, float)


def test_regression_linear_model():
    model, x_df = train_simple_regression_model(Ridge())
    shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(model, x_df)
    assert len(shap_values_df) == len(x_df)
    assert len(global_shap_df) == len(list(x_df))
    assert isinstance(shap_expected_value, float)


def test_regression_tree_model():
    model, x_df = train_simple_regression_model(ExtraTreesRegressor())
    shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(model, x_df)
    assert len(shap_values_df) == len(x_df)
    assert len(global_shap_df) == len(list(x_df))
    assert isinstance(shap_expected_value, float)


def test_regression_gradient_boosting_model():
    model, x_df = train_simple_regression_model(GradientBoostingRegressor())
    shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(model, x_df)
    assert len(shap_values_df) == len(x_df)
    assert len(global_shap_df) == len(list(x_df))
    assert isinstance(shap_expected_value, float)


def test_kernel_shap_regression():
    model, x_df = train_simple_regression_model(GradientBoostingRegressor())
    shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(model, x_df, use_kernel=True)
    assert len(shap_values_df) == len(x_df)
    assert len(global_shap_df) == len(list(x_df))
    assert isinstance(shap_expected_value, float)


def test_kernel_shap_classification():
    model, x_df = train_simple_classification_model(ExtraTreesClassifier())
    x_df = x_df.head(10)
    shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(model, x_df, use_kernel=True)
    assert len(shap_values_df) == len(x_df)
    assert len(global_shap_df) == len(list(x_df))
    assert isinstance(shap_expected_value, float)


def test_calibrated_extra_trees_model():
    model, x_df = train_simple_classification_model(CalibratedClassifierCV(base_estimator=ExtraTreesClassifier(),
                                                                           cv=3))
    shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(model, x_df)
    assert len(shap_values_df) == len(x_df)
    assert len(global_shap_df) == len(list(x_df))
    assert isinstance(shap_expected_value, float)


def test_calibrated_extra_trees_model_with_kernel():
    model, x_df = train_simple_classification_model(CalibratedClassifierCV(base_estimator=ExtraTreesClassifier(),
                                                                           cv=3))
    shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(model, x_df, use_kernel=True)
    assert len(shap_values_df) == len(x_df)
    assert len(global_shap_df) == len(list(x_df))
    assert isinstance(shap_expected_value, float)


def test_calibrated_xgboost_model():
    model, x_df = train_simple_classification_model(CalibratedClassifierCV(base_estimator=XGBClassifier(n_jobs=1)))
    shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(model, x_df, use_kernel=True)
    assert len(shap_values_df) == len(x_df)
    assert len(global_shap_df) == len(list(x_df))
    assert isinstance(shap_expected_value, float)


def test_produce_shap_values_and_summary_plots_extra_trees_regressor():
    model, x_df = train_simple_regression_model(ExtraTreesRegressor())
    produce_shap_values_and_summary_plots(model, x_df, 'etr_output')
    assert os.path.exists('etr_output/plots/shap_values_dot.png')


def test_produce_shap_values_and_summary_plots_random_forest_classifier():
    model, x_df = train_simple_classification_model(RandomForestClassifier())
    produce_shap_values_and_summary_plots(model, x_df, 'rfc_output')
    assert os.path.exists('rfc_output/plots/shap_values_dot.png')


def test_produce_shap_values_and_summary_plots_calibrated_classifier():
    model, x_df = train_simple_classification_model(CalibratedClassifierCV(base_estimator=ExtraTreesClassifier(),
                                                                           cv=3))
    produce_shap_values_and_summary_plots(model, x_df, 'cc_output')
    assert os.path.exists('cc_output/plots/shap_values_dot.png')


def test_produce_shap_values_and_summary_plots_calibrated_classifier_with_kernel():
    model, x_df = train_simple_classification_model(CalibratedClassifierCV(base_estimator=ExtraTreesClassifier(),
                                                                           cv=3))
    x_df = x_df.head(10)
    produce_shap_values_and_summary_plots(model, x_df, 'kcc_output', use_kernel=True)
    assert os.path.exists('kcc_output/plots/shap_values_dot.png')


def test_produce_shap_values_and_summary_plots_xgboost_classifier():
    model, x_df = train_simple_classification_model(XGBClassifier(n_jobs=1))
    produce_shap_values_and_summary_plots(model, x_df, 'xgbc_output')
    assert os.path.exists('xgbc_output/plots/shap_values_dot.png')


# TODO: train models only once
# TODO: add teardown function
