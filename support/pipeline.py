# Data manipulation
import pandas as pd
import numpy as np

# sklearn model
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer, RobustScaler, StandardScaler
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, make_scorer, confusion_matrix

# sklearn build
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# vizualisation
from matplotlib import pyplot as plt
import seaborn as sns


NUMERIC_FEATURES = [
    'Customer_Age',
    'Months_on_book',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

CATEGORICAL_FEATURES =  [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

DISCRETE_FEATURES = [
    "Contacts_Count_12_mon",
    "Dependent_count",
    "Months_Inactive_12_mon",
    "Total_Relationship_Count"
]

FINAL_NUMERIC = [
    'Customer_Age',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
]

CAT_FEATURES1 =  [
    'Gender',
    'Marital_Status',
]

CAT_FEATURES2 =  [
    'Education_Level',
    'Income_Category',
    'Card_Category'
]

def categorical_pipeline():
    # Create the transformers for categorical features
    cat_transfo = ColumnTransformer([('categoricals', 'passthrough', CATEGORICAL_FEATURES)])
#     cat_transfo1 = ColumnTransformer([('categoricals', OneHotEncoder(), CAT_FEATURES1)], remainder='passthrough')
#     cat_transfo2 = ColumnTransformer([('categoricals', OrdinalEncoder(), CAT_FEATURES2)], remainder='passthrough')

#     threshold_n=0.95
#     thresholder = VarianceThreshold(threshold= threshold_n*(1 - threshold_n))

    # Create the pipeline to transform categorical features
    categorical_pipeline = Pipeline([
        # to keep only the categorical column we want
        ('cat_ct', cat_transfo),
        # Encoding as Ordinal
#         ('ordinal_enc', OrdinalEncoder()),
        # encoding as one-hot
        ('onehot', OneHotEncoder())
    ])

    return categorical_pipeline


def discrete_pipeline():
    # Create the transformers for categorical features
    disc_transfo = ColumnTransformer([('discretes', 'passthrough', DISCRETE_FEATURES)])

    # Create the pipeline to transform categorical features
    discrete_pipeline = Pipeline([
        # to keep only the categorical column we want
        ('dis_ct', disc_transfo),
        # encoding as ordinal
        ('ordinal_enc', OrdinalEncoder()),
        # scaling
        ('scaler', 'passthrough')
    ])

    return discrete_pipeline

def numeric_pipeline():
    # Create the transformers for numeric features
    num_transfo = ColumnTransformer([('numerics', 'passthrough', FINAL_NUMERIC)])

    # Create the pipeline to transform numeric features
    numerical_pipeline = Pipeline([
        # to keep only the numerical column we want
        ('num_ct', num_transfo),
        # removing variable with low variance
        ('variancethreshold', VarianceThreshold(threshold=0.0)),
        # scaler
        ('scaler', 'passthrough')
    ])

    return numerical_pipeline


def build_pipeline():
    # Create the categorical and numeric pipelines
    categorical_pipe = categorical_pipeline()
    numerical_pipe = numeric_pipeline()
    discrete_pipe = discrete_pipeline()
    # Create the feature union of categorical and numeric attributes
    ft_union = FeatureUnion([
        ('cat_pipeline', categorical_pipe),
        ('dis_pipeline', discrete_pipe),
        ('num_pipeline', numerical_pipe)
    ])

    pipeline = Pipeline([
        ('ft_union', ft_union),
        ('variancethreshold', VarianceThreshold(threshold=0.0))
    ])

    return pipeline


def get_scorer(score_func=f1_score):
    """
    return score function for GRID SEARCH
    """
    scorer = make_scorer(score_func)
    return scorer


def fit_pipeline(features_var, target_var, model_step_name, model_estimator, hyper_param_grid, pipeline_steps, scoring_function=None, nb_fold=5):
    
    model = model_estimator
    pipe = clone(pipeline_steps)
    pipe.steps.append([model_step_name, model_estimator])

    grid_search_pipe = GridSearchCV(pipe, hyper_param_grid,
                                    scoring=scoring_function,
                                    cv=nb_fold, n_jobs=-1)
    grid_search_pipe.fit(features_var, target_var)

    result = pd.DataFrame(grid_search_pipe.cv_results_)

    best_result = result.query('rank_test_score == 1')
    test_mean = best_result['mean_test_score'].values[0]
    test_std = best_result['std_test_score'].values[0]
    
    return (grid_search_pipe, model_step_name, test_mean, test_std)


def retrieve_pipeline_model(grid_model):
    model = grid_model[0].best_estimator_.steps[-1][1]
    pipeline = Pipeline(grid_model[0].best_estimator_.steps[:-1])
    full_pipeline = grid_model[0].best_estimator_

    return model, pipeline, full_pipeline


# Define subset of var to select in future Data : ColumnSelector(my_subset_of_var)
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, subset):
        super().__init__()
        self.subset = subset

    def transform(self, X, *_):
        return X.loc[:, self.subset]

    def fit(self, *_):
        return self

    
class ScalerSelector(BaseEstimator, TransformerMixin):

    def __init__(self, scaler=StandardScaler()):
        super().__init__()
        self.scaler = scaler

    def fit(self, X, y=None):
        return self.scaler.fit(X)

    def transform(self, X, y=None):
        return self.scaler.transform(X)
    
    
def binary_confusion_matrix_plot(label_value, predicted_proba, threshold=0.5, cmaps='Reds', ax=None):

        # tn, fp, fn, tp
        confusion_matrix_threshold = confusion_matrix(label_value, predicted_proba > threshold)

        metric_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
        metric_counts = ["{0:0.0f}".format(value) for value in confusion_matrix_threshold.flatten()]
        metric_percentages = ["{0:.2%}".format(value) for value in
                              confusion_matrix_threshold.flatten() / np.sum(confusion_matrix_threshold)]
        confusion_ = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                      zip(metric_names, metric_counts, metric_percentages)]

        confusion_values = np.asarray(confusion_).reshape(2, 2)

        # use the function below to plot a nice & pretty confusion matrix
        x_axis_labels = ["Pred. " + "Negative", "Pred. " + "Positive"]
        y_axis_labels = ["Negative", "Positive"]
        
        fig = sns.heatmap(confusion_matrix_threshold, annot=confusion_values,
                    xticklabels=x_axis_labels, yticklabels=y_axis_labels, fmt='', cmap='Reds', ax=ax)
        plt.xlabel("Predicted label", fontsize="medium")
        plt.ylabel("True label", fontsize="medium")
        plt.title("Confusion Matrix", fontsize="medium")
