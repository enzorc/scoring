# Data manipulation
import pandas as pd
import numpy as np

# sklearn
from sklearn.metrics import confusion_matrix

# vizualisation
from matplotlib import pyplot as plt
import seaborn as sns

# other
import time


def load_data(path, *args, target_var="Attrition_Flag"):
    import pandas as pd
    data = pd.read_csv(path)
    data[target_var] = data[target_var].map({args[0]: True, args[1]: False})
    X, y = data.drop([target_var], axis=1), data[target_var]
    return X, y


def display_params(pipe_or_model_=None):
    from pprint import pprint
    if pipe_or_model_ is not None:
        pprint(sorted(pipe_or_model_.get_params().keys()))
        print("")
        liner = str(pipe_or_model_.__doc__).split('Parameters\n    ----------\n')[1].split('\n\n    Attributes\n')[0].replace('\n        ', '\n').splitlines()
        liner = [i.strip() for i in liner if " : " in i] # <<< the key is to use " : " as our anchor
        pprint(liner)


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
        
#         return confusion_matrix_threshold, confusion_values


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed


def plot_categorical(col_cat_list, data, fig_dims = (12, 7)):
    nb_obs = len(data)
    for col_name in col_cat_list:
        fig, ax = plt.subplots(figsize=fig_dims)
        sns.set(font_scale=1.5)
        sns.countplot(x=col_name, ax=ax, data=data, order=data[col_name].value_counts().index)
        for p in ax.patches:
            ax.annotate('{} \n[%{:.1f}]'.format(p.get_height(), p.get_height()/nb_obs*100), (p.get_x()+0.1, p.get_height()+50))

            
def plot_categorical_distrib(col_cat_list, data, fig_dims = (12, 7)):
    for col_name in col_cat_list:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_dims)
        sns.set(font_scale=1.5)

        grouped_data = data.groupby(col_name)["Attrition_Flag"].value_counts().rename('count').reset_index()
        pivot_data = grouped_data.pivot(columns='Attrition_Flag', values='count', index=col_name)
        pivot_data = pivot_data.reindex(data[col_name].value_counts().index)

        pivot_data.div(pivot_data.sum(1), axis=0).plot.bar(stacked=True, ax=ax1)
        sns.countplot(x=col_name, ax=ax2, data=data, order=data[col_name].value_counts().index, color = 'C0')
        plt.xticks(rotation=90)
        plt.title(col_name)
        

def attrition_rate_by_cat(col_name, data):
    
    card_owner_churn = data.groupby(col_name)
    card_owner_churn = card_owner_churn["Attrition_Flag"].value_counts(normalize=True)
    card_owner_churn = card_owner_churn.rename('rate').reset_index()
    card_owner_churn = card_owner_churn.query("Attrition_Flag == 'Attrited Customer'")
    
    return card_owner_churn.style.bar()


def plot_feature_importance(importance,names,model_type):

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    

def correlation_matrix(data, metric="spearman", target_var=None, figsize=(11,11)):

    corr = data.corr('spearman')
    
    if target_var:
        assert type(target_var) == str, "'target_var' should be string"
        assert target_var in corr.columns, "'target_var' is not an attribute of the correlation data, check if it's numeric"
        corr = corr.sort_values(target_var, axis=0, ascending=False, key=abs)
        corr = corr.sort_values(target_var, axis=1, ascending=False, key=abs)

    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    plt.figure(figsize=figsize)
    sns.set(font_scale=1)
    sns.heatmap(corr, annot=True, center=0, cmap=cmap, mask=mask, square=True, linewidths=.5, cbar_kws={"shrink": .5}, fmt=".2g")
    plt.title('Correlation Matrix')




# def drop_col_high_freq(serie, thresh=.8):
#     if (serie.value_counts()/len(serie)>thresh).any()
# test = pd.Series([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,2,2,2,3,3,0,0,0,0,0])
# (test.value_counts()/len(test)>.9).any() 
















# # Class for regular expressions application
# class ApplyRegex(BaseEstimator, TransformerMixin):
    
#     def __init__(self, regex_transformers):
#         super().__init__()
#         self.regex_transformers = regex_transformers
        
#     def fit(self, X, y=None):
#         return self
    
#     def transform(self, X, y=None):
#         # Applying all regex functions in the regex_transformers dictionary
#         for regex_name, regex_function in self.regex_transformers.items():
#             X = regex_function(X)
            
#         return X
  
    
# # Class for stopwords removal from the corpus
# class StopWordsRemoval(BaseEstimator, TransformerMixin):
    
#     def __init__(self, text_stopwords):
#         super().__init__()
#         self.text_stopwords = text_stopwords
#     def fit(self, X, y=None):
#         return self
    
#     def transform(self, X, y=None):
#         return [' '.join(stopwords_removal(comment, self.text_stopwords)) for comment in X]

    
# # Class for apply the stemming process
# class StemmingProcess(BaseEstimator, TransformerMixin):
    
#     def __init__(self, stemmer):
#         super().__init__()
#         self.stemmer = stemmer
    
#     def fit(self, X, y=None):
#         return self
    
#     def transform(self, X, y=None):
#         return [' '.join(stemming_process(comment, self.stemmer)) for comment in X]
    
# # Class for extracting features from corpus
# class TextFeatureExtraction(BaseEstimator, TransformerMixin):
    
#     def __init__(self, vectorizer):
#         super().__init__()
#         self.vectorizer = vectorizer
        
#     def fit(self, X, y=None):
#         return self
    
#     def transform(self, X, y=None):
#         return self.vectorizer.fit_transform(X).toarray()