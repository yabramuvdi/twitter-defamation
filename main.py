import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# The objective of this script is to accurately predict if a tweet
# contains defamatory language against Spain's current president (Pedro
# Sanchez). To do this, we will use some preprocessed tweets that contain a
# label indicating if they are defamatory (=1) or not (=0). We will train a
# variety of models and, through a comprehensive gridsearch cross validation,
# will find the best performing one. Finally, we will predict a defamation
# score for a set of tweets for which we did not have any label.

#######################################
# Preparing the data
#######################################

df = pd.read_csv('spain_twitter_data.csv')
# some additional preprocessing
min_tweet_length = 4
tweet_column = 'tweet_words'
df = df[df[tweet_column].apply(lambda x: len(x.split(',')) > min_tweet_length)]
# split the data into labeled and unlabeled tweets
df_label = df[df.is_labeled == 1]
df_no_label = df[df.is_labeled == 0]
# we will be working with the lemmatized version of the tweets
tweets_label = df_label.lemmas
tweets_no_label = df_no_label.lemmas
y_label = df_label.defamatory
# train test split
X_train, X_test, y_train, y_test = train_test_split(tweets_label,
                                                    y_label,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle=True)


#######################################
# Searching for the best model
#######################################


def pipeline_gridsearch(x, y, pipeline, parameters, cv, score='accuracy'):
    """ function to perform gridsearch crossvalidation for a given pipeline
    and return the best performing model
    """

    best_model = GridSearchCV(pipeline, parameters, cv=cv,
                              scoring=score, verbose=0)
    best_model.fit(x, y)
    pipeline.set_params(**best_model.best_params_)
    return best_model


def evaluate(x, y, model, scoring_func):
    """ function to predict a class given a set of features and report the
    score from the given function and the confusion matrix
    """

    y_hat = model.predict(x)
    score_res = scoring_func(y, y_hat)
    confusion_mat = confusion_matrix(y_test, y_hat)
    return (score_res, confusion_mat)


# define pipelines
pipelines = [Pipeline([('vect', TfidfVectorizer()),
                       ('RF', RandomForestClassifier())]),
             Pipeline([('vect', TfidfVectorizer()),
                       ('KN', KNeighborsClassifier())]),
             Pipeline([('vect', TfidfVectorizer()),
                       ('LR', LogisticRegression(solver='liblinear',
                                                 max_iter=1000))]),
             Pipeline([('vect', TfidfVectorizer()),
                       ('NB', MultinomialNB())])]

# models parameters
params_RF = {'vect__min_df': (0.001, 0.005),
             'vect__max_df': (0.97, 0.9),
             'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
             'RF__n_estimators': (10, 100, 200),
             'RF__max_depth': (None, 10, 20),
             'RF__max_features': ('auto', 'log2')
             }

params_KN = {'vect__min_df': (0.001, 0.005, 0.01),
             'vect__max_df': (0.97, 0.9),
             'vect__ngram_range': ((1, 1), (1, 2)),
             'KN__n_neighbors': (1, 3, 5),
             'KN__weights': ('distance', 'uniform')
             }

params_LR = {'vect__min_df': (0.001, 0.005, 0.01),
             'vect__max_df': (0.97, 0.9),
             'vect__ngram_range': ((1, 1), (1, 2)),
             'LR__penalty': ('l1', 'l2')
             }

params_NB = {'vect__min_df': (0.001, 0.005, 0.01),
             'vect__max_df': (0.97, 0.9),
             'vect__ngram_range': ((1, 1), (1, 2)),
             'NB__alpha': (0.5, 1)
             }

parameters = [params_RF, params_KN, params_LR, params_NB]
names = ['Vect_RF', 'Vect_KN', 'Vect_LR', 'Vect_NB']
models = {}
cross_validations = 2
scoring_function = 'f1_micro'

for pipeline, params, name in zip(pipelines, parameters, names):
    # grdisearch and cross_validate
    best_model = pipeline_gridsearch(X_train, y_train, pipeline,
                                     params, cross_validations,
                                     scoring_function)
    # evaluate
    score, confusion_mat = evaluate(X_test, y_test, best_model, f1_score)
    # save results
    models[name] = {'best_model': best_model, 'score': score,
                    'confusion_mat': confusion_mat, 'pipeline': pipeline,
                    'parameters': params}
# display results
results = pd.DataFrame({
    'Model': names,
    'Score': [values['score'] for name, values in models.items()]})
results = results.sort_values(by='Score', ascending=False)
print(results)
# get the name of the best model
best_model_name = results.iloc[0, 0]

#######################################
# Fitting the optimal models
#######################################

for name, values in models.items():
    # get and fit the vectorizer from the pipeline
    vectorizer = values['pipeline'][0]
    vectorizer = vectorizer.fit(tweets_label)
    # get and fit the model from the pipeline
    model = values['pipeline'][1]
    X_label = vectorizer.transform(tweets_label)
    model.fit(X_label, y_label)
    X_no_label = vectorizer.transform(tweets_no_label)
    # calculate probabilities (can be interpreted as defamation scores)
    defam_score = model.predict_proba(X_no_label)[:, 1]
    models[name]['defam_scores'] = defam_score

# display results for the model with best results
cols = ['defamation_score', 'tweets']
defam_tweets = pd.DataFrame({'defamation_score':
                                 models[best_model_name]['defam_scores'],
                             'tweets': df_no_label.tweet_text})
defam_tweets = defam_tweets.sort_values(['defamation_score'], ascending=False)
print(defam_tweets.head(10))
