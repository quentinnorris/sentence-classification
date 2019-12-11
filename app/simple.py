import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from reader.filereader import read_glove_vectors, read_input_data

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn import svm

import numpy as np

# load data
texts, labels_index, labels = read_input_data("../data");
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

print(texts);

tfidf_vec = TfidfVectorizer(stop_words="english"); #token_pattern=r'\b\w+\b'

text_clf = Pipeline([ 
    ('tfvec', tfidf_vec),
    ('clf', svm.LinearSVC())
    #('clf', KNeighborsClassifier()),
    #('clf', MultinomialNB()),
    #('clf', LogisticRegression()),
    #('clf', DecisionTreeClassifier()),
    #('clf', RandomForestClassifier()),
])

# comment out parmeters that you want to tune, not all of them are here yet
grid_params = {
    # 'tfvec__min_df': (2,3),
    'tfvec__max_df': (0.8, 0.9),
    # 'tfvec__ngram_range': ((1,3),(1,2)),
    # 'tfvec__sublinear_tf': (True, False),
    # 'tfvec__norm': ('l1', 'l2'),
    # 'clf__alpha': np.linspace(1, 1.5, 6), # For Naive Bayes
    # 'clf__fit_prior': [True, False], # For Naive Bayes
    # 'clf__decision_function_shape': ('ovo', 'ovr'), # For svm.SVC
    # 'clf__max_iter': np.arange(600, 6000, 600),
    # 'clf__fit_intercept': (True, False),
    # 'clf__intercept_scaling': (0.1, 2.0, 0.2),
    # 'clf__multi_class': ('ovr', 'crammer_singer'),
    # 'clf__dual': (True, False),
    # 'clf__penalty': ('l1', 'l2'),
    # 'clf__C': np.arange(0.2, 2.0, 0.2),
    # 'clf__tol': np.arange(0.00002, 0.0002, 0.00006),
    # 'clf__solver': ('newton-cg', 'lbfgs'),
    # 'clf__n_estimators': (10, 100),
    # 'clf__random_state': (0,1,2),
    # 'clf__max_features': ('auto', 30, 60),
    # 'clf__learning_rate': ('optimal', 'optimal'),
    # 'clf__eta0': (1,1),
    # 'clf__average': (1, 5, 10,20),
    # 'clf__epsilon': (0.5, 1),
    # 'clf__loss': ('modified_huber', 'huber'),
    # 'clf__alpha': (0.00001, 0.0001, 0.001),
}

gsCV = GridSearchCV(text_clf, grid_params, verbose=5);
gsCV.fit(X_train, y_train)

print("Best Score: ", gsCV.best_score_)
print("Best Params: ", gsCV.best_params_)

preds = gsCV.predict(X_test);

print(metrics.classification_report(y_test, preds, digits=5, target_names=labels_index.keys()))




