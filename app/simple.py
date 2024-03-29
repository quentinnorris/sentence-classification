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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

import numpy as np

# load data
texts, labels_index, labels = read_input_data("../data");
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2);

tfidf_vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), norm='l2'); #token_pattern=r'\b\w+\b'

text_clf = Pipeline([ 
    ('tfvec', tfidf_vec),
    #('clf', LinearSVC(0.9))
    #('clf', KNeighborsClassifier(n_neighbors=7))
    #('clf', MultinomialNB(alpha=1.8)),
    #('clf', LogisticRegression(C=3.1, class_weight='balanced')),
    #('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample'))
    ('clf', AdaBoostClassifier(n_estimators=100, )),
])

# comment out parmeters that you want to tune, not all of them are here yet
grid_params = {
    #### tfvec ####
    # 'tfvec__min_df': (2,3),
    # 'tfvec__max_df': (0.8, 0.9),
    # 'tfvec__ngram_range': ((1,3),(1,2)),
    # 'tfvec__sublinear_tf': (True, False),
    # 'tfvec__norm': ('l1', 'l2')
    #### LinearSVC() ####
    # 'clf__C': (0.5,0.8,1,1.5)
    #### KNeighboursClassifier ####
    #'clf__n_neighbors': (2,3,4,5,6,7,8)
    #### MultiNomialNB() ####
    #'clf__alpha': (1.7, 1.8, 1.9, 2, 2.1)
    #### LogisticRegression() ####
    # 'clf__C': (3, 3.1),
    # 'clf__solver': ('lbfgs', 'liblinear', 'sag', 'saga'),
    # 'clf__class_weight': ('balanced', None)
    #### RandomForestClassifier() ####
    # 'clf__n_estimators': (50, 100, 150)
    # 'clf__class_weight': ('balanced', 'balanced_subsample', None)
    #### AdaBoostClassifier() ####
    'clf__n_estimators': (30, 50, 100, 130)
    # 'clf__class_weight': ('balanced', 'balanced_subsample', None)
}

gsCV = GridSearchCV(text_clf, grid_params, cv=5, verbose=5);
gsCV.fit(X_train, y_train)

print("Best Score: ", gsCV.best_score_)
print("Best Params: ", gsCV.best_params_)

preds = gsCV.predict(X_test);

print(metrics.classification_report(y_test, preds, digits=5, target_names=labels_index.keys()))




