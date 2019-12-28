from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import numpy as np

def get_scores(clf, model_data, fasttext_data, model_fasttext_data,y, cv):

    print('model')
    cvs = cross_val_score(estimator=clf,
                          X=np.array(model_data),
                          y=np.array(y),
                          scoring='roc_auc',
                          cv=cv)
    print(np.mean(cvs), cvs)

    print('fasttext')
    cvs = cross_val_score(estimator=clf,
                          X=np.array(fasttext_data),
                          y=np.array(y),
                          scoring='roc_auc',
                          cv=cv)
    print(np.mean(cvs), cvs)

    print('fasttext + model')
    cvs = cross_val_score(estimator=clf,
                          X=np.array(model_fasttext_data),
                          y=np.array(y),
                          scoring='roc_auc',
                          cv=cv)
    print(np.mean(cvs), cvs)


def print_score(model_data, fasttext_data, model_fasttext_data, y):
    cv = StratifiedKFold(n_splits=3, random_state=1)

    print('LogisticRegression')
    clf = LogisticRegression()
    get_scores(clf, model_data, fasttext_data, model_fasttext_data, y, cv)
    print()
    print('XGBoost')
    clf = XGBClassifier()
    get_scores(clf, model_data, fasttext_data, model_fasttext_data, y, cv)

