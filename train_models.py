from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from lib.preprocessor import Preprocessor

import pandas as pd
import joblib

try:

    datasets_paths = joblib.load("datasets_paths.pkl")

    models = {
        'logistic regression':OneVsRestClassifier(LogisticRegression()),
        'Decision tree': DecisionTreeClassifier(),
        'multi nomial': MultinomialNB()
    }

    grid_params_dict = {
        'logistic regression':{
            'estimator__solver': ['liblinear'],
            'estimator__penalty': ['l1', 'l2'],
        },
        'Decision tree':{
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 2, 4, 6, 8, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'multi nomial' :{
        'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0],
        'fit_prior': [True, False]
        }
    }


    best_estimators = {}
    for key, val in models.items():
        best_estimators[key] = None

    X       = joblib.load(datasets_paths[3][0])
    y_train = joblib.load('data/train_labels.pkl')

    for esti_name, param_grid in grid_params_dict.items():

        grid_search = GridSearchCV(
            estimator=models[esti_name],
            param_grid=param_grid,
            scoring='accuracy',
            cv=5,
            n_jobs=-1
        )
        grid_search.fit(X, y_train)

        print(f"{esti_name}:")
        print(f"Best Alpha: {grid_search.best_params_}")
        print(f"Best Score: {grid_search.best_score_:.4f}")
        print("_________________________________________________")
        best_estimators[esti_name] = grid_search.best_estimator_


    del X

    df_test = pd.read_csv('data/test.csv')
    X_test = df_test["tweet"].values.tolist()
    y_test = df_test["sentiment"].values.tolist()


    for path, approache in datasets_paths:
        
        X_train = joblib.load(path)

        print(f"{approache[0]} + {approache[1]}")

        for name, model in best_estimators.items():

            print(f"machine learning algorithm : {name}")

            copy_model = clone(model)  # already has best params
            copy_model.fit(X_train, y_train)

            print("fit done!")
            # X_test_processed = Preprocessor.process(
            #     raws=X_test,
            #     processing_params={
            #         "vectorization": approache[0],
            #         "method": None if approache[1] == 'only' else approache[1],
            #     },
            #     show_trans=False
            # )

            # y_pred = copy_model.predict(X_test_processed)
            # try:
            #     print(f"accuracy score : {accuracy_score(y_pred=y_pred, y_true=y_test)}")
            # except Exception as e:
            #     print("f accuracy :", str(e))


except Exception as e:
    print("error :", str(e))
