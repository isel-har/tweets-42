from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score#, roc_auc_score
from sklearn.base import clone
from lib.preprocessor import Preprocessor

import pandas as pd
import joblib

try:
    models = {
        'logistic regression':OneVsRestClassifier(LogisticRegression()),
        'Decision tree': DecisionTreeClassifier(),
        'multi nomial': MultinomialNB()
    }

    grid_params_dict = {
        'logistic regression':{
            'estimator__solver': ['liblinear'],
            'estimator__penalty': ['l1', 'l2'],
            'estimator__max_iter':[100, 200, 300]
        },
        'Decision tree':{
            'criterion': ['gini'],
            'max_depth': [None, 2],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [1, 4]
        },
        'multi nomial' :{
        'alpha': [0.5, 10.0],
        'fit_prior': [True, False]
        }
    }

    datasets_paths = joblib.load("datasets_paths.pkl")

    X       = joblib.load(datasets_paths[3][0])
    y_train = joblib.load('data/train_labels.pkl')


    best_estimators = {}
    for key, val in models.items():
        best_estimators[key] = None

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



    top_score = 0.0
    top_approach = ""
    top_ml_algo  = ""
    for path, approache in datasets_paths:
        
        X_train    = joblib.load(path)
        vectorizer = approache[0]

        X_test_processed, _ = Preprocessor.process(
                raws=X_test,
                processing_params={
                    "method": None if approache[1] == 'only' else approache[1],
                },
                show_trans=False,
                vectorizer=vectorizer
        )

        for name, model in best_estimators.items():


            copy_model = clone(model)
            copy_model.fit(X_train, y_train)

            y_pred = copy_model.predict(X_test_processed)

            acc = accuracy_score(y_pred=y_pred, y_true=y_test)
            if acc > top_score:
                top_approach = f"{vectorizer.__class__.__name__}, {approache[1]}"
                top_ml_algo = name
                top_score = acc

            with open("results_table.txt", 'a') as f:

                f.write(f"""ML algorithm used : {name}\n
                method used : {approache[1]}\n
                vectorizer used : {vectorizer.__class__.__name__}\n
                best hyperparams: {model.get_params()}\n
                score : {acc}\n
                ______________________________________________\n
                """)


    with open("results_table.txt", 'a') as f:

        f.write(f"""
        Highest score is {top_score} using:
        ml algo:{top_ml_algo}\n
        approaches: {top_approach}
        """)
    


except Exception as e:
    print("error :", str(e))