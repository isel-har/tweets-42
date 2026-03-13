from lib.data_preparation import DataPreparation
from lib.preprocessor import Preprocessor
import joblib


vec_approaches = ['binary', 'bow', 'tf-idf']
methods = [None, 'stem', 'lemmatize','stem+', 'misspelling', 'lem+misspelling']

try:
    df = DataPreparation.load()
    df_train, _ = DataPreparation.save_train_test_split(df)

    X = df_train["tweet"].values.tolist()
    y = df_train["sentiment"].values.tolist()

    joblib.dump(y, "data/train_labels.pkl")

    del y

    datasets_paths = []

    for vec_a in vec_approaches:
    
        for m in methods:

            print(f"vectorizer : {vec_a}, method {m}")
            X_transformed, vectorizer = Preprocessor.process(
                raws=X,
                processing_params={
                'vectorization':vec_a,
                'method':m
                },
                show_trans=False
            )

            method_used = 'only' if m == None else m

            path = f"data/{vec_a}_{method_used}.pkl"
            joblib.dump(X_transformed, path)
            datasets_paths.append((path, (vectorizer, method_used)))


    joblib.dump(datasets_paths, "datasets_paths.pkl")
    print("datasets are saved in data/")

except Exception as e:
    print(f"exception : {str(e)}")