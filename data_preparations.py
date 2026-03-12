from lib.data_preparation import DataPreparation
from lib.preprocessor import Preprocessor
import joblib


vec_approaches = ['binary', 'bow', 'tf-idf']
proc_approaches_dict = {
    'methods':[None, 'stem', 'lemmatize','stem+'],
    # 'misspelling': [False, True],
    # 'lem+miss':[False, True]
}
try:
    df = DataPreparation.load()
    df_train = DataPreparation.save_train_test_split(df)


    X = df_train["tweet"].values.tolist()
    y = df_train["sentiment"].values.tolist()

    joblib.dump(y, "data/train_labels.pkl")

    datasets_paths = []

    for vec_a in vec_approaches:

        for key, methods in proc_approaches_dict.items():
            
            for m in methods:
                print(f"vectorization: {vec_a} + method used: {m}")
                transformed = Preprocessor.process(raws=X, processing_params={
                    'vectorization':vec_a,
                    'method':m
                }, show_trans=False)
                print(f"after vectorization {vec_a}")
                print("##__________________________________________________________________________##")

                m_ = 'only' if m == None else m

                path = f"data/{vec_a}_{m_}.pkl"

                joblib.dump(transformed, path)
                datasets_paths.append((path, (vec_a, m_)))

    joblib.dump(datasets_paths, "datasets_paths.pkl")
except Exception as e:
    print(f"exception : {str(e)}")