from lib.data_preparation import DataPreparation
from lib.preprocessor import Preprocessor
import joblib


vec_approaches = ['binary', 'bow', 'tf-idf']
proc_approaches_dict = {
    'methods':[None, 'stem', 'lemmatize','stem+'],
    # 'misspelling': [False, True],
    # 'lem+miss':[False, True]
}

df = DataPreparation.load()
df_train = DataPreparation.save_train_test_split(df)


X = df_train["tweet"].values.tolist()
y = df_train["sentiment"].values.tolist()

joblib.dump(y, "data/train_labels.pkl")

for vec_a in vec_approaches:

    for key, approaches in proc_approaches_dict.items():
        
        for a in approaches:
            print(f"vectorization: {vec_a} + method used: {a}")
            transformed = Preprocessor.process(raws=X, processing_params={
                'vectorization':vec_a,
                'method':a
            })
            print(f"after vectorization {vec_a}")
            print(transformed)
            print("##__________________________________________________________________________##")
            joblib.dump(transformed, f"data/{vec_a}_{a}.pkl")
