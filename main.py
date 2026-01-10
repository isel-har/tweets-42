# from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Input
from sklearn.metrics import accuracy_score
from preprocessor import Preprocessor
import pandas as pd
import numpy as np

def main():
    try:

        df_neg = pd.read_csv('data/processedNegative.csv')
        df_ntr = pd.read_csv('data/processedNeutral.csv')
        df_pos = pd.read_csv('data/processedPositive.csv')

        tweets = df_neg.columns.to_list()
        tweets.extend(df_ntr.columns.to_list())
        tweets.extend(df_pos.columns.to_list()) 

        y_ = [
            (df_neg.columns.size, float(0)),
            (df_ntr.columns.size, float(1)),
            (df_pos.columns.size, float(2))
        ]

        y = []
        for c in y_:
            for _ in range(c[0]):
                y.append(c[1])

        y = np.array(y)

        proc = Preprocessor(vectorization='tf-idf')
        X = proc.process(raw_sentences=tweets, processing_params={})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # model = OneVsRestClassifier(LogisticRegression(solver='liblinear', random_state=42))
        model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.01),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        model.fit(X_train, y_train, epochs=10, batch_size=32)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(np.argmax(y_pred, axis=1), y_test)

        ### cosine similarity

        print(f'word2vec + mlp  accuracy reached: {accuracy}')
        # y_pred = model.predict(X_test)
        # print("model is trained!")
        # sentence = input("enter a sentence:")
        
        # print("sentence processing...")
        # x = proc.process([sentence], processing_params={})
        
        # y_options = {
        #     0.0 : "negative",
        #     1.0 : "neutral",
        #     2.0 : "positive"
        # }

        # print(f"sentene is : {y_options[model.predict(x)[0]]}")

    except Exception as e:
        print("exception :", str(e))

if __name__ == "__main__":
    main()
