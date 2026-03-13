from lib.preprocessor import Preprocessor
from lib.data_preparation import DataPreparation
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score


df = DataPreparation.load()
df_train, df_test = DataPreparation.save_train_test_split(df, save=False)


X = df_train['tweet'].values.tolist()
y = df_train['sentiment'].values.tolist()


X_tranformed, _ = Preprocessor.process(
        raws=X,
        processing_params={
            'method':'lemmatize',
            'vectorization': 'word2vec'
        },
        show_trans=False,
        stopwords=True
    )


model = OneVsRestClassifier(LogisticRegression(max_iter=100, penalty='l1', solver='liblinear'))
model.fit(X_tranformed, y)

X_test = df_test['tweet'].values.tolist()
y_test = df_test['sentiment'].values.tolist()

X_test_transformed, _ = Preprocessor.process(
        raws=X_test,
        processing_params={
            'method':'lemmatize',
            'vectorization': 'word2vec'
        },
        show_trans=False,
        stopwords=True
)
y_pred = model.predict(X_test_transformed)

print(f"word2vec accuracy: {accuracy_score(y_pred=y_pred, y_true=y_test)}")

