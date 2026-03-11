from sklearn.model_selection import train_test_split
from preprocessor import Preprocessor
import pandas as pd
import numpy as np

df_neg = pd.read_csv("../data/processedNegative.csv")
df_ntr = pd.read_csv("../data/processedNeutral.csv")
df_pos = pd.read_csv("../data/processedPositive.csv")


X_neg = df_neg.columns.to_numpy(dtype=str)
X_ntr = df_ntr.columns.to_numpy(dtype=str)
X_pos = df_pos.columns.to_numpy(dtype=str)

y_neg = np.zeros(len(X_neg), dtype=int)
y_ntr = np.ones(len(X_ntr), dtype=int)
y_pos = np.full(len(X_pos), 2, dtype=int)

X = np.concatenate([X_neg, X_ntr, X_pos])
y = np.concatenate([y_neg, y_ntr, y_pos])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Raw:", X_train[:3], "\n")

cleaned = Preprocessor.clean(X_train)
tokenized = Preprocessor.tokenize(cleaned)

print("cleaned + tokenized tweets", tokenized[:3], "\n")

lemmatized = Preprocessor.lemmatize(tokenized)
stemmed = Preprocessor.stemming(tokenized)
stemmed_plus = Preprocessor.stemming_plus(tokenized)


print("Lemmatized:", lemmatized[:3], "\n")
print("Stemmed: ", stemmed[:3], "\n")
print("Stem+: ", stemmed_plus[:3], "\n")

lem_stopwords = Preprocessor.stopwords(lemmatized)

print("Lemmatized + stopwords: ", lem_stopwords[:3], "\n")

print("vectorization using different approaches...")
vec_proccesses = {
    "bow_lemmatized": Preprocessor.vectorize(lemmatized, vectorization="bow"),
    "tf_idf_lemmatized": Preprocessor.vectorize(lemmatized, vectorization="tf-idf"),
    "binary_stemmed": Preprocessor.vectorize(stemmed, vectorization="binary"),
    "tf_idf_stemmed": Preprocessor.vectorize(stemmed, vectorization="tf-idf"),
    "bow_stopwords": Preprocessor.vectorize(lem_stopwords, vectorization="bow"),
    "tf_idf_stopwords": Preprocessor.vectorize(lem_stopwords, vectorization="tf-idf"),
    "word2vec_lemmatized": Preprocessor.vectorize(
        lem_stopwords, vectorization="word2vec"
    ),
}

for name, vectors in vec_proccesses.items():
    print(f"{name} shape: {vectors.shape}")
    print(vectors[:3])

print("save train test split of raw tweets")

np.savez("dataset_split.npz",
         X_train=X_train,
         X_test=X_test,
         y_train=y_train,
         y_test=y_test)