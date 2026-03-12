from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib

datasets_paths = joblib.load("datasets_paths.pkl")

train_dataset  = pd.read_csv("data/train.csv")

tweets = train_dataset['tweet']


for path, approache in datasets_paths:

    X = joblib.load(path)
    cos_sim = cosine_similarity(X)

    pairs = []
    n = len(X)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, cos_sim[i , j]))


    top_10 = sorted(pairs, key=lambda x: x[2], reverse=True)[:10]
    
    print(f"top 10 similar tweets using {approache[0]} + {approache[1]}")
    for i, j, score in top_10:
        print(f"[{score:.3f}]")
        print("Tweet A:", tweets[i])
        print("Tweet B:", tweets[j])
        print("-" * 50)

