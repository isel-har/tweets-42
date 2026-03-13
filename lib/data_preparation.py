from sklearn.model_selection import train_test_split
import pandas as pd

class DataPreparation:

    @staticmethod
    def load():

        tweets_paths = {
            1:'data/raw/processedPositive.csv',
            -1:'data/raw/processedNegative.csv',
            0:'data/raw/processedNeutral.csv'
        }

        tweets = []
        sentiments = []
        for label, path in tweets_paths.items():
            tweets_tmp = pd.read_csv(path).columns.to_list().copy()
            labels = [label for n in range(len(tweets_tmp))]
            tweets.extend(tweets_tmp)
            sentiments.extend(labels)

        df = pd.DataFrame(data={'tweet':tweets, 'sentiment':sentiments})
        return df.sample(frac=1)


    @staticmethod
    def save_train_test_split(df, save=True, index=False):
        
        df_train, df_test = train_test_split(df, test_size=0.2)

        if save:
            df_train.to_csv('data/train.csv', index=index)
            df_test.to_csv('data/test.csv', index=index)
            return df_train, None

        return df_train, df_test
