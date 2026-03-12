from lib.data_preparation import DataPreparation

df = DataPreparation.load()

DataPreparation.save_train_test_split(df)