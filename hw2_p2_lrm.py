import pandas as pd
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

# Get Data
train_df = pd.read_csv("./models_data/training_dataset.csv")
print(train_df.head())

test_df = pd.read_csv("./models_data/testing_blackscholes.csv")
print(test_df.head())

# Process data for training
train_data = train_df.drop('w_big', axis=1)
y_train = pd.DataFrame(train_df['w_big'])
y_train.loc[y_train['w_big'] < 1, 'w_big'] = 0
y_train.loc[y_train['w_big'] > 1, 'w_big'] = 1

# Process data for training
test_data = test_df.drop('w_big', axis=1)
y_test = pd.DataFrame(test_df['w_big'])
y_test.loc[y_test['w_big'] < 1, 'w_big'] = 0
y_test.loc[y_test['w_big'] > 1, 'w_big'] = 1

## Apply LRM model 
reg = LinearRegression().fit(train_data,y_train['w_big'])
print(reg.score(train_data,y_train['w_big']))
print(reg.score(test_data,y_test['w_big']))