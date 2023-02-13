import pandas as pd
from sklearn.linear_model import LinearRegression

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

# New feature Vdd_f
train_data.loc[train_data['freq_big_cluster'] /
               1000000000 == 0.9, 'vdd_f'] = 0.975*0.975*1000000000*0.9
train_data.loc[train_data['freq_big_cluster'] /
               1000000000 == 1, 'vdd_f'] = 1000000000
train_data.loc[train_data['freq_big_cluster'] /
               1000000000 == 2, 'vdd_f'] = 1.362*1.362*2000000000

# Process data for training
test_data = test_df.drop('w_big', axis=1)
y_test = pd.DataFrame(test_df['w_big'])
y_test.loc[y_test['w_big'] < 1, 'w_big'] = 0
y_test.loc[y_test['w_big'] > 1, 'w_big'] = 1

test_data.loc[test_data['freq_big_cluster']/1000000000 ==
              1.5, 'vdd_f'] = 1.1375*1.1375*1000000000*1.5

## Apply LRM model 
reg = LinearRegression().fit(train_data,y_train['w_big'])
print(reg.score(train_data,y_train['w_big']))
print(reg.score(test_data,y_test['w_big']))