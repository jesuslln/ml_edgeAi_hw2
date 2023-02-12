import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import numpy


# Get Data
train_df = pd.read_csv("./models_data/training_dataset.csv")
print(train_df.head())

test_df = pd.read_csv("./models_data/testing_blackscholes.csv")
print(test_df.head())

# Process data for training
train_data = train_df.drop(columns=['w_big', 'total_watts', 
                           'w_little', 'w_gpu', 'w_mem'])
y_train = pd.DataFrame(train_df['w_big'])
y_train.loc[y_train['w_big'] < 1, 'w_big'] = 0
y_train.loc[y_train['w_big'] > 1, 'w_big'] = 1

# New feature Vdd_f
train_data.loc[train_data['freq_big_cluster']/1000000000 == 0.9, 'vdd_f'] = 0.975*0.975*1000000000*0.9
train_data.loc[train_data['freq_big_cluster']/1000000000 == 1, 'vdd_f'] = 1000000000
train_data.loc[train_data['freq_big_cluster'] /1000000000 == 2, 'vdd_f'] = 1.362*1.362*2000000000


# Process data for training
test_data = test_df.drop(columns=['w_big', 'total_watts',
                                  'w_little', 'w_gpu', 'w_mem'])
y_test = pd.DataFrame(test_df['w_big'])
y_test.loc[y_test['w_big'] < 1, 'w_big'] = 0
y_test.loc[y_test['w_big'] > 1, 'w_big'] = 1
test_data.loc[test_data['freq_big_cluster']/1000000000 == 1.5, 'vdd_f'] = 1.1375*1.1375*1000000000*1.5


print(numpy.unique(train_data['freq_big_cluster']))
print(numpy.unique(test_data['freq_big_cluster']))
print(train_data)

# Apply MLP model 
mlp_model = MLPClassifier()
mlp_model.fit(train_data, y_train)
print(mlp_model)
y_pred = mlp_model.predict(test_data)

## Metric evaluation
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
