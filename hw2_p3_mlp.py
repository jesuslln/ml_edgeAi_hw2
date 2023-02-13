import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import metrics


# Get Data
train_df = pd.read_csv("./models_data/training_dataset.csv")
test_black_df = pd.read_csv("./models_data/testing_blackscholes.csv")
test_body_df = pd.read_csv("./models_data/testing_bodytrack.csv")

# Process data for training
train_data = train_df.drop(columns=['w_big', 'total_watts', 
                           'w_little', 'w_gpu', 'w_mem'])
train_data.drop(train_data.tail(1).index, inplace=True)

temp4_train = pd.DataFrame(train_df['temp4'])[1:len(train_df)]
temp5_train = pd.DataFrame(train_df['temp5'])[1:len(train_df)]
temp6_train = pd.DataFrame(train_df['temp6'])[1:len(train_df)]
temp7_train = pd.DataFrame(train_df['temp7'])[1:len(train_df)]

# Process data for training
test_data_blackshcoles = test_black_df.drop(columns=['w_big', 'total_watts',
                                  'w_little', 'w_gpu', 'w_mem'])
test_data_blackshcoles.drop(test_data_blackshcoles.tail(1).index, inplace=True)

temp4_black_test = pd.DataFrame(test_black_df['temp4'])[1:len(test_black_df)]
temp5_black_test = pd.DataFrame(test_black_df['temp5'])[1:len(test_black_df)]
temp6_black_test = pd.DataFrame(test_black_df['temp6'])[1:len(test_black_df)]
temp7_black_test = pd.DataFrame(test_black_df['temp7'])[1:len(test_black_df)]

# body track data

test_data_bodytrack = test_body_df.drop(columns=['w_big', 'total_watts',
                                                     'w_little', 'w_gpu', 'w_mem'])
test_data_bodytrack.drop(test_data_bodytrack.tail(1).index, inplace=True)

temp4_body_test = pd.DataFrame(test_body_df['temp4'])[1:len(test_body_df)]
temp5_body_test = pd.DataFrame(test_body_df['temp5'])[1:len(test_body_df)]
temp6_body_test = pd.DataFrame(test_body_df['temp6'])[1:len(test_body_df)]
temp7_body_test = pd.DataFrame(test_body_df['temp7'])[1:len(test_body_df)]

# Apply MLP model 
mlp_model_4 = MLPClassifier()
mlp_model_4.fit(train_data, temp4_train)
temp4_black_pred = mlp_model_4.predict(test_data_blackshcoles)
temp4_body_pred = mlp_model_4.predict(test_data_bodytrack)

# temp5
mlp_model_5 = MLPClassifier()
mlp_model_5.fit(train_data, temp5_train)
temp5_black_pred = mlp_model_5.predict(test_data_blackshcoles)
temp5_body_pred = mlp_model_5.predict(test_data_bodytrack)

# temp6
mlp_model_6 = MLPClassifier()
mlp_model_6.fit(train_data, temp6_train)
temp6_black_pred = mlp_model_6.predict(test_data_blackshcoles)
temp6_body_pred = mlp_model_6.predict(test_data_bodytrack)

# temp7
mlp_model_7 = MLPClassifier()
mlp_model_7.fit(train_data, temp7_train)
temp7_black_pred = mlp_model_7.predict(test_data_blackshcoles)
temp7_body_pred = mlp_model_7.predict(test_data_bodytrack)

## Metric evaluation
print(metrics.accuracy_score(temp4_black_test, temp4_black_pred))
print(metrics.accuracy_score(temp5_black_test, temp5_black_pred))
print(metrics.accuracy_score(temp6_black_test, temp6_black_pred))
print(metrics.accuracy_score(temp7_black_test, temp7_black_pred))
print(metrics.accuracy_score(temp4_body_test, temp4_body_pred))
print(metrics.accuracy_score(temp5_body_test, temp5_body_pred))
print(metrics.accuracy_score(temp6_body_test, temp6_body_pred))
print(metrics.accuracy_score(temp7_body_test, temp7_body_pred))

