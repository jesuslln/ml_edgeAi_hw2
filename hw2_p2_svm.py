import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics


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

print(y_train)
print(y_test)

# Apply SVC model 
svclassifier = SVC(kernel='linear')
svclassifier.fit(train_data,y_train['w_big'])
y_pred = svclassifier.predict(test_data)

## Metric evaluation
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

print(metrics.confusion_matrix(y_test, y_pred))