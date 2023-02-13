import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Get Data
train_df = pd.read_csv("./models_data/training_dataset.csv")
print(train_df.head())

# test_df = pd.read_csv("./models_data/testing_blackscholes.csv")
test_df = pd.read_csv("./models_data/testing_bodytrack.csv")
print(test_df.head())

# Process data for training
train_data = train_df.drop('w_big', axis=1)
y_train = pd.DataFrame(train_df['w_big'])
y_train.loc[y_train['w_big'] < 1, 'w_big'] = 0
y_train.loc[y_train['w_big'] > 1, 'w_big'] = 1

# Initialize the scaler
scaler = MinMaxScaler()

# Normalize temps & freq
train_data['temp4'] = scaler.fit_transform(train_data[['temp4']])
train_data['temp5'] = scaler.fit_transform(train_data[['temp5']])
train_data['temp6'] = scaler.fit_transform(train_data[['temp6']])
train_data['temp7'] = scaler.fit_transform(train_data[['temp7']])
train_data['temp_gpu'] = scaler.fit_transform(train_data[['temp_gpu']])
train_data['freq_big_cluster'] = scaler.fit_transform(train_data[['freq_big_cluster']])

# Process data for testing
test_data = test_df.drop('w_big', axis=1)
y_test = pd.DataFrame(test_df['w_big'])
y_test.loc[y_test['w_big'] < 1, 'w_big'] = 0
y_test.loc[y_test['w_big'] > 1, 'w_big'] = 1

# Normalize temps & freq
test_data['temp4'] = scaler.fit_transform(test_data[['temp4']])
test_data['temp5'] = scaler.fit_transform(test_data[['temp5']])
test_data['temp6'] = scaler.fit_transform(test_data[['temp6']])
test_data['temp7'] = scaler.fit_transform(test_data[['temp7']])
test_data['temp_gpu'] = scaler.fit_transform(test_data[['temp_gpu']])
test_data['freq_big_cluster'] = scaler.fit_transform(test_data[['freq_big_cluster']])

# Apply SVC model 
svclassifier = SVC(kernel='linear')
svclassifier.fit(train_data,y_train['w_big'])
y_pred = svclassifier.predict(test_data)

## Metric evaluation
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

labels = [0,1]
cm = metrics.confusion_matrix(y_test, y_pred, labels=labels)
# Create a figure and axis object
fig, ax = plt.subplots()

# Create a heatmap using the confusion matrix
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

# Label the axes
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=['Idle','Active'], yticklabels=['Idle','Active'],
       xlabel='Predicted label', ylabel='True label')

# Loop over data dimensions and create text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

# Add a title to the plot
# ax.set_title("Confusion Matrix: blackscholes")
ax.set_title("Confusion Matrix: bodytrack")

# Show the plot
plt.show()