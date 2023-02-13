import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Get Data
train_df = pd.read_csv("./models_data/training_dataset.csv")
print(train_df.head())

#  test_df = pd.read_csv("./models_data/testing_blackscholes.csv")
test_df = pd.read_csv("./models_data/testing_bodytrack.csv")
print(test_df.head())

# Process data for training
train_data = train_df.drop(columns=['total_watts','w_big','w_little','w_gpu','w_mem'])
y_train = pd.DataFrame(train_df['w_big'])

# Process data for testing
test_data = test_df.drop(columns=['total_watts','w_big','w_little','w_gpu','w_mem'])
y_test = pd.DataFrame(test_df['w_big'])

## Apply LRM model 
reg = LinearRegression().fit(train_data,y_train['w_big'])
print(reg.score(train_data,y_train['w_big']))
y_pred = reg.predict(test_data)
print(mean_squared_error(y_test['w_big'],y_pred))

#path for plots
plot_path = "plots/question2"
time = np.arange(0, test_df.shape[0])

plt.figure()
plt.plot(time, test_df['w_big'], 'o', label='True Values')
plt.plot(time, y_pred, label='Predicted Values')
plt.legend()
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('Power [W]')
#  plt.title('Predicted vs Actual: blackscholes')
plt.title('Predicted vs Actual: bodytrack')
#  file_name = plot_path + "_hw2_q2_blackscholes.png"
file_name = plot_path + "_hw2_q2_bodytrack.png"
plt.savefig(file_name)
plt.show()