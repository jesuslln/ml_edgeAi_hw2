import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Get Data
train_df = pd.read_csv("./models_data/training_dataset.csv")
for index, row in train_df.iterrows():
    freq = train_df.loc[index,'freq_big_cluster']
    if freq == 900000000:
        v_dd = 0.975
    elif freq == 1000000000:
        v_dd = 1
    elif freq == 1500000000:
        v_dd = 1.1375
    elif freq == 2000000000:
        vdd = 1.362

    train_df.at[index, 'dyn_w'] = pow(v_dd,2) * freq
print(train_df.head())

# Process data for training
train_data = train_df.drop(columns=['total_watts','w_big','w_little','w_gpu','w_mem'])
y_train = pd.DataFrame(train_df['w_big'])

scaler = StandardScaler()
scaler.fit(train_data)

# Transform data with scaler
train_data_scaled = scaler.transform(train_data)

## Apply LRM model 
reg = LinearRegression().fit(train_data_scaled,y_train['w_big'])

# Get coefficients and feature importances
coefficients = reg.coef_
feature_importances = abs(coefficients) / sum(abs(coefficients))

# Plot feature importances
features = train_data.columns
indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10,5))
plt.title("Feature Importances")
plt.grid(True)
plt.bar(range(train_data.shape[1]), feature_importances[indices])
plt.xticks(range(train_data.shape[1]), features[indices], rotation=90)
plt.ylabel('Importance')
plt.xlabel('Features')
file_name = "plots/hw2_q3.png"
plt.savefig(file_name)
plt.show()
