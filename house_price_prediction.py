import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD THE DATASET
train_data_path = r"C:\Users\PRAGYAN\Desktop\house_price_pred\train.csv"
test_data_path = r"C:\Users\PRAGYAN\Desktop\house_price_pred\test.csv"

df_train = pd.read_csv(train_data_path)
df_test = pd.read_csv(test_data_path)

df = pd.concat([df_train, df_test])

print("Shape of Integrated Data/df: ", df.shape)

print("Shape of df_train:", df_train.shape)
print("Shape of df_test:", df_test.shape)

df_train.head()
df_test.head()

null_count = df.isnull().sum()
null_count
null_percent = df.isnull().sum()/df.shape[0] * 100
null_percent

df = df.dropna()

# Pairplot to visualize relationships between features and target variable
sns.pairplot(df)
plt.show()

# Pairplot to visualize relationships between features and target variable
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Selecting features and target variable from the DataFrame
features = ['square_footage', 'bedrooms', 'bathrooms']
target = 'price'

X = df[features]
y = df[target]

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the Linear Regression model
from sklearn.linear_model import LinearRegression
linear_reg_model = LinearRegression()

# Training the model with training data
linear_reg_model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = linear_reg_model.predict(X_test)

# Calculating Mean Squared Error
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculating R^2 Score
r2 = r2_score(y_test, y_pred)
print(f'R^2 Score: {r2}')

# Plotting actual vs predicted prices to visualize model performance
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

# Predicting the price of a new house based on given features
new_house_data = pd.DataFrame({
    'square_footage': [2000],
    'bedrooms': [3],
    'bathrooms': [2]
})

predicted_price = linear_reg_model.predict(new_house_data)
print(f'Predicted Price for the new house: {predicted_price[0]}')