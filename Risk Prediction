# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn import tree
import joblib

# Load the dataset
data = pd.read_csv('data\merged_df.csv')

# Drop the user ID column (assumed to be the first column)
data = data.iloc[:, 1:]

# Define a numerical risk score from 0 to 10 based on selected features
# Risky investments contribute to higher risk; conservative investments contribute to lower risk

data['risk_numerical'] = 0  # Initialize with 0

# Add points for risky assets
data['risk_numerical'] += np.where(data['Individual stocks'] == 1, 3, 0)
data['risk_numerical'] += np.where(data['Microcap stocks or penny stocks'] == 1, 2, 0)
data['risk_numerical'] += np.where(data['Mutual funds'] == 1, 1, 0)
data['risk_numerical'] += np.where(data['Private placements'] == 1, 1, 0)
data['risk_numerical'] += np.where(data['REITs'] == 1, 1, 0)

# Subtract points for conservative assets
data['risk_numerical'] -= np.where(data['Whole life insurance'] == 1, 2, 0)
data['risk_numerical'] -= np.where(data['Individual bonds'] == 1, 2, 0)

# Add or subtract points for other financial behaviors
data['risk_numerical'] += np.where(data['too much debt'] >= 4, 2, 0)
data['risk_numerical'] -= np.where(data['good at dealing with day-to-day financial matters'] >= 6, 1, 0)
data['risk_numerical'] += np.where(data['Trading experience'] >= 3, 1, 0)
data['risk_numerical'] -= np.where(data['Trading experience'] < 3, 1, 0)
data['risk_numerical'] += np.where(data['Total investment'] >= 6, 1, 0)
data['risk_numerical'] -= np.where(data['Total investment'] < 6, 1, 0)
data['risk_numerical'] += np.where(data['Risk level'] >= 3, 2, 0)

# Ensure the score is within the range of 0 to 10
data['risk_numerical'] = data['risk_numerical'].clip(0, 10)

# Define the features and target variable
features = ["Individual stocks", "Mutual funds", "REITs", "Microcap stocks or penny stocks", 
            "Private placements", "Whole life insurance", "Individual bonds", 
            "too much debt", "good at dealing with day-to-day financial matters", 
            "Trading experience", "Total investment", "Risk level"]
X = data[features]
y = data['risk_numerical']

# Impute missing values with the mean of each column
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Build the decision tree model
tree_model = DecisionTreeRegressor(min_samples_split=20, random_state=123)
tree_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = tree_model.predict(X_test)

# Evaluate the model performance using MAE and RMSE
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot the decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(tree_model, feature_names=features, filled=True, rounded=True)
plt.title("Decision Tree for Risk Numerical Prediction")
plt.show()

joblib.dump(tree_model, 'risk_prediction_model.pkl')
