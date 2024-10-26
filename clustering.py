import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib


data = pd.read_excel('data\cleaned_clustering_data.xlsx')
data = data.apply(lambda col: col.fillna(col.mode()[0]))
gender_map = {
    1: 'Male', 2: 'Male', 3: 'Male', 4: 'Male', 5: 'Male', 6: 'Male',
    7: 'Female', 8: 'Female', 9: 'Female', 10: 'Female', 11: 'Female', 12: 'Female'
}
age_map = {
    1: '18-24', 2: '25-34', 3: '35-44', 4: '45-54', 5: '55-64', 6: '65+',
    7: '18-24', 8: '25-34', 9: '35-44', 10: '45-54', 11: '55-64', 12: '65+'
}

def split_gender_age(value):
    return gender_map.get(value), age_map.get(value)

data[['Gender', 'Age Group']] = data['Gender&Age'].apply(lambda x: pd.Series(split_gender_age(x)))

data.drop(columns=['Gender&Age'], inplace=True)
gender_binary_map = {'Male': 1, 'Female': 0}
data['Gender'] = data['Gender'].map(gender_binary_map)
age_ordinal_map = {
    '18-24': 1,
    '25-34': 2,
    '35-44': 3,
    '45-54': 4,
    '55-64': 5,
    '65+': 6
}
data['Age Group'] = data['Age Group'].map(age_ordinal_map)

X = data.drop(columns=['Risk level', 'NFCSID']) 
y = data['Risk level']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

rfe = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=7)
rfe.fit(X_train, y_train)
selected_features = X.columns[rfe.support_]
print("Selected Features:", selected_features)

X_train = X_train[selected_features]
X_test = X_test[selected_features]
base_models = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('xgb', XGBClassifier(random_state=42))
]

# Define a meta-model
stack_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())

# Train the stacking model
stack_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_stack = stack_model.predict(X_test)
accuracy_stack = accuracy_score(y_test, y_pred_stack)
print(f"Stacking Model Accuracy: {accuracy_stack}")

joblib.dump(stack_model, 'risk_assessment_model.pkl')