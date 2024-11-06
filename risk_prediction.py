#!/usr/bin/env python
# coding: utf-8

# In[2]:


import joblib
import pandas as pd

model2 = joblib.load('risk_prediction_model.pkl')

def get_user_input():
    print("\nPlease enter the following details:")

    # Individual Stocks
    individual_stocks = int(input("1. Do you own individual stocks? (1 = Yes, 0 = No): "))

    # Mutual Funds
    mutual_funds = int(input("2. Do you invest in mutual funds? (1 = Yes, 0 = No): "))

    # REITs
    reits = int(input("3. Do you invest in Real Estate Investment Trusts (REITs)? (1 = Yes, 0 = No): "))

    # Microcap Stocks or Penny Stocks
    microcap_stocks = int(input("4. Do you invest in microcap stocks or penny stocks? (1 = Yes, 0 = No): "))

    # Private Placements
    private_placements = int(input("5. Do you invest in private placements? (1 = Yes, 0 = No): "))

    # Whole Life Insurance
    whole_life_insurance = int(input("6. Do you have whole life insurance? (1 = Yes, 0 = No): "))

    # Individual Bonds
    individual_bonds = int(input("7. Do you invest in individual bonds? (1 = Yes, 0 = No): "))

    # Too Much Debt
    too_much_debt = int(input("8. Too much debt (1 - Strongly Disagree, ..., 7 - Strongly Agree): "))

    # Good at Dealing with Day-to-Day Financial Matters
    good_financial_matters = int(input("\n9. How good are you at dealing with day-to-day financial matters? (1-10): "))

    # Trading Experience
    trading_experience = int(input("\n10. How many years of trading experience do you have? (Enter a number, e.g., 5): "))

    # Total Investment
    total_investment = int(input("11. Total number of investments in the portfolio (Enter a number, e.g., 5): "))

    # Risk Level
    user_risk_level = int(input("12. Self-assessed risk level (1-10): "))

    # Construct the input dictionary
    input_data = {
        'Individual stocks': individual_stocks,
        'Mutual funds': mutual_funds,
        'REITs': reits,
        'Microcap stocks or penny stocks': microcap_stocks,
        'Private placements': private_placements,
        'Whole life insurance': whole_life_insurance,
        'Individual bonds': individual_bonds,
        'too much debt': too_much_debt,
        'good at dealing with day-to-day financial matters': good_financial_matters,
        'Trading experience': trading_experience,
        'Total investment': total_investment,
        'Risk level': user_risk_level
    }
    
    return input_data

# Get user input and create a DataFrame
input_data = get_user_input()
input_df = pd.DataFrame([input_data])

# Predict the risk score using the trained decision tree model
predicted_risk_score = model2.predict(input_df.values)[0]

# Display the predicted risk score
print(f"\nPredicted Risk Score: {predicted_risk_score}")


# In[ ]:




