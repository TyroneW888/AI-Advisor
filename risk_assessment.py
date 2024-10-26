import joblib
import pandas as pd
model = joblib.load('risk_assessment_model.pkl')
def get_user_input():
    print("Please enter the following details:")

    too_much_debt = int(input("1. Too much debt (1 - Strongly Disagree, ..., 7 - Strongly Agree): "))
    
    print("\n2. How often do you have money left over at the end of the month?")
    print("   1: Never\n   2: Rarely\n   3: Sometimes\n   4: Often\n   5: Always")
    money_left = int(input("Choose the appropriate number: "))
    
    print("\n3. Household Annual Income:")
    print("   1: Less than $15,000\n   2: $15,000 - $24,999\n   3: $25,000 - $34,999\n   4: $35,000 - $49,999")
    print("   5: $50,000 - $74,999\n   6: $75,000 - $99,999\n   7: $100,000 - $149,999\n   8: $150,000 - $199,999")
    print("   9: $200,000 - $299,999\n   10: $300,000 or more")
    household_income = int(input("Choose the appropriate number: "))

    print("\n4. Current Employment Status:")
    print("   1: Self-employed\n   2: Full-time\n   3: Part-time\n   4: Homemaker\n   5: Student")
    print("   6: Permanently sick/disabled\n   7: Unemployed\n   8: Retired")
    employment_status = int(input("Choose the appropriate number: "))

    print("\n5. Age Group (Choose one of the following numbers):")
    print("   1: 18-24\n   2: 25-34\n   3: 35-44\n   4: 45-54\n   5: 55-64\n   6: 65+")
    age_group = int(input("Enter the number corresponding to your age group: "))
    user_risk_level = int(input("\nPlease provide your self-assessed risk level (1-10): "))
    input_data = {
        'too much debt': too_much_debt,
        'money left': money_left,
        'Household annual income': household_income,
        'Current employment status': employment_status,
        'Age Group': age_group
    }
    
    return input_data, user_risk_level

input_data, user_risk_level = get_user_input()
input_df = pd.DataFrame([input_data])
predicted_risk_level = model.predict(input_df)[0]
if predicted_risk_level < user_risk_level:
    print("\nWarning: The model suggests a lower risk level than provided. Please review!")
else:
    print("\nThe predicted risk level matches or exceeds the provided risk level.")
    
print(f"\nPredicted Risk Level: {predicted_risk_level}")
print(f"User-Provided Risk Level: {user_risk_level}")
