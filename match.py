def match_investors_for_user(risk_level, investors):
    if 0 <= risk_level < 2:
        investor_risk_level = 1
    elif 2 <= risk_level < 4:
        investor_risk_level = 2
    elif 4 <= risk_level < 6:
        investor_risk_level = 3
    elif 6 <= risk_level < 8:
        investor_risk_level = 4
    elif 8 <= risk_level <= 10:
        investor_risk_level = 5
    else:
        return [None, None, None]  

    potential_matches = investors[investors['risk'] == investor_risk_level]['portfolio name'].tolist()
    if len(potential_matches) >= 3:
        return potential_matches[:3]
    else:
        return potential_matches + [None] * (3 - len(potential_matches))
investors = pd.read_excel("portfolio.xlsx")
user_risk_level = predicted_risk_score
matched_investors = match_investors_for_user(user_risk_level, investors)

# print match result
print(f"user risk score is {user_risk_level}，matching investment port：")
print(f"port 1: {matched_investors[0]}")
print(f"port 2: {matched_investors[1]}")
print(f"port 3: {matched_investors[2]}")
