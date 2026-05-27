def build_preferences_menu_response(preferences):
    if not preferences:
        return "You have not set any preferences yet."
    
    response = "Here are your current preferences:\n"
    for pref in preferences:
        response += f"- {pref} - {preferences[pref]}\n"
    
    return response