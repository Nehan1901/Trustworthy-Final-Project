import google.generativeai as genai
import os

# Configure API key
genai.configure(api_key="AIzaSyATF5nkfo7vddVB_wdzW6WJ5Fn92CorgoI")

# Use the latest available Gemini 1.5 Pro model
model = genai.GenerativeModel("gemini-1.5-pro-latest")  # Updated model name

def explain_action_with_llm(action, observation):
    action_map = {
        0: "Hold current course.",
        1: "Climb to a higher altitude.",
        2: "Turn to avoid potential collision."
    }

    prompt = f"""
You are an AI assistant trained in aviation safety.
Explain the pilot's decision based on flight conditions and the chosen action.

Action: {action_map.get(action, 'Unknown')}
Flight Conditions:
- Distance to nearby aircraft: {round(float(observation[0]), 2)} nautical miles
- Altitude Difference: {round(float(observation[1]), 2)} feet
- Velocity Difference: {round(float(observation[2]), 2)} knots
- Heading Difference: {round(float(observation[3]), 2)} degrees
- Vertical Rate Difference: {round(float(observation[4]), 2)} feet/min
"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating explanation: {e}"
