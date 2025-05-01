import google.generativeai as genai
import os


# Make sure your key is set in the environment
genai.configure(api_key="AIzaSyATF5nkfo7vddVB_wdzW6WJ5Fn92CorgoI")

try:
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content("Say hello like a pilot.")
    print("Gemini Pro is working:")
    print(response.text.strip())


except Exception as e:
    print("Gemini Pro is NOT working:")
    print(e)
