from flask import Flask, render_template, request, jsonify
from joblib import load
from google import genai
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)

model = load('rf_model.joblib')

gemini_api_key = os.getenv('GEMINI_API_KEY')


def predict_health_score(user_data):
    user_data_modified = {
        'Age': user_data['Age'],
        'Gender_Encode': user_data['Gender'],
        'Sleep_Hours': user_data['Sleep_Hours'],
        'Work_Hours': user_data['Work_Hours'],
        'Physical_Activity_Hours': user_data['Physical_Activity_Hours'],
        'Social_Media_Usage': user_data['Social_Media_Usage'],
        'Stress_Level_Encode': user_data['Stress_Level'],
        'Diet_Quality_Encode': user_data['Diet_Quality'],
        'Smoking_Habit_Encode': user_data['Smoking_Habit'],
        'Alcohol_Consumption_Encode': user_data['Alcohol_Consumption']
    }
    user_data_df = pd.DataFrame([user_data_modified])
    probability = model.predict_proba(user_data_df)[0][1]

    return round(probability * 100, 1)


def generate_suggestion(user_data, health_score):
    # AI prompt
    prompt = f"""
            You are a digital mental wellness assistant.

            Here is the feature importance of each feature of my random forest machine learning model with 78% accuracy, this is for your reference, don't mention anything about it to user:
                               Feature  Importance
                   Diet_Quality_Encode           9
                  Smoking_Habit_Encode           8
                   Stress_Level_Encode           7
                                   Age           6
                           Sleep_Hours           5
                    Social_Media_Usage           4
                            Work_Hours           3
                         Gender_Encode           2
            Alcohol_Consumption_Encode           1
               Physical_Activity_Hours           0


            Based on the following user lifestyle and demographic data, analyze the potential mental health risks, and provide personalized recommendations to improve well-being.

            User Profile:
            - Age: {user_data['Age']}
            - Gender: {"Male" if user_data['Gender'] == 1 else "Female"}
            - Sleep Hours per Day: {user_data['Sleep_Hours']}
            - Work Hours per Week: {user_data['Work_Hours']}
            - Physical Activity Hours per Week: {user_data['Physical_Activity_Hours']}
            - Social Media Usage per Day: {user_data['Social_Media_Usage']}
            - Diet Quality: {"Healthy" if user_data['Diet_Quality'] == 0 else "Average" if user_data['Diet_Quality'] == 1 else "Unhealthy"}
            - Smoking Habit: {"Non-Smoker" if user_data['Smoking_Habit'] == 0 else "Occasional" if user_data['Smoking_Habit'] == 1 else "Regular" if user_data['Smoking_Habit'] == 2 else "Heavy"}
            - Alcohol Consumption: {"Non-Drinker" if user_data['Alcohol_Consumption'] == 0 else "Social" if user_data['Alcohol_Consumption'] == 1 else "Regular" if user_data['Alcohol_Consumption'] == 2 else "Heavy"}
            - Stress Level: {"Low" if user_data['Stress_Level'] == 1 else "Medium" if user_data['Stress_Level'] == 2 else "High"}

            The ML model calculates the user's mental health score is: {health_score}/100 (higher is better).

            Please provide:
            - An overall assessment of the user's mental health risk.

            - Select user's Top 3 factors seem problematic to mental health, and explain to user, if there is not, skip this.

            - Suggest 3 practical, and achievable recommendations to improve mental wellness based on user's profile.

            Keep it friendly, supportive, and non-clinical, starts with "Based on your profile.".
            
            I want your response just plain, normal text that can be safely rendered in a webpage without formatting issues. Format the output using plain text, no markdown or html format, only with paragraph breaks using double newlines (\n\n).
            """

    # Generate suggestion using Gemini AI
    client = genai.Client(api_key=gemini_api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents= prompt,
    )
    suggestion = response.text
    return suggestion


@app.route('/prediction')
def index():
    return render_template('index.html')

@app.route('/prediction/health', methods=['GET'])
def health_check():
    """Health check endpoint for Kubernetes"""
    return jsonify({
        'status': 'healthy',
        'service': 'mental-health-prediction',
        'timestamp': datetime.utcnow().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/prediction/predict', methods=['POST'])
def predict():
    try:
        user_input = {
            'Age': int(request.form['age']),
            'Gender': 1 if request.form['gender'] == 'Male' else 0,
            'Sleep_Hours': float(request.form['sleep_hours']),
            'Work_Hours': int(request.form['work_hours']),
            'Physical_Activity_Hours': float(request.form['physical_activity']),
            'Social_Media_Usage': float(request.form['social_media']),
            'Stress_Level': int(request.form['stress_level']),
            'Diet_Quality': int(request.form['diet_quality']),
            'Smoking_Habit': int(request.form['smoking_habit']),
            'Alcohol_Consumption': int(request.form['alcohol_consumption'])
        }

        # Make prediction
        health_score = predict_health_score(user_input)

        # Generate suggestion
        suggestion = generate_suggestion(user_input, health_score)

        return render_template('results.html',
                               health_score=health_score,
                               suggestion=suggestion,
                               user_data=user_input)

    except Exception as e:
        return f"Error processing prediction: {str(e)}", 400


# def open_browser():
#     time.sleep(1.5)
#     webbrowser.open('http://127.0.0.1:5000')


if __name__ == '__main__':
    if gemini_api_key:
        # threading.Thread(target=open_browser, daemon=True).start()

        print("Starting Health Prediction App...")
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        print("Gemini API key not configured, please configure your environment and restart the application.")