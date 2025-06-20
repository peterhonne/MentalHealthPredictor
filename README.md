# MentalHealthPredictor

A Flask web application that predicts health scores based on lifestyle factors and provides AI-powered personalized health recommendations using Google's Gemini API.

## Features

- **Intuitive Interface**: HTML-based UI for easy input of lifestyle factors (age, gender, sleep hours, stress level, etc.).
- **Predictive Modeling**: Random Forest model (78% accuracy) predicts mental health risks based on 10 key features.
- **Personalized Recommendations**: Gemini API generates tailored suggestions for sleep, stress, and lifestyle improvements.
- **Actionable Insights**: Outputs a health score and evidence-based guidance to support proactive mental health management.

## Analytics Foundation

- **Dataset**: 50,000 records with demographic, clinical, and lifestyle factors (e.g., sleep, diet, smoking).
- **Key Insights**:
  - Higher stress and fewer sleep hours strongly correlate with mental health risks.
  - Females and younger individuals show elevated risk, often linked to social media.
  - Heavy smoking/drinking increases mental illness severity.
- **Modeling**:
  - Random Forest (78% accuracy) outperformed Logistic Regression (74.6% with SMOTE) and SVM (73.6%).
