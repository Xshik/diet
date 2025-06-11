from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and label encoders
model = pickle.load(open('model/meal_recommendation_model.pkl', 'rb'))
label_encoder_gender = pickle.load(open('model/label_encoder_gender.pkl', 'rb'))
label_encoder_activity_level = pickle.load(open('model/label_encoder_activity.pkl', 'rb'))
label_encoder_health_conditions = pickle.load(open('model/label_encoder_health.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        # Get user input from the form
        age = float(request.form['age'])
        gender = request.form['gender']
        bmi = float(request.form['bmi'])
        activity_level = request.form['activity_level']
        health_conditions = request.form['health_conditions']
        
        # Encode the categorical features
        gender_encoded = label_encoder_gender.transform([gender])[0]
        activity_level_encoded = label_encoder_activity_level.transform([activity_level])[0]
        health_conditions_encoded = label_encoder_health_conditions.transform([health_conditions])[0]

        # Prepare the input for prediction
        input_data = np.array([[age, gender_encoded, bmi, activity_level_encoded, health_conditions_encoded]])

        # Predict the meal categories
        prediction = model.predict(input_data)

        # Assuming the model returns meal categories as an array
        food_allergies, dietary_preference, breakfast, lunch, dinner, snacks, recommendation = prediction[0]

        # Render the template with individual meal categories
        return render_template('result.html', 
                               breakfast=breakfast, 
                               lunch=lunch, 
                               dinner=dinner, 
                               snacks=snacks, 
                               recommendation=recommendation)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
