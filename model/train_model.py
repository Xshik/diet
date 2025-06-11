import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
data = pd.read_csv('data/diet_dataset.csv')  # Ensure you have your dataset at the correct location

# Initialize label encoders
label_encoder_gender = LabelEncoder()
label_encoder_activity = LabelEncoder()
label_encoder_health = LabelEncoder()

# Encode the categorical features
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
data['Activity Level'] = label_encoder_activity.fit_transform(data['Activity Level'])
data['Health Conditions'] = label_encoder_health.fit_transform(data['Health Conditions'])

# Define features and target variable
X = data[['Age', 'Gender', 'BMI', 'Activity Level', 'Health Conditions']]
y = data[['Food Allergies', 'Dietary Preference', 'Breakfast', 'Lunch', 'Dinner', 'Snacks', 'Recommended_Meal_Plan']]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
pickle.dump(model, open('model/meal_recommendation_model.pkl', 'wb'))

# Save the label encoders
pickle.dump(label_encoder_gender, open('model/label_encoder_gender.pkl', 'wb'))
pickle.dump(label_encoder_activity, open('model/label_encoder_activity.pkl', 'wb'))
pickle.dump(label_encoder_health, open('model/label_encoder_health.pkl', 'wb'))

print("Model and LabelEncoders saved successfully!")
