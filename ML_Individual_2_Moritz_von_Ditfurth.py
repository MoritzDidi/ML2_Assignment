import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import joblib

file_path = 'bodyPerformance.csv'  
data = pd.read_csv(file_path)

# Exploratory Data Analysis 
st.subheader('Exploratory Data Analysis')
st.write("Here's a glimpse of the dataset:")
st.dataframe(data.head())

st.write("Summary statistics of the dataset:")
st.dataframe(data.describe())

X = data.drop('class', axis=1)
y = data['class']

# Preprocessing for numerical data
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

# Preprocessing for categorical data
categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create the Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=0)

# Create the pipeline with preprocessing and the classifier
pipeline_gb = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', gb_classifier)])

# Train the model
pipeline_gb.fit(X_train, y_train)

# Predict on the test data
y_pred_gb = pipeline_gb.predict(X_test)

# Evaluate the model
classification_results_gb = classification_report(y_test, y_pred_gb)
st.text('Model Evaluation:\n' + classification_results_gb)
st.title('Body Performance Classification')

# Explanation of different classes
st.subheader('Explanation of Classes')
st.write('''
Class A: High body performance
Class B: Above-average body performance
Class C: Average body performance
Class D: Below-average body performance
''')

# Assuming the user inputs all the features in a form
with st.form('prediction_form'):
    age = st.number_input('Age', min_value=1)
    gender = st.selectbox('Gender', ['M', 'F'])
    height_cm = st.number_input('Height (cm)')
    weight_kg = st.number_input('Weight (kg)')
    body_fat_percent = st.number_input('Body Fat (%)')
    diastolic = st.number_input('Diastolic')
    systolic = st.number_input('Systolic')
    grip_force = st.number_input('Grip Force')
    sit_and_bend_forward_cm = st.number_input('Sit and Bend Forward (cm)')
    sit_ups_counts = st.number_input('Sit-ups Counts')
    broad_jump_cm = st.number_input('Broad Jump (cm)')
    submit_button = st.form_submit_button("Predict")

if submit_button:
    # Create a dataframe from the inputs
    features = pd.DataFrame([[age, gender, height_cm, weight_kg, body_fat_percent,
                              diastolic, systolic, grip_force, sit_and_bend_forward_cm,
                              sit_ups_counts, broad_jump_cm]],
                            columns=['age', 'gender', 'height_cm', 'weight_kg', 'body fat_%',
                                     'diastolic', 'systolic', 'gripForce', 'sit and bend forward_cm',
                                     'sit-ups counts', 'broad jump_cm'])
    # Use the model to make predictions
    prediction = pipeline_gb.predict(features)
    
    # Provide an explanation based on the predicted class
    class_explanation = {
        'A': 'High body performance. Keep up the good work!',
        'B': 'Above-average body performance. You are doing well!',
        'C': 'Average body performance. There is room for improvement.',
        'D': 'Below-average body performance. Let’s work on it!'
    }
    
    st.write(f'The predicted class is: {prediction[0]}')
    st.write(class_explanation[prediction[0]])
