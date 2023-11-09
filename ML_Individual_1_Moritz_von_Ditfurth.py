import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Title of the app
st.title('Diabetes Prediction App')

# App description
st.markdown("""
This interactive web app predicts the likelihood of diabetes using a machine learning model. Navigate through the app to visualize data, understand the distribution of variables, and ultimately predict the likelihood of diabetes.
""")

# Load the dataset
@st.cache(allow_output_mutation=True)
def load_data():
    file_path = 'diabetes.csv'
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Display dataset loading
st.markdown("## Load Dataset")
if st.button('Load Diabetes Dataset'):
    diabetes_data = load_data()
    if diabetes_data is not None:
        st.write("Dataset loaded successfully!")
        st.write(diabetes_data.head())
        st.session_state['diabetes_data'] = diabetes_data  # Store in session state
    else:
        st.write("Failed to load the dataset. Please check the file path and format.")

# Optional EDA Section
if 'diabetes_data' in st.session_state:
    st.markdown("## Optional Exploratory Data Analysis")
    if st.checkbox('Show Histograms'):
        for column in st.session_state.diabetes_data.columns[:-1]:  # Exclude 'Outcome'
            st.subheader(f'Histogram for {column}')
            fig, ax = plt.subplots()
            sns.histplot(st.session_state.diabetes_data[column], kde=True, bins=30, ax=ax)
            st.pyplot(fig)

    if st.checkbox('Show Boxplots'):
        for column in st.session_state.diabetes_data.columns[:-1]:  # Exclude 'Outcome'
            st.subheader(f'Boxplot for {column}')
            fig, ax = plt.subplots()
            sns.boxplot(x=st.session_state.diabetes_data[column], ax=ax)
            st.pyplot(fig)

    if st.checkbox('Show Correlation Matrix'):
        st.subheader('Correlation Matrix')
        fig, ax = plt.subplots()
        sns.heatmap(st.session_state.diabetes_data.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# Data Preprocessing
st.markdown("## Data Preprocessing")
if st.button("Preprocess Data"):
    if 'diabetes_data' in st.session_state:
        preprocessed_data = st.session_state.diabetes_data.copy()
        preprocessed_data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = preprocessed_data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
        preprocessed_data.fillna(preprocessed_data.median(), inplace=True)

        scaler = StandardScaler()
        preprocessed_data.iloc[:, :-1] = scaler.fit_transform(preprocessed_data.iloc[:, :-1])

        st.session_state['preprocessed_data'] = preprocessed_data
        st.write("Data preprocessing completed.")
        st.dataframe(preprocessed_data.head())
    else:
        st.error("Dataset not loaded. Please load the dataset first.")

# Model Building and Training
st.markdown("## Model Building and Training")
if st.button("Build and Train Model"):
    if 'preprocessed_data' in st.session_state:
        X = st.session_state['preprocessed_data'].drop(columns='Outcome')
        y = st.session_state['preprocessed_data']['Outcome']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_classifier = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

        try:
            grid_search.fit(X_train, y_train)
            st.session_state['model'] = grid_search
            st.success("Model trained successfully!")
            st.write("Best Model Parameters:")
            st.json(grid_search.best_params_)
        except Exception as e:
            st.error(f"Error during model training: {e}")

# Model Evaluation
st.markdown("## Model Evaluation")
if st.button("Evaluate Model"):
    if 'model' in st.session_state and 'preprocessed_data' in st.session_state:
        X_test = st.session_state['preprocessed_data'].drop(columns='Outcome')
        y_test = st.session_state['preprocessed_data']['Outcome']

        predictions = st.session_state['model'].predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)

        st.subheader("Model Performance")
        st.write(f'Accuracy: {accuracy}')
        st.text(report)
    else:
        st.error("Model not trained. Please train the model first.")
