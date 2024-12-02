import streamlit as st
import pandas as pd
from model import encode_data, load_model, predict_model
import logging

# Title of the app
st.title("Employee Performance Prediction App")

# File uploader for CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV into a pandas DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the first few rows of the uploaded file
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Show the columns in the DataFrame
    st.write("Columns in the dataset:", df.columns)

    # Process and encode data
    
        # Encode or preprocess the data if needed
    df_encoded = encode_data(df)
    if 'PerformanceRating' in df_encoded.columns:
        X = df_encoded.drop(columns=["PerformanceRating"],axis=1)
        original_df=df_encoded.drop(columns=["PerformanceRating"],axis=1)
        # Load the pre-trained model
    model = load_model()

        # Perform prediction on the uploaded data
    logging.info("Model and encoder loaded successfully.")
    predictions = predict_model(model, X)

        # Display the predictions alongside the original data
    X['Predicted PerformanceRating'] = predictions
    st.write("Predictions:")
    st.write("1-Low (L),2-Good (0), 3-Excellent(1),4-Outstanding (2)")
    st.dataframe(X[['EmpNumber', 'Age', 'Gender', 'EducationBackground', 'MaritalStatus',
       'EmpDepartment', 'EmpJobRole', 'BusinessTravelFrequency',
       'DistanceFromHome', 'EmpEducationLevel', 'EmpEnvironmentSatisfaction',
       'EmpHourlyRate', 'EmpJobInvolvement', 'EmpJobLevel',
       'EmpJobSatisfaction', 'NumCompaniesWorked', 'OverTime',
       'EmpLastSalaryHikePercent', 'EmpRelationshipSatisfaction',
       'TotalWorkExperienceInYears', 'TrainingTimesLastYear',
       'EmpWorkLifeBalance', 'ExperienceYearsAtThisCompany',
       'ExperienceYearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager', 'Attrition','Predicted PerformanceRating']])

else : 
    st.write("Upload file")
