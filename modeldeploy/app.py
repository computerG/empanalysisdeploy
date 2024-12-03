import streamlit as st
import pandas as pd
from model import encode_data, load_model, predict_model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Title of the app
st.title("Employee Performance Prediction App")

# List of columns to display (all 28 columns including prediction)
columns_to_display = [
    'EmpNumber', 'Age', 'Gender', 'EducationBackground', 'MaritalStatus',
    'EmpDepartment', 'EmpJobRole', 'BusinessTravelFrequency', 'DistanceFromHome',
    'EmpEducationLevel', 'EmpEnvironmentSatisfaction', 'EmpHourlyRate',
    'EmpJobInvolvement', 'EmpJobLevel', 'EmpJobSatisfaction', 'NumCompaniesWorked',
    'OverTime', 'EmpLastSalaryHikePercent', 'EmpRelationshipSatisfaction',
    'TotalWorkExperienceInYears', 'TrainingTimesLastYear', 'EmpWorkLifeBalance',
    'ExperienceYearsAtThisCompany', 'ExperienceYearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager', 'Attrition', 'Predicted PerformanceRating'
]

# File uploader for CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Form for manual data entry
with st.form(key='manual_data_form'):
    st.subheader("Or Enter Data Manually:")
    
    # Create form fields for all 28 columns
    emp_number = st.text_input("Employee Number")
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"]) 
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    
    # Corrected typo in EducationBackground options list
    education_background = st.selectbox("Education Background", ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources']) 
    st.write('Label for Education Background: 1-Below College, 2-College, 3-Bachelor, 4-Master, 5-Doctor')
    emp_education_level = st.selectbox("Education Level", [1, 2, 3, 4, 5])
    
    emp_department = st.selectbox("Department", ['Sales', 'Development', 'Research & Development', 'Human Resources', 'Finance', 'Data Science'])
    
    # Job Role Options
    emp_job_role = st.selectbox("Employee Job Role", [
        'Sales Executive', 'Manager', 'Developer', 'Sales Representative', 'Human Resources',
        'Senior Developer', 'Data Scientist', 'Senior Manager R&D', 'Laboratory Technician',
        'Manufacturing Director', 'Research Scientist', 'Healthcare Representative',
        'Research Director', 'Manager R&D', 'Finance Manager', 'Technical Architect',
        'Business Analyst', 'Technical Lead', 'Delivery Manager'
    ])
    
    # Job Involvement level
    st.write('Label for Job Involvement: 1-- Low, 2-- Medium, 3-- High, 4-- Very High')
    emp_job_involvement = st.number_input("Job Involvement (1 to 4)", min_value=1, max_value=4, step=1)
    
    # Other fields as before
    emp_job_level = st.number_input("Job Level (1 to 5)", min_value=1, max_value=5, step=1)
    st.write("Emp job satisfaction: 1-- Low, 2-- Medium, 3-- High, 4-- Very High")
    emp_job_satisfaction = st.number_input("Job Satisfaction (1 to 4)", min_value=1, max_value=4, step=1)
    
    business_travel = st.selectbox("Business Travel Frequency", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    distance_from_home = st.number_input("Distance From Home (in km)", min_value=0, step=1)
    emp_environment_satisfaction = st.number_input("Environment Satisfaction (1 to 4)", min_value=1, max_value=4, step=1)
    emp_hourly_rate = st.number_input("Hourly Rate", min_value=0, step=1)
    
    num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, step=1)
    over_time = st.selectbox("Over Time", ["Yes", "No"])
    emp_last_salary_hike_percent = st.number_input("Salary Hike Percent", min_value=0, max_value=100, step=1)
    emp_relationship_satisfaction = st.number_input("Relationship Satisfaction (1 to 4)", min_value=1, max_value=4, step=1)
    
    total_work_experience = st.number_input("Total Work Experience (in years)", min_value=0, step=1)
    training_times_last_year = st.number_input("Training Times Last Year", min_value=0, step=1)
    emp_work_life_balance = st.number_input("Work-Life Balance (1 to 4)", min_value=1, max_value=4, step=1)
    experience_years_at_this_company = st.number_input("Years at This Company", min_value=0, step=1)
    experience_years_in_current_role = st.number_input("Years in Current Role", min_value=0, step=1)
    years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, step=1)
    years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, step=1)
    attrition = st.selectbox("Attrition", ["Yes", "No"])

    # Submit button for the form
    submit_button = st.form_submit_button(label="Submit Data")

# Check if file is uploaded or form data is submitted
if uploaded_file is not None:
    try:
        # Read the uploaded CSV into a pandas DataFrame
        df = pd.read_csv(uploaded_file)

        # Display the first few rows of the uploaded file
        st.write("Data Preview:")
        st.dataframe(df.head())

        # Show the columns in the DataFrame
        st.write("Columns in the dataset:", df.columns)

        # Encode or preprocess the data if needed
        df_encoded = encode_data(df)

        if 'PerformanceRating' in df_encoded.columns:
            X = df_encoded.drop(columns=["PerformanceRating"], axis=1)

            # Load the pre-trained model
            model = load_model()

            # Perform prediction on the uploaded data
            logging.info("Model and encoder loaded successfully.")
            predictions = predict_model(model, X)

            # Add predictions to the data
            X['Predicted PerformanceRating'] = predictions

            # Display the predictions alongside all 28 columns
            st.write("Predictions:")
            st.write("1-Low (L), 2-Good (0), 3-Excellent (1), 4-Outstanding (2)")
            st.dataframe(X[columns_to_display])
          
    except Exception as e:
        logging.error(f"Error processing the file: {e}")
        st.error(f"An error occurred while processing the file: {e}")

elif submit_button:
    try:
        # Create a dataframe from manual input (28 fields)
        manual_data = {
            "EmpNumber": [emp_number],
            "Age": [age],
            "Gender": [gender],
            "EducationBackground": [education_background],
            "MaritalStatus": [marital_status],
            "EmpDepartment": [emp_department],
            "EmpJobRole": [emp_job_role],
            "BusinessTravelFrequency": [business_travel],
            "DistanceFromHome": [distance_from_home],
            "EmpEducationLevel": [emp_education_level],
            "EmpEnvironmentSatisfaction": [emp_environment_satisfaction],
            "EmpHourlyRate": [emp_hourly_rate],
            "EmpJobInvolvement": [emp_job_involvement],
            "EmpJobLevel": [emp_job_level],
            "EmpJobSatisfaction": [emp_job_satisfaction],
            "NumCompaniesWorked": [num_companies_worked],
            "OverTime": [over_time],
            "EmpLastSalaryHikePercent": [emp_last_salary_hike_percent],
            "EmpRelationshipSatisfaction": [emp_relationship_satisfaction],
            "TotalWorkExperienceInYears": [total_work_experience],
            "TrainingTimesLastYear": [training_times_last_year],
            "EmpWorkLifeBalance": [emp_work_life_balance],
            "ExperienceYearsAtThisCompany": [experience_years_at_this_company],
            "ExperienceYearsInCurrentRole": [experience_years_in_current_role],
            "YearsSinceLastPromotion": [years_since_last_promotion],
            "YearsWithCurrManager": [years_with_curr_manager],
            "Attrition": [attrition]
        }
        
        manual_df = pd.DataFrame(manual_data)
        
        # Encode the manually entered data
        X = encode_data(manual_df)

        # Load the pre-trained model
        model = load_model()

        # Perform prediction on the manually entered data
        logging.info("Model and encoder loaded successfully.")
        predictions = predict_model(model, X)

        # Display the predictions alongside all 28 columns
        X['Predicted PerformanceRating'] = predictions
        st.write("Predictions:")
        st.write("1-Low (L), 2-Good (0), 3-Excellent (1), 4-Outstanding (2)")

        # Display the input data and predicted rating
        st.subheader("Submitted Data and Prediction")
        st.dataframe(X[columns_to_display])

    except Exception as e:
        logging.error(f"Error processing manual data: {e}")
        st.error(f"An error occurred while processing the manual data: {e}")
else:
    st.write("Please upload a CSV file or fill out the form above to proceed.")
