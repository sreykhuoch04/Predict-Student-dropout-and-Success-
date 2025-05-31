import streamlit as st
import joblib
import pandas as pd

# Load trained XGBoost model
model = joblib.load('tuned_xgb_model.pkl')

st.set_page_config(page_title="Student Dropout Prediction", layout="centered")
st.title("🎓 Student Dropout Prediction App")
st.markdown("Enter the student's details to predict whether they are at risk of dropping out.")

# --- User Inputs ---
marital_status = st.number_input("Marital Status (e.g., 1 = Single, 2 = Married)", value=1)
application_mode = st.number_input("Application Mode", value=1)
application_order = st.number_input("Application Order", min_value=1, value=1)
course = st.number_input("Course ID", value=1)
attendance = st.radio("Attendance Type", [1, 0], format_func=lambda x: "Daytime" if x == 1 else "Evening")
previous_qualification = st.number_input("Previous Qualification", value=1)
mother_qualification = st.number_input("Mother's Qualification", value=1)
father_qualification = st.number_input("Father's Qualification", value=1)
mother_occupation = st.number_input("Mother's Occupation", value=0)
father_occupation = st.number_input("Father's Occupation", value=0)
displaced = st.radio("Displaced", [0, 1])
debtor = st.radio("Debtor", [0, 1])
tuition_fees_up_to_date = st.radio("Tuition Fees Up-To-Date", [0, 1])
gender = st.radio("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
scholarship_holder = st.radio("Scholarship Holder", [0, 1])
age = st.number_input("Age", min_value=15, max_value=60, value=20)
avg_enrolled = st.number_input("Average Enrolled ", value=0.0)
avg_approved = st.number_input("Average Approved ", value=0.0)
avg_grade = st.number_input("Average Grade", value=0.0)

# --- Prepare input data ---
input_data = pd.DataFrame([[
    marital_status, application_mode, application_order, course,
    attendance, previous_qualification, mother_qualification, father_qualification,
    mother_occupation, father_occupation, displaced, debtor,
    tuition_fees_up_to_date, gender, scholarship_holder, age,
    avg_enrolled, avg_approved, avg_grade
]], columns=[
    'Marital_status', 'Application_mode', 'Application_order', 'Course',
    'Daytime/evening_attendance', 'Previous_qualification',
    'Mother_qualification', 'Father_qualification', 'Mother_occupation',
    'Father_occupation', 'Displaced', 'Debtor', 'Tuition_fees_up_to_date',
    'Gender', 'Scholarship_holder', 'Age', 'avg_enrolled',
    'avg_approved', 'avg_grade'
])

# --- Predict ---
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    result = "🎓 Graduated" if prediction == 1 else "⚠️ Dropped Out"
    st.subheader("📊 Prediction Result")
    st.success(f"The student **{result}**.")

    st.markdown(f"""
    **Prediction Confidence:**
    - Dropout Probability: `{probabilities[0]:.2%}`
    - Graduation Probability: `{probabilities[1]:.2%}`
    """)

