#importing the required libraries
import streamlit as st
from data.data_dictionary import *
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
#importing the prediction pipeline
pipe=pickle.load(open('prediction_pipeline.pkl','rb'))
#title of webpage
st.title('Student Dropout Predictor')
st.subheader("Enter student details")
#creating a form to take input from user of the features which are required for the prediction
#for the categorical columns in the data I have taken string input from user and then used data_dictionary to return it's corresponding numeric value
with st.form('details'):
    col1,col2=st.columns(2)
    course_key=st.selectbox("Courses",Course.keys())
    marital_status_key=col1.selectbox("Marital status",Marital_status.keys())
    application_mode_key=col2.selectbox("Application mode",Application_mode.keys())
    mother_occupation_key=col2.selectbox("Mother's Occupation",Mother_occupation.keys())
    debtor_key=col1.selectbox("Debtor",Debtor.keys())
    tution_fees_key=col2.selectbox("Tution fees up to Date",Tuition_fees_up_to_date.keys())
    scholarship_key=col1.selectbox("Scholarship holder",Scholarship_holder.keys())
    international_key=col2.selectbox("International student",International.keys())
    age=col1.number_input("Age at enrollment",step=1)
    cu1credited=col2.number_input("Curricular units 1st sem (credited)",step=1)
    cu1enrolled=col1.number_input("Curricular units 1st sem (enrolled)",step=1)
    cu1evaluations=col2.number_input("Curricular units 1st sem (evaluations)",step=1)
    cu1approved=col1.number_input("Curricular units 1st sem (approved)",step=1)
    cu2credited=col2.number_input("Curricular units 2nd sem (credited)",step=1)
    cu2enrolled=col1.number_input("Curricular units 2nd sem (enrolled)	",step=1)
    cu2evaluations=col2.number_input("Curricular units 2nd sem (evaluations)",step=1)
    cu2approved=col1.number_input("Curricular units 2nd sem (approved)",step=1)
    cu2grade=col2.number_input("Curricular units 2nd sem (grade)",step=1)
    cu2without_evaluation=col1.number_input("Curricular units 2nd sem (without evaluations)",step=1)

    submit=st.form_submit_button("Predict")

#storing the user's input into a query list
query=[
    Marital_status[marital_status_key],
    Application_mode[application_mode_key],
    Course[course_key],
    Mother_occupation[mother_occupation_key],
    Debtor[debtor_key],
    Tuition_fees_up_to_date[tution_fees_key],
    Scholarship_holder[scholarship_key],
    age,
    International[international_key],
    cu1credited,
    cu1enrolled,
    cu1evaluations,
    cu1approved,
    cu2credited,
    cu2enrolled,
    cu2evaluations,
    cu2approved,
    cu2grade,
    cu2without_evaluation
    ]

if submit:
    input_array=np.array(query).reshape(1,19)
    st.subheader("The probability of this student to dropout is :")
    st.title(round(pipe.predict_proba(input_array)[0][1],2))


