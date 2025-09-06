
# Class/ASD - Classified result as 0 or 1. Here 0 represents No and 1 represents Yes. This is the target column, and during submission submit the values as 0 or 1 only.

import random
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
model = pickle.load(open("C:/Users/Mahdi/OneDrive/Desktop/Projects/Autism Disease Prediction/model/best_model.pkl", 'rb'))

def ASD(input):
    input_array = np.asarray(input)
    
    # Reshaping the inputs
    reshaped_inputs = input_array.reshape(1, -1)

    # Making prediction
    prediction = model.predict(reshaped_inputs)

    if prediction[0] == 0:
        return "No"
    else:
        return "Yes"


# A1_Score,A2_Score,A3_Score,A4_Score,A5_Score,A6_Score,A7_Score,A8_Score,A9_Score,A10_Score,age,gender,ethnicity,jaundice,austim,contry_of_res,used_app_before,result,relation

def main():
    st.title("Autism Spectrum Disorder Prediction")
    st.header("Social Responsiveness Scale")

    # Dropdown for ethnicity
    ethnicityDict = {'White-European' : 
    'White-European', 'Middle Eastern' : 'Middle Eastern', 'Pasifika' : 'Pasifika', 'Black' : 'Black', 'Hispanic' : 'Hispanic', 'Asian' : 'Asian', 'Turkish' : 'Turkish', 'South Asian' : 'South Asian', 'Latino' : 'Latino', 'Others' : 'Others'}

    ethnicity_val = list(ethnicityDict.values())
    
    encoded_ethnicity = label_encoder.fit(ethnicity_val).transform(ethnicity_val)

    dict_ethnicity = dict(zip(ethnicityDict.keys(), encoded_ethnicity))
    
    ethnicity_dropdown = st.selectbox("Select Ethnicity", dict_ethnicity)
    ethnicity = dict_ethnicity[ethnicity_dropdown]


    # 2nd Input
    country_of_res = ['Austria', 'India', 'United States', 'South Africa', 'Jordan','United Kingdom', 'Brazil', 'New Zealand', 'Canada', 'Kazakhstan','United Arab Emirates', 'Australia', 'Ukraine', 'Iraq', 'France','Malaysia', 'Vietnam', 'Egypt', 'Netherlands', 'Afghanistan','Oman', 'Italy', 'Bahamas', 'Saudi Arabia', 'Ireland', 'Aruba','Sri Lanka', 'Russia', 'Bolivia', 'Azerbaijan', 'Armenia','Serbia', 'Ethiopia', 'Sweden', 'Iceland', 'China', 'Angola','Germany', 'Spain', 'Tonga', 'Pakistan', 'Iran', 'Argentina','Japan', 'Mexico', 'Nicaragua', 'Sierra Leone', 'Czech Republic','Niger', 'Romania', 'Cyprus', 'Belgium', 'Burundi', 'Bangladesh']

    country_of_res_encoded = label_encoder.fit(country_of_res).transform(country_of_res)

    country_dict = dict(zip(country_of_res, country_of_res_encoded))

    residence_dropdown = st.selectbox("Country", country_of_res)

    country = country_dict[residence_dropdown]

    # Input of gender
    genderDict =  {"Male" : 1, "Female" : 0}
    gender_input = st.radio("Gender", genderDict)
    gender = [genderDict[gender_input]]
    
    # Input of Jaundice and Autism
    other_params = ["Jaundice", "Autism"]
    ans_others_param = {"Yes" : 1, "No" : 0}

    jaund_aut = []
    for i in other_params:
        ans = st.radio(i, ans_others_param)
        jaund_aut.append(ans_others_param[ans])

    used_app_before = [random.randint(0, 1)]

    # age input
    age = [st.slider("Age(in month) of kid", 1, 50, 24)]

    # result input
    result = [st.slider("Enter result", 0.0, 50.0, 23.1)]

    # Questions (A1_Score - A10_score)
    questions_list = []
    questionnaires = [
    "Avoids eye contact with others.",
    "Has trouble understanding other people’s feelings.",
    "Prefers to be alone rather than with others.",
    "Has unusually intense interests.",
    "Has difficulty understanding the back-and-forth flow of normal conversation..",
    "Responds inappropriately to other's attempts at social interaction (e.g., doesn’t answer or walks away)." ,
    "Shows little interest in playing with other children.",
    "Overreacts to minor changes in routine or environment.",
    "Focuses intensely on particular topics or interests, often at the exclusion of everything else.",
    "Uses unusual tone of voice, pitch, or rhythm when speaking."
    ]    
    ans = {"Yes" : 1, "No" : 0}
    for questions in questionnaires:
        answer = st.radio(questions,ans)
        questions_list.append(ans[answer])  
    
    relation = [random.randint(0, 1)]

    # All inputs in one list
    inputs = np.concatenate([questions_list,age,gender,[ethnicity],jaund_aut,[country],used_app_before,result,relation])

    # Button
    if(st.button("Predict")):
        result = ASD(inputs)
        st.success(result)


if __name__ == "__main__":
    main()