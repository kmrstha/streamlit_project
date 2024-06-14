import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title('Insurance Prediction System')

# Reading dataset
df = pd.read_csv("insurance.csv")
df = df.dropna()

# Data cleaning
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)
df['smoker'] = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)
region_dummies = pd.get_dummies(df['region'], drop_first=False)

auto = pd.concat([df, region_dummies], axis=1)
auto = auto.drop(['region'], axis=1)

# Start ML
df_train, df_test = train_test_split(auto, train_size=0.85, test_size=0.15, random_state=1)

# Train ML
features = ['age', 'bmi', 'children', 'sex', 'smoker', 'northeast', 'northwest', 'southeast', 'southwest']
X_train = df_train[features]
y_train = df_train['expenses']
lr = LinearRegression()
lr_model = lr.fit(X_train, y_train)

# Taking input from user
age = st.number_input("Age", 18, 70)
bmi = st.number_input("BMI", 0, 40)
children = st.number_input("Children", 0, 5)
gender = st.radio("Gender", ["Male", "Female"])
smoker = st.radio("Smoker", ["Yes", "No"])
region = st.radio("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

sex = 1 if gender == "Male" else 0
smoker_yes = 1 if smoker == "Yes" else 0
northeast = 1 if region == "Northeast" else 0
northwest = 1 if region == "Northwest" else 0
southeast = 1 if region == "Southeast" else 0
southwest = 1 if region == "Southwest" else 0

if st.button("Submit"):
    data = {'age': [age], 'bmi': [bmi], 'children': [children], 'sex': [sex], 'smoker': [smoker_yes], 
            'northeast': [northeast], 'northwest': [northwest], 'southeast': [southeast], 'southwest': [southwest]}
    input_df = pd.DataFrame(data)
    predicted_expenses = lr_model.predict(input_df)
    st.write("Predicted expenses: $", predicted_expenses[0])
