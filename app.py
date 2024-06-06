import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

st.title('Salary Prediction System')

#reading dataset
df = pd.DataFrame(pd.read_csv("Salary Data.csv"))
df = df.dropna()
# df1 = df['Education Level'].value_counts()
# st.bar_chart(df1)

#data cleaning
gender = pd.get_dummies(df['Gender'], drop_first = True)
auto = pd.concat([df, gender], axis = 1)
education = pd.get_dummies(auto['Education Level'], drop_first = False)
auto = pd.concat([auto, education], axis = 1)
auto = auto.drop(['Gender','Education Level'],axis = 1)
auto = auto.drop(['Male'],axis = 1)

#Start ml
# We specify this so that the train and test data set always have the same rows, respectively
df_train, df_test = train_test_split(auto, train_size = 0.85, test_size = 0.15, random_state = 1)
#train ml
X_train = df_train[['Age','Years of Experience',"Bachelor's",
                    "Master's","PhD"]]
y_train = df_train['Salary']
poly_reg = PolynomialFeatures(degree=7)
X_poly = poly_reg.fit_transform(X_train)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y_train)

#taking input from user
age = st.number_input("Age", 20, 60)
exp = st.number_input("Experience", 0, 40)
edu = st.radio("Education", ["Bachelor's", "Master's","PhD"])
st.write(age,exp,edu)
if edu == "Bachelor's":
	e1 = 1
else:
	e1 = 0
if edu == "Master's":
	e2 = 1
else:
	e2 = 0

if edu == "PhD":
	e3 = 1
else:
	e3 = 0

if st.button("submit"):

	data = {'Age':[age],
	        'Years of Experience':[exp],
	        "Bachelor's":[e1],
	        "Master's":[e2],
	         "PhD":[e3]}

	df = pd.DataFrame(data)
	df
	predicted_salary = pol_reg.predict(poly_reg.fit_transform(df))
	st.write("Predicted Salary",predicted_salary)
# print("Predicted Salary using Polynomial regression",pol_reg.predict(poly_reg.fit_transform(data)))