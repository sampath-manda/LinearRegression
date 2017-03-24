import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")


bmi_life_model = linear_model.LinearRegression()
bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])


laos_life_expectancy = bmi_life_model.predict(21.07931)

print(laos_life_expectancy)