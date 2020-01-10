from PIL import Image

import pandas as pd

column = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction",
          "Age", "Outcome"]
data = pd.read_csv('pima-indians-diabetes.data.csv', names=column)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
print(data.head(10))

bayes_png = Image.open('images/bayes.PNG')
bayes_png.show()

outcome_1 = Image.open('images/outcome1.PNG')
outcome_1.show()
outcome_0 = Image.open('images/outcome0.PNG')
outcome_0.show()

# 1.Calculate Priors
# 2.Calculate Likelihood
# 3.Calculate Marginal Probability
# 4.Apply Bayes Classifier To New Data Point
# 5.Understand what has just happen

# ############## Calculating Priors ################# #

# Number of patients of outcome 1
n_outcome_1 = data['Outcome'][data['Outcome'] == 1].count()
# print(n_outcome_1)
# Number of patients of outcome 1
n_outcome_0 = data['Outcome'][data['Outcome'] == 0].count()
# print(n_outcome_0)
# Total number of people
total_people = data['Outcome'].count()
# print(total_people)

# Probability of outcome1
P_outcome_1 = n_outcome_1/total_people
# print(P_outcome_1)
# Probability of outcome0
P_outcome_0 = n_outcome_0/total_people
# print(P_outcome_0)

# ############## Calculating Likelihood ################# #

bay_1 = Image.open('images/bay1.PNG')
bay_1.show()

# Now first calculate the means of the data according to outcome
# Group the data by gender and calculate the mean of each feature

data_mean = data.groupby('Outcome').mean()
print(data_mean.head())


# Now first calculate the variance of the data according to outcome
# Group the data by gender and calculate the variance of each feature

data_variance = data.groupby('Outcome').var()
print(data_variance.head())



