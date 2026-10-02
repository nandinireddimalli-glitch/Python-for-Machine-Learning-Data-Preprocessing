"""
Machine Learning Data Preprocessing Project

This project demonstrates:
- Data loading
- Exploratory Data Analysis
- Missing value handling
- Feature selection
- Categorical encoding
- Feature scaling
- Data visualization
- Saving processed data
"""







import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print("Libraries imported successfully.")

data={
    "Student_ID":[101,102,103,104,105,106,107,108],
    "Age":[20,21,np.nan,22,20,23,21,22],
    "Gender":["Female","Male","Female","Male",np.nan,"Female","Male","Female"],
    "Study_Hours":[3,5,2,np.nan,6,4,5,3],
    "Attendance":[85,90,70,75,95,np.nan,88,80],
    "Previous_Score": [75, 80, 65, 70, 90, 85, 78, 72],
    "Passed": ["Yes", "Yes", "No", "No", "Yes", "Yes", "Yes", "No"]

}
df=pd.DataFrame(data)
print("\n Student Dataset:")
print(df)
print("\n Dataset Shape:")
print(df.shape)
print("\n Dataset information:")
print(df.info())
print("\n Statistical summary:")
print(df.describe())
print("\n Missing values:")
print(df.isnull().sum())
print("\n 1.Dataset Shape:")
print(df.shape)
print("\n 2.First 5 Rows:")
print(df.head())
print("\n 3.Column Names:")
print(df.columns)
print("\n4. Data Types:")
print(df.dtypes)
print("\n5. Missing Values:")
print(df.isnull().sum())
print("\n6. Duplicate Rows:")
print(df.duplicated().sum())
df = df.drop_duplicates()
print("\n7. Statistical Summary:")
print(df.describe())
df["Age"]=df["Age"].fillna(df["Age"].median())
print("\nAge after filling missing value:")
print(df["Age"])
df["Study_Hours"]=df["Study_Hours"].fillna(df["Study_Hours"].median())
df["Attendance"] = df["Attendance"].fillna(df["Attendance"].median())
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
X=df.drop(columns=["Student_ID","Passed"])
Y=df["Passed"]
print("\nFeatures (X):")
print(X)

print("\nTarget (y):")
print(Y)
X=pd.get_dummies(X,columns=["Gender"],dtype=int)
print("\nFeatures after encoding:")
print(X)
Y = Y.map({"No": 0, "Yes": 1})

print("\nEncoded Target:")
print(Y)
print("\nFinal Feature Data Types:")
print(X.dtypes)
from sklearn.preprocessing import MinMaxScaler
numerical_columns=[
    "Age",
    "Study_Hours",
    "Attendance",
    "Previous_Score"
]
scaler=MinMaxScaler()
X[numerical_columns]=scaler.fit_transform(X[numerical_columns])
print("\nFeatures after Min-Max Scaling:")
print(X)
plt.figure(figsize=(8, 5))

plt.hist(df["Attendance"], bins=5)

plt.title("Distribution of Student Attendance")
plt.xlabel("Attendance")
plt.ylabel("Number of Students")

plt.show()
plt.figure(figsize=(8, 5))

plt.scatter(
    df["Study_Hours"],
    df["Previous_Score"]
)

plt.title("Study Hours vs Previous Score")
plt.xlabel("Study Hours")
plt.ylabel("Previous Score")

plt.show()


processed_df = X.copy()

processed_df["Passed"] = Y

print("\nFinal Processed Dataset:")
print(processed_df)
processed_df.to_csv(
    "cleaned_student_dataset.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")