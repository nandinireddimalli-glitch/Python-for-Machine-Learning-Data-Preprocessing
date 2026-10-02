# Machine Learning Data Preprocessing Using Python

## 1. Project Overview

This project demonstrates the fundamental steps involved in preparing data for Machine Learning using Python.

The project focuses on loading, exploring, cleaning, transforming, and preparing a sample student performance dataset for Machine Learning.

The main technologies used are:

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## 2. Objectives

The objectives of this project are:

1. Understand the fundamentals of Machine Learning data preprocessing using Python.
2. Learn how to load and inspect a dataset.
3. Identify and handle missing values.
4. Check for duplicate records.
5. Select relevant features.
6. Encode categorical variables into numerical values.
7. Normalize numerical features.
8. Perform Exploratory Data Analysis (EDA).
9. Visualize data using graphs.
10. Save the cleaned and processed dataset.

---

## 3. Dataset Description

A sample student performance dataset was created for this project.

The dataset contains information about students, including:

| Column         | Description                | Data Type          |
| -------------- | -------------------------- | ------------------ |
| Student_ID     | Unique student identifier  | Integer            |
| Age            | Age of the student         | Numerical          |
| Gender         | Gender of the student      | Categorical        |
| Study_Hours    | Daily study hours          | Numerical          |
| Attendance     | Attendance percentage      | Numerical          |
| Previous_Score | Previous examination score | Numerical          |
| Passed         | Whether the student passed | Categorical/Target |

The dataset contains 8 records and 7 columns.

---

# 4. Libraries Used

## NumPy

NumPy was used to represent missing numerical values using `np.nan` and for numerical operations.

```python
import numpy as np
```

## Pandas

Pandas was used for creating and manipulating the dataset.

```python
import pandas as pd
```

## Matplotlib

Matplotlib was used to create data visualizations.

```python
import matplotlib.pyplot as plt
```

## Seaborn

Seaborn was imported for statistical data visualization.

```python
import seaborn as sns
```

## Scikit-learn

Scikit-learn was used for feature scaling.

```python
from sklearn.preprocessing import MinMaxScaler
```

---

# 5. Data Loading and Creation

The sample dataset was created using a Python dictionary and converted into a Pandas DataFrame.

```python
data = {
    "Student_ID": [101, 102, 103, 104, 105, 106, 107, 108],
    "Age": [20, 21, np.nan, 22, 20, 23, 21, 22],
    "Gender": ["Female", "Male", "Female", "Male",
               np.nan, "Female", "Male", "Female"],
    "Study_Hours": [3, 5, 2, np.nan, 6, 4, 5, 3],
    "Attendance": [85, 90, 70, 75, 95, np.nan, 88, 80],
    "Previous_Score": [75, 80, 65, 70, 90, 85, 78, 72],
    "Passed": ["Yes", "Yes", "No", "No",
               "Yes", "Yes", "Yes", "No"]
}

df = pd.DataFrame(data)
```

---

# 6. Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the structure and quality of the dataset.

## 6.1 Dataset Shape

```python
df.shape
```

Result:

```text
(8, 7)
```

This means the dataset contains:

* 8 rows
* 7 columns

---

## 6.2 Viewing the Dataset

The `head()` function was used to display the first five records.

```python
df.head()
```

This helped inspect the dataset before preprocessing.

---

## 6.3 Checking Data Types

The `dtypes` function was used to identify numerical and categorical columns.

```python
df.dtypes
```

Numerical columns include:

* Age
* Study_Hours
* Attendance
* Previous_Score

Categorical columns include:

* Gender
* Passed

---

# 7. Handling Missing Values

Missing values were identified using:

```python
df.isnull().sum()
```

The dataset initially contained missing values in:

* Age
* Gender
* Study_Hours
* Attendance

## 7.1 Numerical Missing Values

Median imputation was used for the numerical columns.

```python
df["Age"] = df["Age"].fillna(df["Age"].median())

df["Study_Hours"] = df["Study_Hours"].fillna(
    df["Study_Hours"].median()
)

df["Attendance"] = df["Attendance"].fillna(
    df["Attendance"].median()
)
```

### Why median?

The median represents the middle value of a dataset and is less affected by unusually high or low values than the mean.

---

## 7.2 Categorical Missing Values

The missing value in the `Gender` column was replaced using the mode.

```python
df["Gender"] = df["Gender"].fillna(
    df["Gender"].mode()[0]
)
```

### Why mode?

Mode represents the most frequently occurring category and is suitable for categorical data.

---

# 8. Checking Duplicate Records

Duplicate rows were checked using:

```python
df.duplicated().sum()
```

The dataset contained no duplicate records.

Therefore, no duplicate rows needed to be removed.

---

# 9. Feature Selection

The `Student_ID` column was removed because it is only an identifier and does not provide useful information for predicting student performance.

The `Passed` column was separated as the target variable.

```python
X = df.drop(columns=["Student_ID", "Passed"])

y = df["Passed"]
```

Therefore:

### Input Features

* Age
* Gender
* Study_Hours
* Attendance
* Previous_Score

### Target

* Passed

---

# 10. Categorical Encoding

Machine Learning algorithms generally require numerical input.

The `Gender` column contained categorical values:

```text
Female
Male
```

One-hot encoding was applied:

```python
X = pd.get_dummies(
    X,
    columns=["Gender"],
    dtype=int
)
```

This created:

```text
Gender_Female
Gender_Male
```

For example:

```text
Female → Gender_Female = 1, Gender_Male = 0
Male   → Gender_Female = 0, Gender_Male = 1
```

---

# 11. Target Encoding

The target variable `Passed` contained:

```text
Yes
No
```

It was converted into numerical values:

```python
y = y.map({
    "No": 0,
    "Yes": 1
})
```

Therefore:

```text
No  → 0
Yes → 1
```

This makes the target suitable for binary classification algorithms.

---

# 12. Feature Scaling

The numerical features had different ranges.

For example:

* Age was around 20–23.
* Attendance was around 70–95.
* Previous scores were around 65–90.

Min-Max Scaling was applied using Scikit-learn.

```python
from sklearn.preprocessing import MinMaxScaler

numerical_columns = [
    "Age",
    "Study_Hours",
    "Attendance",
    "Previous_Score"
]

scaler = MinMaxScaler()

X[numerical_columns] = scaler.fit_transform(
    X[numerical_columns]
)
```

Min-Max Scaling transforms values to a range between 0 and 1.

The formula is:

```text
X_scaled = (X - X_min) / (X_max - X_min)
```

This makes numerical features more comparable in scale.

---

# 13. Data Visualization

Visualization was performed to understand patterns in the dataset.

## 13.1 Attendance Distribution

A histogram was created using:

```python
plt.hist(df["Attendance"], bins=5)

plt.title("Distribution of Student Attendance")
plt.xlabel("Attendance")
plt.ylabel("Number of Students")

plt.show()
```

This visualization shows how attendance values are distributed among students.

---

## 13.2 Study Hours vs Previous Score

A scatter plot was created:

```python
plt.scatter(
    df["Study_Hours"],
    df["Previous_Score"]
)

plt.title("Study Hours vs Previous Score")
plt.xlabel("Study Hours")
plt.ylabel("Previous Score")

plt.show()
```

This visualization allows us to examine the relationship between study hours and previous examination scores.

---

# 14. Final Processed Dataset

After preprocessing, the processed features and target variable were combined:

```python
processed_df = X.copy()

processed_df["Passed"] = y
```

The final dataset was saved as:

```python
processed_df.to_csv(
    "cleaned_student_dataset.csv",
    index=False
)
```

The resulting file is:

```text
cleaned_student_dataset.csv
```

---

# 15. Preprocessing Summary

| Preprocessing Step         | Action Taken                          |
| -------------------------- | ------------------------------------- |
| Data loading               | Created Pandas DataFrame              |
| Data inspection            | Checked shape, columns and data types |
| Missing values             | Identified missing records            |
| Numerical missing values   | Replaced using median                 |
| Categorical missing values | Replaced using mode                   |
| Duplicate checking         | Checked for duplicate rows            |
| Feature selection          | Removed Student_ID                    |
| Target selection           | Selected Passed                       |
| Categorical encoding       | Applied one-hot encoding to Gender    |
| Target encoding            | Converted Yes/No to 1/0               |
| Feature scaling            | Applied Min-Max Scaling               |
| Visualization              | Created histogram and scatter plot    |
| Output                     | Saved cleaned dataset as CSV          |

---

# 16. Final Result

The original dataset contained missing values and categorical variables.

After preprocessing:

* Missing numerical values were handled.
* Missing categorical values were handled.
* Duplicate records were checked.
* The irrelevant Student_ID column was removed from the feature set.
* Categorical data was converted into numerical form.
* Numerical features were scaled.
* The target variable was encoded.
* Data visualizations were created.
* The final processed dataset was saved as a CSV file.

The resulting dataset is ready to be used as input for further Machine Learning tasks.

---

# 17. Project Structure

```text
ML-Data-Preprocessing/
│
├── ML-Data-Preprocessing.py
├── cleaned_student_dataset.csv
└── README.md
```

---

# 18. Conclusion

This project provided practical experience with the fundamental data preprocessing techniques used in Machine Learning.

The project demonstrated how Python, NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn can be used to inspect, clean, transform, visualize, and prepare data for Machine Learning.

Data preprocessing is an important stage because the quality and structure of the input data can affect the performance and usability of Machine Learning models.
