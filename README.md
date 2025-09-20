 **Google Colab project** for **Simple Linear Regression** using your uploaded dataset (`StudentsPerformance.csv`).

Here’s a **minimal and clear Colab notebook code** + a **README.md** for your GitHub repo.

---

## 📓 **Simple Linear Regression – Colab Notebook Code**

```python
# 📌 Simple Linear Regression on StudentsPerformance dataset

# 1️⃣ Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2️⃣ Load the dataset
df = pd.read_csv("/content/StudentsPerformance.csv")
print("Dataset shape:", df.shape)
df.head()

# 3️⃣ Select features for regression
# Example: Predicting 'math score' based on 'reading score'
X = df[['reading score']]  # Independent variable
y = df['math score']       # Dependent variable

# 4️⃣ Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 6️⃣ Make predictions
y_pred = model.predict(X_test)

# 7️⃣ Evaluate model performance
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 8️⃣ Plot the regression line
plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, color='blue', label="Actual values")
plt.plot(X_test, y_pred, color='red', linewidth=2, label="Regression line")
plt.xlabel("Reading Score")
plt.ylabel("Math Score")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
```

---

## 📄 **README.md for GitHub**

```markdown
# 📊 Simple Linear Regression – Students Performance

This repository contains a **Simple Linear Regression** project implemented in **Google Colab** using the **StudentsPerformance.csv** dataset.

---

## 📌 Project Overview

The goal of this project is to:
- Perform **Simple Linear Regression**
- Predict **Math Score** based on **Reading Score**
- Visualize the regression line
- Evaluate model performance using metrics like **MSE** and **R² score**

---

## 📂 Project Structure

```

├── Simple\_Linear\_Regression.ipynb  # Main Google Colab Notebook
├── StudentsPerformance.csv         # Dataset
└── README.md                       # Project Documentation

````

---

## 📊 Steps Performed

1️⃣ **Data Loading** – Load dataset using pandas  
2️⃣ **Feature Selection** – Use `reading score` as independent variable (X) and `math score` as dependent variable (y)  
3️⃣ **Train-Test Split** – Split dataset into training & testing sets  
4️⃣ **Model Training** – Train a `LinearRegression()` model from scikit-learn  
5️⃣ **Evaluation** – Calculate **Mean Squared Error (MSE)** & **R² score**  
6️⃣ **Visualization** – Plot regression line against actual test data  

---

## 📈 Example Output

- **Regression Equation:**  
  `math_score = intercept + slope * reading_score`

- **Metrics:**  
  - Mean Squared Error (MSE)  
  - R² Score (Goodness of Fit)

- **Plot:**  
  A scatter plot with the regression line.

---

## ⚙️ Installation & Usage

1. Clone this repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
````

2. Open the notebook in **Google Colab**.

3. Install dependencies (if running locally):

```bash
pip install pandas numpy scikit-learn matplotlib
```

4. Run all cells to reproduce the results.

---

## 📜 License

This project is licensed under the **MIT License**.

---



Would you like me to make this project slightly **more advanced** by including **multiple linear regression** (e.g., predicting math score using reading + writing scores) for a richer project? This would look very professional on GitHub.
```
