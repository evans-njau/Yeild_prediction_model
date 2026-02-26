# 🌾 Maji Ndogo Agricultural Yield Prediction Model

An end-to-end Machine Learning project that predicts agricultural crop yield using environmental and farming-related features.  
This project demonstrates data preprocessing, feature engineering, model training, and evaluation using a Random Forest Regressor.

---

## Table of Contents

- Project Overview  
- Dataset  
- Technologies Used  
- Installation  
- Project Workflow  
- Model Evaluation  
- Results  
- Future Improvements  
- Project Structure  
- Author  

---

## Project Overview

Accurate crop yield prediction is important for agricultural planning and food security.  
This project builds a supervised regression model that estimates crop yield based on:

- Crop type  
- Season  
- Rainfall  
- Temperature  
- Fertilizer usage  
- Other agricultural variables  

The model uses a Random Forest Regressor to capture nonlinear relationships between environmental factors and yield.

---

## Dataset

Dataset Source:

https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/Python/Crop_yield.csv

### Target Variable
- `Yield`

### Feature Variables
- Crop
- Season
- Rainfall
- Temperature
- Fertilizer
- Other farming conditions

---

## Technologies Used

- Python 3.x  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Jupyter Notebook  

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

---

## Project Workflow

### 1️ Import Libraries

Essential data science and ML libraries are imported.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

---

### 2️ Load Dataset

```python
agri_data = pd.read_csv(
    "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/Python/Crop_yield.csv"
)
```

---

### 3️ Data Preprocessing

- One-hot encoding of categorical variables  
- Dropping the first category to prevent multicollinearity  

```python
data_encoded = pd.get_dummies(
    agri_data,
    drop_first=True,
    dtype=int
)
```

---

### 4️ Model Training Function

```python
def Train_Model(data, n_estimator, target_variable):

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error

    # Split features and target
    X = data.drop(columns=[target_variable])
    Y = data[target_variable]

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=0.2,
        random_state=42
    )

    # Model initialization
    model = RandomForestRegressor(
        n_estimators=n_estimator,
        random_state=42
    )

    # Train model
    model.fit(X_train, Y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluation
    print("R2 Score:", r2_score(Y_test, predictions))
    print("Mean Squared Error:", mean_squared_error(Y_test, predictions))

    return model
```

---

### 5️ Train the Model

```python
model = Train_Model(data_encoded, 100, "Yield")
```

---

## Model Evaluation

The model performance is evaluated using:

- **R² Score** – Measures how well predictions fit actual data  
- **Mean Squared Error (MSE)** – Measures average squared prediction error  

These metrics provide insight into prediction accuracy and model reliability.

---

## Why Random Forest?

Random Forest is chosen because:

- It handles nonlinear relationships effectively  
- It reduces overfitting compared to a single decision tree  
- It works well with encoded categorical variables  
- It provides robust performance without heavy hyperparameter tuning  

---

## Results

The model successfully predicts crop yield with strong regression performance metrics.  
Exact results depend on random state and dataset split but demonstrate solid predictive capability.

---

## Future Improvements

- Hyperparameter tuning using GridSearchCV  
- Cross-validation  
- Feature importance visualization  
- Model comparison (Linear Regression, XGBoost, Gradient Boosting)  
- Model deployment using Flask or FastAPI  
- Convert notebook into a production-ready ML pipeline  

---

## Project Structure

```
├── Crop_Yield_Prediction.ipynb
├── README.md
└── requirements.txt
```

---

##  Author

Evans Njau  
Machine Learning Engineer  

---

## ⭐ Acknowledgment

Dataset provided by Explore AI Public Data Repository.
