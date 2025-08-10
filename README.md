
# Heart Disease Prediction Project

## Overview
This project analyzes and predicts the risk of heart disease using machine learning. It covers data loading, preprocessing, feature engineering, dimensionality reduction, classification, clustering, hyperparameter tuning, and model export for deployment.

---

## Project Structure

- `heart_disease.csv` — Raw dataset  
- `notebooks/` — Jupyter notebooks for step-by-step exploration and model training  
- `models/` — Saved machine learning models (.pkl files)  
- `results/` — Generated plots and evaluation metrics  

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YusufAbozeid/heart-disease-prediction.git
cd heart-disease-prediction
(Optional) Create and activate a virtual environment:

bash
Copy code
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
Install required packages:

bash
Copy code
pip install -r requirements.txt
Usage
Run Jupyter notebooks
Explore data, perform EDA, feature selection, model training, and evaluation in the notebooks folder:

bash
Copy code
jupyter notebook
Example: Load saved model and predict heart disease risk
python
Copy code
import joblib
import pandas as pd

# Load saved model pipeline (preprocessing + model)
model = joblib.load('best_rf_pipeline.pkl')

# Define model features
features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak',
            'cp_1', 'cp_2', 'cp_3', 'cp_4',
            'restecg_0', 'restecg_1', 'restecg_2',
            'slope_1', 'slope_2', 'slope_3',
            'thal_3.0', 'thal_6.0', 'thal_7.0',
            'ca_0.0', 'ca_1.0', 'ca_2.0', 'ca_3.0',
            'sex_0', 'sex_1',
            'fbs_0', 'fbs_1',
            'exang_0', 'exang_1']

# Create input data dict with default zeros
data_dict = {feat: 0 for feat in features}

# Update with actual patient data
data_dict.update({
    'age': 58.0,
    'trestbps': 130.0,
    'chol': 250.0,
    'thalach': 150.0,
    'oldpeak': 2.3,
    'cp_2': 1,
    'restecg_0': 1,
    'slope_1': 1,
    'thal_6.0': 1,
    'ca_1.0': 1,
    'sex_1': 1,
    'fbs_0': 1,
    'exang_0': 1
})

input_df = pd.DataFrame([data_dict])

# Predict
prediction = model.predict(input_df)

print("Prediction:", prediction)
if prediction[0] == 1:
    print("The model predicts a high risk of heart disease.")
else:
    print("The model predicts a low risk of heart disease.")
Main Workflow Highlights
Data Preprocessing: Dropping missing values, one-hot encoding categorical variables, standard scaling numerical features.

Exploratory Data Analysis (EDA): Histograms, correlation heatmap, and boxplots by target class.

Dimensionality Reduction: PCA to retain 95% variance, visualized with explained variance and scatter plots.

Feature Selection: Random Forest feature importance, Recursive Feature Elimination (RFE), and Chi-square test for categorical features.

Classification Models: Logistic Regression, Decision Tree, Random Forest, and SVM evaluated on accuracy, precision, recall, F1-score, and ROC AUC.

Unsupervised Learning: K-Means and Hierarchical clustering with PCA visualization and dendrograms.

Hyperparameter Tuning: GridSearchCV and RandomizedSearchCV to optimize Random Forest parameters.

Model Export: Final best pipeline saved with joblib for easy deployment.

Requirements
(You can generate this by running pip freeze > requirements.txt in your environment)

Example key packages:

nginx
Copy code
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
scipy
css
Copy code

If you want, I can help generate a clean `requirements.txt` file based on your code imports too. Just ask!
