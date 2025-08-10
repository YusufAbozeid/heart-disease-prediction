# Heart Disease Prediction Project

## Overview
This project aims to analyze and predict the risk of heart disease using machine learning techniques. It includes data preprocessing, feature selection, dimensionality reduction (PCA), model training, evaluation, and deployment.

## Project Structure
- `data/` : Contains the dataset files.
- `notebooks/` : Jupyter notebooks for each step of the project (data preprocessing, PCA, feature selection, etc.).
- `models/` : Saved trained machine learning models in `.pkl` format.
- `results/` : Outputs such as plots and evaluation metrics.

## Installation
1. Clone the repository:
   git clone https://github.com/YusufAbozeid/heart-disease-prediction.git
cd heart-disease-prediction

2.Create and activate a virtual environment (optional but recommended):

bash
Copy code
python -m venv env
source env/bin/activate    # On Windows use `env\Scripts\activate

3.Install required packages:

bash
Copy code
pip install -r requirements.txt

Usage

1.Run the Jupyter notebook in notebooks/ folder to preprocess data and train models.

2.Load the saved model and predict heart disease risk with this example:

python
Copy code
import joblib
import pandas as pd

# Load saved model
model = joblib.load('models/best_rf_model.pkl')

# Define input features with example patient data
features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak',
            'cp_1', 'cp_2', 'cp_3', 'cp_4',
            'restecg_0', 'restecg_1', 'restecg_2',
            'slope_1', 'slope_2', 'slope_3',
            'thal_3.0', 'thal_6.0', 'thal_7.0',
            'ca_0.0', 'ca_1.0', 'ca_2.0', 'ca_3.0',
            'sex_0', 'sex_1',
            'fbs_0', 'fbs_1',
            'exang_0', 'exang_1']

data_dict = {feat: 0 for feat in features}

# Fill with actual values
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

prediction = model.predict(input_df)

print("Prediction:", prediction)
if prediction[0] == 1:
    print("The model predicts a high risk of heart disease.")
else:
    print("The model predicts a low risk of heart disease.")