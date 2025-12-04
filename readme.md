Heart Disease Prediction with Random Forest

Overview
This project is about predicting the risk of heart disease using a **Random Forest Classifier**. The idea is simple: given patient health data (like age, cholesterol, blood pressure, etc.), the model tries to figure out whether someone is likely to have heart disease.  
Random Forest is a great choice here because it’s robust, handles mixed data well, and doesn’t overfit easily. To make sure the model performs at its best, I also applied **hyperparameter tuning** using GridSearchCV.

Why This Project?
Heart disease is one of the leading causes of death worldwide. Early prediction can help doctors and patients take preventive measures. This project is a small step in that direction — showing how machine learning can support healthcare decisions.

Project Structure
```
Heart_disease_prediction_using_RF/
│── random forest classifier.ipynb   # Notebook with training + tuning
│── requirements.txt                  # Dependencies
│── randomforest_model.pkl            # Saved trained model
│── README.md                         # Project documentation
```
Dataset
I used the UCI Heart Disease dataset, which includes features like:
- Age  
- Sex  
- Chest pain type  
- Resting blood pressure  
- Cholesterol level  
- Maximum heart rate achieved  
- Exercise-induced angina  

The target variable is binary:  
- `0` → No heart disease  
- `1` → Heart disease present  

Model Training
Here’s the workflow I followed:
1. Preprocessed the dataset (handled missing values, encoded categorical features).  
2. Trained a baseline Random Forest model.  
3. Tuned hyperparameters using GridSearchCV to find the best combination.  

Example of the tuning setup:
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}
```
Results
- The tuned Random Forest achieved strong accuracy on the test set.  
- Important features included **age**, **cholesterol**, and **maximum heart rate**.  
- Confusion matrix and classification report confirmed balanced performance across classes.
