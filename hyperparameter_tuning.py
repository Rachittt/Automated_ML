import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier
import joblib
# import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# Function for regression model hyperparameter tuning
def tune_regression_model(X_train, y_train):
    # Regression Models
    regression_models = {
        'Linear Regression': LinearRegression(),
        # 'Random Forest Regressor': RandomForestRegressor(),
        # 'Decision Tree Regressor': DecisionTreeRegressor(),
        # 'Support Vector Regressor': SVR(),
        # 'K-Neighbors Regressor': KNeighborsRegressor(),
        # 'XGBoost Regressor': XGBRegressor()
    }

    best_model_name = None
    best_model_score = float('-inf')  # Initialize with a very low value

    for model_name, model in regression_models.items():
        print(f"Tuning hyperparameters for {model_name}...")
        param_grid = get_regression_param_grid(model_name)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
        grid_search.fit(X_train, y_train)

        print(f"Best Hyperparameters for {model_name}: {grid_search.best_params_}")
        print(f"Best MSE on Validation Set: {grid_search.best_score_}\n")
        if grid_search.best_score_ > best_model_score:
            best_model_score = grid_search.best_score_
            best_model_name = model_name
            best_model = grid_search

    return best_model, best_model_name, best_model_score

# Function for classification model hyperparameter tuning
def tune_classification_model(X_train, y_train):
    # Classification Models
    classification_models = {
        'Logistic Regression': LogisticRegression(),
        # 'Random Forest Classifier': RandomForestClassifier(),
        # 'Decision Tree Classifier': DecisionTreeClassifier(),
        # 'Support Vector Classifier': SVC(),
        # 'K-Neighbors Classifier': KNeighborsClassifier(),
        # 'XGBoost Classifier': XGBClassifier()
    }

    best_model_name = None
    best_model_score = 0  # Initialize with a very low value

    for model_name, model in classification_models.items():
        print(f"Tuning hyperparameters for {model_name}...")
        param_grid = get_classification_param_grid(model_name)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
        grid_search.fit(X_train, y_train)

        print(f"Best Hyperparameters for {model_name}: {grid_search.best_params_}")
        print(f"Best Accuracy on Validation Set: {grid_search.best_score_}\n")
        if grid_search.best_score_ > best_model_score:
            best_model_score = grid_search.best_score_
            best_model_name = model_name
            best_model = grid_search

    return best_model, best_model_name, best_model_score

# Function to get hyperparameter grid for regression models
def get_regression_param_grid(model_name):
    if model_name == 'Linear Regression':
        return {}
    elif model_name == 'Random Forest Regressor':
        return {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
    elif model_name == 'Decision Tree Regressor':
        return {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
    elif model_name == 'Support Vector Regressor':
        return {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    elif model_name == 'K-Neighbors Regressor':
        return {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    elif model_name == 'XGBoost Regressor':
        return {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]}
    else:
        raise ValueError(f"Unsupported regression model: {model_name}")

# Function to get hyperparameter grid for classification models
def get_classification_param_grid(model_name):
    if model_name == 'Logistic Regression':
        return {'C': [0.1, 1, 10], 'max_iter': [100, 200, 300]}
    elif model_name == 'Random Forest Classifier':
        return {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
    elif model_name == 'Decision Tree Classifier':
        return {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
    elif model_name == 'Support Vector Classifier':
        return {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    elif model_name == 'K-Neighbors Classifier':
        return {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    elif model_name == 'XGBoost Classifier':
        return {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]}
    else:
        raise ValueError(f"Unsupported classification model: {model_name}")


def print_regression_scores(model, X_test, y_test, st):
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate and print various regression scores
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Squared Error (MSE): {round(mse, 2)}")
    st.write(f"Mean Absolute Error (MAE): {round(mae, 2)}")
    st.write(f"R-squared (R2): {round(r2, 2)}")
    # print(f"Mean Squared Error (MSE): {mse:.4f}")
    # print(f"Mean Absolute Error (MAE): {mae:.4f}")
    # print(f"R-squared (R2): {r2:.4f}")


def print_classification_scores(model, X_test, y_test, st):
    # Make predictions
    y_pred = model.predict(X_test)

    confusion_mat = confusion_matrix(y_test, y_pred)

    sns.heatmap(confusion_mat, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot()
    plt.savefig('download/confusion_matrix.png')

    # st.write("Classification Report")
    st.text('Model Report:\n ' + classification_report(y_test, y_pred))


def final_model(X, y, best_model):
    model = best_model.fit(X, y)
    joblib.dump(model, 'download/model.joblib')