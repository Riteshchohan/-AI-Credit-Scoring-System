import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """
    Load the dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

def preprocess_data(df):
    """
    Preprocess the data: handle missing values and encode categorical columns.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        tuple: (X_preprocessed, y, preprocessor)
    """
    # Separate features and target, drop Loan_ID as it's not useful for prediction
    X = df.drop(['Loan_Status', 'Loan_ID'], axis=1)
    y = df['Loan_Status'].map({'Y': 1, 'N': 0})  # Convert to numeric

    # Check for any unmapped values
    if y.isnull().any():
        print(f"Warning: Found {y.isnull().sum()} unmapped values in target column")
        y = y.dropna()  # Remove rows with unmapped targets
        X = X.loc[y.index]  # Keep corresponding rows in X

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Create preprocessing pipeline
    numerical_transformer = SimpleImputer(strategy='mean')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Fit and transform the data
    X_preprocessed = preprocessor.fit_transform(X)

    print(f"Data preprocessed. Features shape: {X_preprocessed.shape}")
    return X_preprocessed, y, preprocessor

def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier model.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        RandomForestClassifier: Trained model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained successfully.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print detailed metrics.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def calculate_credit_score(probability):
    """
    Convert prediction probability to credit score (300-900 range).

    Args:
        probability (float): Probability of loan approval (0.0 to 1.0)

    Returns:
        int: Credit score between 300 and 900
    """
    return int(300 + (probability * 600))

def predict_loan(data_dict, model, preprocessor):
    """
    Predict loan approval for new data and calculate credit score.

    Args:
        data_dict (dict): Dictionary containing feature values.
        model: Trained model.
        preprocessor: Fitted preprocessor.

    Returns:
        tuple: (prediction, probability, credit_score)
            - prediction: 0 (Rejected) or 1 (Approved)
            - probability: Probability of approval (0.0 to 1.0)
            - credit_score: Credit score between 300 and 900
    """
    # Convert dict to DataFrame
    input_df = pd.DataFrame([data_dict])

    # Check if all required columns are present
    required_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                     'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                     'Credit_History', 'Property_Area']

    missing_cols = [col for col in required_cols if col not in input_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Preprocess the input
    input_preprocessed = preprocessor.transform(input_df)

    # Make prediction
    prediction = model.predict(input_preprocessed)[0]
    probability = model.predict_proba(input_preprocessed)[0][1]  # Probability of approval (class 1)

    # Calculate credit score using reusable function
    credit_score = calculate_credit_score(probability)

    return prediction, probability, credit_score

def test_model():
    """
    Comprehensive test function for the loan approval model.
    """
    print("=" * 60)
    print("LOAN APPROVAL MODEL COMPREHENSIVE TEST")
    print("=" * 60)

    # File path
    file_path = 'loan_data.csv'

    # Load data
    df = load_data(file_path)
    if df is None:
        return

    # Preprocess data
    X, y, preprocessor = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    print("\n" + "=" * 60)
    print("PREDICTION TESTS")
    print("=" * 60)

    # Test cases with realistic values from the dataset
    test_cases = [
        {
            'name': 'High Income, Good Credit (Realistic)',
            'data': {
                'Gender': 'Male',
                'Married': 'Yes',
                'Dependents': '0',
                'Education': 'Graduate',
                'Self_Employed': 'No',
                'ApplicantIncome': 10000,  # More realistic for the dataset
                'CoapplicantIncome': 2000,
                'LoanAmount': 150,  # In hundreds, as per dataset
                'Loan_Amount_Term': 360,
                'Credit_History': 1.0,
                'Property_Area': 'Urban'
            }
        },
        {
            'name': 'Low Income, Bad Credit (Realistic)',
            'data': {
                'Gender': 'Female',
                'Married': 'No',
                'Dependents': '2',
                'Education': 'Not Graduate',
                'Self_Employed': 'Yes',
                'ApplicantIncome': 2000,
                'CoapplicantIncome': 0,
                'LoanAmount': 120,
                'Loan_Amount_Term': 360,
                'Credit_History': 0.0,
                'Property_Area': 'Rural'
            }
        },
        {
            'name': 'Medium Income, Average Credit (Realistic)',
            'data': {
                'Gender': 'Male',
                'Married': 'Yes',
                'Dependents': '1',
                'Education': 'Graduate',
                'Self_Employed': 'No',
                'ApplicantIncome': 5000,
                'CoapplicantIncome': 1000,
                'LoanAmount': 130,
                'Loan_Amount_Term': 360,
                'Credit_History': 1.0,
                'Property_Area': 'Semiurban'
            }
        },
        {
            'name': 'Very High Income, Good Credit',
            'data': {
                'Gender': 'Male',
                'Married': 'Yes',
                'Dependents': '0',
                'Education': 'Graduate',
                'Self_Employed': 'No',
                'ApplicantIncome': 30000,  # High but within dataset range
                'CoapplicantIncome': 5000,
                'LoanAmount': 200,
                'Loan_Amount_Term': 360,
                'Credit_History': 1.0,
                'Property_Area': 'Urban'
            }
        }
    ]

    for test_case in test_cases:
        print(f"\nTest Case: {test_case['name']}")
        print("-" * 40)
        try:
            prediction, prob, credit_score = predict_loan(test_case['data'], model, preprocessor)
            print(f"Input: {test_case['data']}")
            print(f"Prediction: {'Approved' if prediction == 1 else 'Rejected'}")
            print(f"Probability of Approval: {prob:.4f}")
            print(f"Credit Score: {credit_score:.0f}")
        except Exception as e:
            print(f"Error: {e}")

    # Test error handling
    print("\n" + "=" * 60)
    print("ERROR HANDLING TESTS")
    print("=" * 60)

    print("\nTest: Missing required column")
    try:
        incomplete_data = {
            'Gender': 'Male',
            'Married': 'Yes',
            # Missing other required columns
        }
        predict_loan(incomplete_data, model, preprocessor)
    except ValueError as e:
        print(f"Expected error caught: {e}")

    print("\nTest: Invalid data types")
    try:
        invalid_data = {
            'Gender': 'Male',
            'Married': 'Yes',
            'Dependents': '0',
            'Education': 'Graduate',
            'Self_Employed': 'No',
            'ApplicantIncome': 'invalid',  # Should be numeric
            'CoapplicantIncome': 0,
            'LoanAmount': 15000,
            'Loan_Amount_Term': 360,
            'Credit_History': 1.0,
            'Property_Area': 'Urban'
        }
        predict_loan(invalid_data, model, preprocessor)
    except Exception as e:
        print(f"Error caught: {e}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    test_model()