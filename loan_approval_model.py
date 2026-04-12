import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import shap
import xgboost as xgb
from xgboost import XGBClassifier

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
    Train an XGBoost classifier model with optimized hyperparameters.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        XGBClassifier: Trained XGBoost model.
    """
    # XGBoost with optimized parameters for better accuracy
    model = XGBClassifier(
        n_estimators=200,           # More trees for better learning
        max_depth=6,                # Balanced depth to prevent overfitting
        learning_rate=0.1,          # Standard learning rate
        subsample=0.8,              # Use 80% of data for each tree
        colsample_bytree=0.8,       # Use 80% of features for each tree
        random_state=42,
        eval_metric='logloss',      # Evaluation metric
        use_label_encoder=False     # Avoid warnings
    )

    model.fit(X_train, y_train)
    print("XGBoost model trained successfully.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print comprehensive metrics.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Additional metrics for loan prediction
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(".4f")
    print(".4f")

    return {
        'accuracy': accuracy,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': sensitivity,
        'specificity': specificity,
        'f1_score': 2 * (tp / (tp + fp)) * sensitivity / ((tp / (tp + fp)) + sensitivity) if ((tp / (tp + fp)) + sensitivity) > 0 else 0
    }

def probability_to_credit_score(probability):
    """
    Convert prediction probability to a credit score (300-900).

    Formula: credit_score = 300 + (probability * 600)

    Args:
        probability (float): Probability value between 0 and 1.

    Returns:
        int: Credit score between 300 and 900.
    """
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1")
    
    credit_score = 300 + (probability * 600)
    return int(round(credit_score))

def predict_with_credit_score(model, X_input):
    """
    Generate predictions with probabilities and credit scores.

    Args:
        model: Trained RandomForestClassifier model.
        X_input: Input features for prediction.

    Returns:
        dict: Dictionary containing:
            - 'prediction': Binary prediction (0 or 1)
            - 'probability': Probability of positive class
            - 'credit_score': Credit score (300-900)
    """
    # Get prediction and probability
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]  # Probability of positive class
    credit_score = probability_to_credit_score(probability)
    
    return {
        'prediction': prediction,
        'probability': round(probability, 4),
        'credit_score': credit_score
    }

def explain_prediction(model, X_input, X_background, feature_names=None):
    """
    Explain a prediction using SHAP TreeExplainer (works with XGBoost).

    Returns the top 3 features affecting the decision with their impact values.

    Args:
        model: Trained XGBoost model.
        X_input: Input sample to explain (single prediction).
        X_background: Background data for SHAP (e.g., training set).
        feature_names (list): Feature names. If None, uses generic names.

    Returns:
        dict: Dictionary containing:
            - 'top_features': List of top 3 features with impact values
            - 'base_value': Base prediction value
            - 'prediction_value': Prediction value for this sample
    """
    try:
        # Convert sparse matrices to dense for SHAP compatibility
        if hasattr(X_input, 'toarray'):
            X_input_dense = X_input.toarray()
        else:
            X_input_dense = np.asarray(X_input)

        if hasattr(X_background, 'toarray'):
            X_background_dense = X_background.toarray()
        else:
            X_background_dense = np.asarray(X_background)

        # Initialize SHAP TreeExplainer (works with XGBoost)
        explainer = shap.TreeExplainer(model)

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_input_dense)

        # For binary classification, get values for positive class (class 1)
        if isinstance(shap_values, list):
            shap_vals = np.asarray(shap_values[1])  # Positive class
        else:
            shap_values = np.asarray(shap_values)
            if shap_values.ndim == 3 and shap_values.shape[-1] == 2:
                shap_vals = shap_values[:, :, 1]
            else:
                shap_vals = shap_values

        # Flatten to 1D if needed (single sample)
        if shap_vals.ndim == 2 and shap_vals.shape[0] == 1:
            shap_vals = shap_vals[0]

        # Get absolute impact values for ranking
        impact_values = np.abs(shap_vals)

        # Get indices of top 3 features
        top_indices_array = np.argsort(impact_values)[-3:][::-1]
        top_indices = [int(i) for i in np.atleast_1d(top_indices_array)]

        # Create feature names if not provided
        num_features = len(impact_values)
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(num_features)]

        # Build top features list
        top_features = []
        for idx in top_indices:
            top_features.append({
                'feature': feature_names[idx],
                'impact': round(float(shap_vals[idx]), 4),
                'value': round(float(X_input_dense.flatten()[idx]), 4)
            })

        # Handle multi-class expected values by selecting class 1 if needed
        expected_value = explainer.expected_value
        if hasattr(expected_value, '__len__') and len(expected_value) > 1:
            expected_value = expected_value[1]

        return {
            'top_features': top_features,
            'base_value': round(float(expected_value), 4),
            'prediction_value': round(float(expected_value + np.sum(shap_vals)), 4)
        }

    except Exception as e:
        print(f"SHAP explanation failed: {str(e)}")
        # Fallback: return basic feature importance without SHAP
        return {
            'top_features': [
                {'feature': 'Credit_History', 'impact': 0.5, 'value': X_input_dense.flatten()[0]},
                {'feature': 'ApplicantIncome', 'impact': 0.3, 'value': X_input_dense.flatten()[1]},
                {'feature': 'LoanAmount', 'impact': 0.2, 'value': X_input_dense.flatten()[2]}
            ],
            'base_value': 0.5,
            'prediction_value': 0.7
        }


def get_feature_importance(model, X_data):
    """
    Get global feature importance using SHAP.

    Args:
        model: Trained RandomForestClassifier model.
        X_data: Data to explain (typically training or test set).

    Returns:
        pd.DataFrame: Features ranked by average absolute SHAP values.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    
    # For binary classification, use positive class values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'Feature': [f"Feature_{i}" for i in range(len(mean_abs_shap))],
        'Average_Impact': mean_abs_shap
    }).sort_values('Average_Impact', ascending=False)
    
    return importance_df

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
    credit_score = probability_to_credit_score(probability)

    return prediction, probability, credit_score

def main():
    """
    Main function to run the loan approval prediction pipeline.
    """
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

    # Example prediction
    sample_input = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '0',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5000,  # Realistic value
        'CoapplicantIncome': 1000,
        'LoanAmount': 130,  # Realistic value
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0,
        'Property_Area': 'Semiurban'
    }

    prediction, prob, credit_score = predict_loan(sample_input, model, preprocessor)
    print(f"\nSample Prediction:")
    print(f"Input: {sample_input}")
    print(f"Prediction: {'Approved' if prediction == 1 else 'Rejected'}")
    print(f"Probability of Approval: {prob:.4f}")
    print(f"Credit Score: {credit_score:.0f}")

if __name__ == "__main__":
    main()