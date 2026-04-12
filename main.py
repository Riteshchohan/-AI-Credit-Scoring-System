"""
FastAPI Backend for Loan Approval Prediction System
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import pandas as pd
import pickle
import os
import logging
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
from loan_approval_model import (
    load_data,
    preprocess_data,
    train_model,
    predict_with_credit_score,
    explain_prediction
)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# SECURITY CONFIGURATION
# ============================================================================

SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# ============================================================================
# LIFESPAN EVENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    load_model_and_data()
    logger.info("Model loaded successfully")
    yield
    # Shutdown
    logger.info("Shutting down...")

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Loan Approval Prediction API",
    description="ML-powered API for loan approval prediction with explainability",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS so the React frontend can call this API from localhost:3001
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL VARIABLES (Model & Preprocessor)
# ============================================================================

MODEL = None
PREPROCESSOR = None
X_TRAIN = None
FEATURE_NAMES = None

# ============================================================================
# AUTHENTICATION SCHEMAS
# ============================================================================

class UserCreate(BaseModel):
    """Schema for user registration"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    password: str = Field(..., min_length=6)

class UserLogin(BaseModel):
    """Schema for user login"""
    username: str
    password: str

class Token(BaseModel):
    """Schema for authentication token"""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Schema for token payload"""
    username: Optional[str] = None

# ============================================================================
# AUTHENTICATION FUNCTIONS
# ============================================================================

def verify_password(plain_password, hashed_password):
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def authenticate_user(username: str, password: str = None, token_validation: bool = False):
    """Authenticate a user (simplified - in production use proper password hashing)"""
    # For demo purposes, using simple password check
    # In production, this should use proper password hashing like bcrypt
    users_db = {
        "admin": {
            "username": "admin",
            "email": "admin@example.com",
            "password": "admin123",  # Plain text for demo
            "disabled": False
        },
        "user": {
            "username": "user",
            "email": "user@example.com",
            "password": "user123",  # Plain text for demo
            "disabled": False
        }
    }

    user = users_db.get(username)
    if not user:
        return False

    # For token validation, just check if user exists
    if token_validation:
        return user

    # For login, check password
    if user.get("password") != password:
        return False
    return user

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = authenticate_user(token_data.username, token_validation=True)  # Just check if user exists for token validation
    if user is False:
        raise credentials_exception
    return user

# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================

class LoanApplicationRequest(BaseModel):
    """Schema for loan application input"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Gender": "Male",
                "Married": "Yes",
                "Dependents": "2",
                "Education": "Graduate",
                "Self_Employed": "No",
                "ApplicantIncome": 50000,
                "CoapplicantIncome": 0,
                "LoanAmount": 200000,
                "Loan_Amount_Term": 360,
                "Credit_History": 1.0,
                "Property_Area": "Urban"
            }
        }
    )
    
    Gender: str = Field(..., description="Male/Female")
    Married: str = Field(..., description="Yes/No")
    Dependents: str = Field(..., description="Number of dependents (0, 1, 2, 3+)")
    Education: str = Field(..., description="Graduate/Undergraduate")
    Self_Employed: str = Field(..., description="Yes/No")
    ApplicantIncome: float = Field(..., description="Income in USD")
    CoapplicantIncome: float = Field(..., description="Co-applicant income in USD")
    LoanAmount: float = Field(..., description="Loan amount in USD")
    Loan_Amount_Term: float = Field(..., description="Loan term in months")
    Credit_History: float = Field(..., description="Credit history (0.0-1.0)")
    Property_Area: str = Field(..., description="Urban/Semiurban/Rural")

class FeatureImportance(BaseModel):
    """Schema for SHAP feature importance"""
    feature: str
    impact: float
    value: float

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    loan_approved: int = Field(..., description="0=Rejected, 1=Approved")
    approval_probability: float = Field(..., description="Probability of approval (0.0-1.0)")
    credit_score: int = Field(..., description="Credit score (300-900)")
    top_features: List[FeatureImportance] = Field(..., description="Top 3 features affecting decision")

class HealthResponse(BaseModel):
    """Schema for health check response"""
    status: str
    message: str

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _get_transformed_feature_names(preprocessor) -> list:
    """
    Best-effort extraction of transformed feature names from sklearn ColumnTransformer.
    Returns a list matching the transformed feature order.
    """
    if preprocessor is None:
        return None

    # Newer sklearn: ColumnTransformer.get_feature_names_out()
    try:
        names = preprocessor.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        pass

    # Fallback: build from transformers_ (less precise for nested pipelines)
    try:
        names = []
        for name, transformer, cols in getattr(preprocessor, "transformers_", []):
            if transformer == "drop":
                continue
            if transformer == "passthrough":
                if cols is None:
                    continue
                if isinstance(cols, (list, tuple)):
                    names.extend([str(c) for c in cols])
                else:
                    names.append(str(cols))
                continue

            # Handle Pipeline with OneHotEncoder at the end
            try:
                if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
                    ohe = transformer.named_steps["onehot"]
                    ohe_names = ohe.get_feature_names_out(cols)
                    names.extend([str(n) for n in ohe_names])
                    continue
            except Exception:
                pass

            # Handle OneHotEncoder directly
            try:
                if hasattr(transformer, "get_feature_names_out"):
                    out = transformer.get_feature_names_out(cols)
                    names.extend([str(n) for n in out])
                    continue
            except Exception:
                pass

            # Last resort: just use the original columns
            if isinstance(cols, (list, tuple)):
                names.extend([str(c) for c in cols])
            else:
                names.append(str(cols))

        return names or None
    except Exception:
        return None

def _prettify_feature_name(raw: str) -> str:
    """
    Convert sklearn feature names like 'num__ApplicantIncome' or
    'cat__Gender_Male' to a cleaner label.
    """
    if raw is None:
        return raw
    s = str(raw)
    s = s.replace("num__", "").replace("cat__", "")
    # ColumnTransformer can yield e.g. "cat__onehot__Gender_Male"
    s = s.replace("onehot__", "")
    return s

def load_model_and_data():
    """
    Load trained model and preprocessor from saved files.
    If they don't exist, train them from scratch.
    """
    global MODEL, PREPROCESSOR, X_TRAIN, FEATURE_NAMES
    
    model_path = "loan_model.pkl"
    preprocessor_path = "loan_preprocessor.pkl"
    x_train_path = "loan_x_train.pkl"
    
    # Check if model exists
    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        print("Loading saved model and preprocessor...")
        with open(model_path, "rb") as f:
            MODEL = pickle.load(f)
        with open(preprocessor_path, "rb") as f:
            PREPROCESSOR = pickle.load(f)

        raw_names = _get_transformed_feature_names(PREPROCESSOR)
        FEATURE_NAMES = [_prettify_feature_name(n) for n in raw_names] if raw_names else None
        
        # Load X_TRAIN if available (for SHAP explanations)
        if os.path.exists(x_train_path):
            with open(x_train_path, "rb") as f:
                X_TRAIN = pickle.load(f)
        else:
            # Reload training data if X_TRAIN not available
            df = load_data("loan_data.csv")
            if df is None:
                raise Exception("Failed to load loan_data.csv")
            X_TRAIN, _, _ = preprocess_data(df)
            # Convert sparse to dense for SHAP
            if hasattr(X_TRAIN, 'toarray'):
                X_TRAIN = X_TRAIN.toarray()
    else:
        print("Training new model...")
        # Load and preprocess data
        df = load_data("loan_data.csv")
        if df is None:
            raise Exception("Failed to load loan_data.csv")
        
        X_TRAIN, y_train, PREPROCESSOR = preprocess_data(df)

        raw_names = _get_transformed_feature_names(PREPROCESSOR)
        FEATURE_NAMES = [_prettify_feature_name(n) for n in raw_names] if raw_names else None
        
        # Convert sparse to dense for SHAP
        if hasattr(X_TRAIN, 'toarray'):
            X_TRAIN = X_TRAIN.toarray()
        
        # Train model
        MODEL = train_model(X_TRAIN, y_train)
        
        # Save for future use
        with open(model_path, "wb") as f:
            pickle.dump(MODEL, f)
        with open(preprocessor_path, "wb") as f:
            pickle.dump(PREPROCESSOR, f)
        with open(x_train_path, "wb") as f:
            pickle.dump(X_TRAIN, f)
        
        print("Model trained and saved!")


def ensure_model_loaded():
    """
    Ensure the model and preprocessor are available.
    """
    global MODEL, PREPROCESSOR, X_TRAIN, FEATURE_NAMES
    if MODEL is None or PREPROCESSOR is None:
        load_model_and_data()

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.post("/register", response_model=Token)
async def register_user(user: UserCreate):
    """
    Register a new user (simplified - in production use database)
    """
    try:
        logger.info(f"User registration attempt: {user.username}")

        # In production, check if user exists in database
        # For demo, we'll just return a token for any registration
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )

        logger.info(f"User registered successfully: {user.username}")
        return {"access_token": access_token, "token_type": "bearer"}

    except Exception as e:
        logger.error(f"Registration failed for {user.username}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {str(e)}"
        )

@app.post("/login", response_model=Token)
async def login_user(user: UserLogin):
    """
    Authenticate user and return JWT token
    """
    try:
        logger.info(f"Login attempt: {user.username}")

        authenticated_user = authenticate_user(user.username, user.password)
        if not authenticated_user:
            logger.warning(f"Failed login attempt: {user.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": authenticated_user["username"]}, expires_delta=access_token_expires
        )

        logger.info(f"Successful login: {user.username}")
        return {"access_token": access_token, "token_type": "bearer"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error for {user.username}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed due to server error"
        )

@app.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint - verify API is running
    """
    return {
        "status": "success",
        "message": "API Running"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_loan(
    request: LoanApplicationRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Predict loan approval with credit score and feature importance.

    - Accepts user loan application data
    - Returns: approval status, probability, credit score, and top 3 features
    - Requires authentication
    """
    try:
        logger.info(f"Prediction request from user: {current_user['username']}")

        # Ensure the model is loaded before prediction
        ensure_model_loaded()
        global FEATURE_NAMES
        if FEATURE_NAMES is None:
            raw_names = _get_transformed_feature_names(PREPROCESSOR)
            FEATURE_NAMES = [_prettify_feature_name(n) for n in raw_names] if raw_names else None

        # Convert request to DataFrame
        input_data = request.model_dump()
        input_df = pd.DataFrame([input_data])

        # Preprocess input
        X_input = PREPROCESSOR.transform(input_df)

        # Get prediction with credit score
        result = predict_with_credit_score(MODEL, X_input)

        # Get feature importance explanation
        explanation = explain_prediction(
            model=MODEL,
            X_input=X_input,
            X_background=X_TRAIN,
            feature_names=FEATURE_NAMES
        )

        logger.info(f"Prediction completed for user {current_user['username']}: approved={result['prediction']}")

        # Format response
        return {
            "loan_approved": result['prediction'],
            "approval_probability": result['probability'],
            "credit_score": result['credit_score'],
            "top_features": explanation['top_features']
        }

    except Exception as e:
        logger.error(f"Prediction failed for user {current_user['username']}: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full error trace
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )

# ============================================================================
# ADDITIONAL ENDPOINTS
# ============================================================================

@app.get("/health")
async def detailed_health():
    """Detailed health check with model status"""
    return {
        "api_status": "running",
        "model_loaded": MODEL is not None,
        "preprocessor_loaded": PREPROCESSOR is not None,
        "training_samples": X_TRAIN.shape[0] if X_TRAIN is not None else 0
    }

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle value errors"""
    return {
        "error": "Invalid input",
        "details": str(exc)
    }

# ============================================================================
# RUN INSTRUCTIONS
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run the API
    # Command: python main.py
    # Or: uvicorn main:app --reload --host 0.0.0.0 --port 8000
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
