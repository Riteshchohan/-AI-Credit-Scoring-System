"""
FastAPI Backend for Loan Approval Prediction System
With FREE AI Financial Assistant using Hugging Face Inference API
(No local model loading - runs on HF servers, zero performance impact)
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
import requests
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
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ============================================================================
# SECURITY CONFIGURATION
# ============================================================================

SECRET_KEY = os.getenv("SECRET_KEY", "change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# Hugging Face Inference API (no local model download needed)
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

MODEL = None
PREPROCESSOR = None
X_TRAIN = None
FEATURE_NAMES = None
LATEST_PREDICTION_CONTEXT = {"result": None, "score": None, "features": []}

# ============================================================================
# AI ASSISTANT - Hugging Face Inference API (lightweight, no model download)
# ============================================================================

def query_hf_api(prompt: str) -> str:
    """
    Call Hugging Face Inference API.
    No model downloaded - runs on HF servers.
    Fast, lightweight, free with HF account.
    """
    try:
        if not HF_API_TOKEN:
            logger.warning("HF_API_TOKEN not set, using fallback suggestions.")
            return None

        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 150, "temperature": 0.7}
        }
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=15)

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").strip()
        else:
            logger.warning(f"HF API returned status {response.status_code}")
        return None

    except requests.Timeout:
        logger.warning("HF API timed out, using fallback.")
        return None
    except Exception as e:
        logger.error(f"HF API call failed: {e}")
        return None


def generate_suggestions(result: str, score: int, features: list) -> str:
    """
    Generate AI financial suggestions using HF Inference API.
    Falls back to rule-based suggestions if API unavailable.
    """
    try:
        feature_text = ", ".join(
            [f"{f['feature']} (impact: {f['impact']:.3f})" for f in features[:3]]
        ) if features else "credit history, income, loan amount"

        prompt = (
            f"You are a fintech financial advisor. "
            f"A loan application was {result} with credit score {score}. "
            f"Top factors: {feature_text}. "
            f"Give 3 short actionable suggestions to improve loan approval chances."
        )

        answer = query_hf_api(prompt)
        return answer if answer else _fallback_suggestions(result, score, features)

    except Exception as e:
        logger.error(f"generate_suggestions failed: {e}")
        return _fallback_suggestions(result, score, features)


def answer_user_question(question: str, context: dict) -> str:
    """
    Answer user questions about their loan using HF Inference API.
    """
    try:
        result = context.get("result", "Unknown")
        score = context.get("score", "Unknown")
        features = context.get("features", [])
        feature_text = ", ".join(
            [f"{f['feature']} (impact: {f['impact']:.3f})" for f in features[:3]]
        ) if features else "not available"

        prompt = (
            f"You are a helpful fintech AI assistant. "
            f"Context: Loan was {result}, credit score is {score}, "
            f"top factors are {feature_text}. "
            f"Question: {question} "
            f"Answer clearly in 2-3 sentences."
        )

        answer = query_hf_api(prompt)
        return answer if answer else "I could not process your question at this time. Please try again."

    except Exception as e:
        logger.error(f"answer_user_question failed: {e}")
        return "I could not process your question at this time. Please try again."


def _fallback_suggestions(result: str, score: int, features: list) -> str:
    """Rule-based fallback when AI API is unavailable."""
    if result == "Approved":
        return (
            f"Your loan was approved with a credit score of {score}. "
            "To maintain your standing: "
            "1. Keep making timely payments on all loans. "
            "2. Avoid taking on excessive new debt. "
            "3. Maintain a stable income and employment record."
        )
    return (
        f"Your loan was rejected with a credit score of {score}. "
        "To improve your chances: "
        "1. Pay all bills on time to build a strong credit history. "
        "2. Reduce existing debt to improve your debt-to-income ratio. "
        "3. Consider applying for a smaller loan amount or adding a co-applicant."
    )


# ============================================================================
# LIFESPAN EVENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_and_data()
    logger.info("ML Model loaded successfully")
    yield
    logger.info("Shutting down...")


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Loan Approval Prediction API",
    description="ML-powered API with AI financial assistant",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ============================================================================
# SCHEMAS
# ============================================================================

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    password: str = Field(..., min_length=6)

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class LoanApplicationRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "Gender": "Male", "Married": "Yes", "Dependents": "2",
        "Education": "Graduate", "Self_Employed": "No",
        "ApplicantIncome": 50000, "CoapplicantIncome": 0,
        "LoanAmount": 200000, "Loan_Amount_Term": 360,
        "Credit_History": 1.0, "Property_Area": "Urban"
    }})
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str

class FeatureImportance(BaseModel):
    feature: str
    impact: float
    value: float

class PredictionResponse(BaseModel):
    loan_approved: int
    approval_probability: float
    credit_score: int
    top_features: List[FeatureImportance]
    ai_suggestions: str

class HealthResponse(BaseModel):
    status: str
    message: str

class UserQuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)

class UserQuestionResponse(BaseModel):
    answer: str
    context_available: bool

# ============================================================================
# AUTH FUNCTIONS
# ============================================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def authenticate_user(username: str, password: str = None, token_validation: bool = False):
    users_db = {
        "admin": {"username": "admin", "email": "admin@example.com", "password": os.getenv("ADMIN_PASSWORD", "admin123"), "disabled": False},
        "user":  {"username": "user",  "email": "user@example.com",  "password": os.getenv("USER_PASSWORD",  "user123"),  "disabled": False}
    }
    user = users_db.get(username)
    if not user:
        return False
    if token_validation:
        return user
    if user.get("password") != password:
        return False
    return user

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    exc = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise exc
    except JWTError:
        raise exc
    user = authenticate_user(username, token_validation=True)
    if not user:
        raise exc
    return user

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _get_transformed_feature_names(preprocessor) -> list:
    if preprocessor is None:
        return None
    try:
        return [str(n) for n in preprocessor.get_feature_names_out()]
    except Exception:
        pass
    try:
        names = []
        for _, transformer, cols in getattr(preprocessor, "transformers_", []):
            if transformer == "drop":
                continue
            if transformer == "passthrough":
                if cols is None:
                    continue
                names.extend([str(c) for c in cols] if isinstance(cols, (list, tuple)) else [str(cols)])
                continue
            try:
                if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
                    names.extend([str(n) for n in transformer.named_steps["onehot"].get_feature_names_out(cols)])
                    continue
            except Exception:
                pass
            try:
                if hasattr(transformer, "get_feature_names_out"):
                    names.extend([str(n) for n in transformer.get_feature_names_out(cols)])
                    continue
            except Exception:
                pass
            names.extend([str(c) for c in cols] if isinstance(cols, (list, tuple)) else [str(cols)])
        return names or None
    except Exception:
        return None

def _prettify_feature_name(raw: str) -> str:
    if raw is None:
        return raw
    return str(raw).replace("num__", "").replace("cat__", "").replace("onehot__", "")

def load_model_and_data():
    global MODEL, PREPROCESSOR, X_TRAIN, FEATURE_NAMES
    model_path, preprocessor_path, x_train_path = "loan_model.pkl", "loan_preprocessor.pkl", "loan_x_train.pkl"
    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        print("Loading saved model...")
        with open(model_path, "rb") as f:
            MODEL = pickle.load(f)
        with open(preprocessor_path, "rb") as f:
            PREPROCESSOR = pickle.load(f)
        raw_names = _get_transformed_feature_names(PREPROCESSOR)
        FEATURE_NAMES = [_prettify_feature_name(n) for n in raw_names] if raw_names else None
        if os.path.exists(x_train_path):
            with open(x_train_path, "rb") as f:
                X_TRAIN = pickle.load(f)
        else:
            df = load_data("loan_data.csv")
            X_TRAIN, _, _ = preprocess_data(df)
            if hasattr(X_TRAIN, 'toarray'):
                X_TRAIN = X_TRAIN.toarray()
    else:
        print("Training new model...")
        df = load_data("loan_data.csv")
        if df is None:
            raise Exception("Failed to load loan_data.csv")
        X_TRAIN, y_train, PREPROCESSOR = preprocess_data(df)
        raw_names = _get_transformed_feature_names(PREPROCESSOR)
        FEATURE_NAMES = [_prettify_feature_name(n) for n in raw_names] if raw_names else None
        if hasattr(X_TRAIN, 'toarray'):
            X_TRAIN = X_TRAIN.toarray()
        MODEL = train_model(X_TRAIN, y_train)
        with open(model_path, "wb") as f:
            pickle.dump(MODEL, f)
        with open(preprocessor_path, "wb") as f:
            pickle.dump(PREPROCESSOR, f)
        with open(x_train_path, "wb") as f:
            pickle.dump(X_TRAIN, f)
        print("Model trained and saved!")

def ensure_model_loaded():
    global MODEL, PREPROCESSOR
    if MODEL is None or PREPROCESSOR is None:
        load_model_and_data()

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.post("/register", response_model=Token)
async def register_user(user: UserCreate):
    try:
        logger.info(f"Registration: {user.username}")
        token = create_access_token({"sub": user.username}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        return {"access_token": token, "token_type": "bearer"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Registration failed: {e}")

@app.post("/login", response_model=Token)
async def login_user(user: UserLogin):
    try:
        logger.info(f"Login attempt: {user.username}")
        auth_user = authenticate_user(user.username, user.password)
        if not auth_user:
            logger.warning(f"Failed login: {user.username}")
            raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
        token = create_access_token({"sub": auth_user["username"]}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        logger.info(f"Successful login: {user.username}")
        return {"access_token": token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Login failed due to server error")

@app.get("/", response_model=HealthResponse)
async def health_check():
    return {"status": "success", "message": "API Running"}

@app.get("/health")
async def detailed_health():
    return {
        "api_status": "running",
        "model_loaded": MODEL is not None,
        "preprocessor_loaded": PREPROCESSOR is not None,
        "training_samples": X_TRAIN.shape[0] if X_TRAIN is not None else 0,
        "ai_assistant": "HF Inference API" if HF_API_TOKEN else "Fallback (no token)"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_loan(request: LoanApplicationRequest, current_user: dict = Depends(get_current_user)):
    try:
        logger.info(f"Prediction request from: {current_user['username']}")
        ensure_model_loaded()

        global FEATURE_NAMES, LATEST_PREDICTION_CONTEXT
        if FEATURE_NAMES is None:
            raw_names = _get_transformed_feature_names(PREPROCESSOR)
            FEATURE_NAMES = [_prettify_feature_name(n) for n in raw_names] if raw_names else None

        input_df = pd.DataFrame([request.model_dump()])
        X_input = PREPROCESSOR.transform(input_df)
        result = predict_with_credit_score(MODEL, X_input)
        explanation = explain_prediction(model=MODEL, X_input=X_input, X_background=X_TRAIN, feature_names=FEATURE_NAMES)

        approval_label = "Approved" if result['prediction'] == 1 else "Rejected"
        credit_score = result['credit_score']
        top_features = explanation['top_features']

        LATEST_PREDICTION_CONTEXT = {"result": approval_label, "score": credit_score, "features": top_features}

        ai_suggestions = generate_suggestions(result=approval_label, score=credit_score, features=top_features)

        logger.info(f"Prediction done for {current_user['username']}: {approval_label}")
        return {
            "loan_approved": result['prediction'],
            "approval_probability": result['probability'],
            "credit_score": credit_score,
            "top_features": top_features,
            "ai_suggestions": ai_suggestions
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

@app.post("/ask", response_model=UserQuestionResponse)
async def ask_question(request: UserQuestionRequest, current_user: dict = Depends(get_current_user)):
    try:
        logger.info(f"AI question from {current_user['username']}: {request.question}")
        if LATEST_PREDICTION_CONTEXT["result"] is None:
            return {"answer": "Please make a loan prediction first, then I can answer your questions.", "context_available": False}
        answer = answer_user_question(question=request.question, context=LATEST_PREDICTION_CONTEXT)
        return {"answer": answer, "context_available": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Question failed: {e}")

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return {"error": "Invalid input", "details": str(exc)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")