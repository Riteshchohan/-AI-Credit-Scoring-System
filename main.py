"""
FastAPI Backend for Loan Approval Prediction System
With smart rule-based AI suggestions (no heavy ML models)
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
# LOGGING
# ============================================================================
 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
 
# ============================================================================
# SECURITY CONFIG
# ============================================================================
 
SECRET_KEY = os.getenv("SECRET_KEY", "change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]
 
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()
 
# ============================================================================
# GLOBAL VARIABLES
# ============================================================================
 
MODEL = None
PREPROCESSOR = None
X_TRAIN = None
FEATURE_NAMES = None
 
# Stores last prediction for /ask endpoint
LATEST_PREDICTION_CONTEXT = {"result": None, "score": None, "features": []}
 
# ============================================================================
# SMART RULE-BASED AI SUGGESTIONS (no heavy model, instant response)
# ============================================================================
 
def generate_suggestions(result: str, score: int, features: list) -> str:
    """
    Generate smart financial suggestions based on loan result,
    credit score, and top SHAP features. Fully rule-based — instant and lightweight.
    """
    suggestions = []
 
    # Analyze top features for personalized advice
    feature_names = [f['feature'].lower() for f in features] if features else []
    feature_impacts = {f['feature'].lower(): f['impact'] for f in features} if features else {}
 
    if result == "Approved":
        intro = f"🎉 Congratulations! Your loan was approved with a credit score of {score}."
        suggestions.append("✅ Keep making all payments on time to maintain your strong credit profile.")
 
        if score < 750:
            suggestions.append("📈 Your score can still improve — pay down existing debts to push above 750.")
        else:
            suggestions.append("💎 Excellent score! Consider negotiating a lower interest rate with your lender.")
 
        if any('income' in f for f in feature_names):
            suggestions.append("💼 Your income was a key factor — maintaining stable employment will help future applications.")
        else:
            suggestions.append("🏦 Diversify savings and maintain an emergency fund of 3-6 months of expenses.")
 
    else:
        intro = f"❌ Your loan was not approved. Your current credit score is {score}."
 
        # Credit history advice
        if any('credit' in f for f in feature_names):
            neg_impact = any(feature_impacts.get(f, 0) < 0 for f in feature_names if 'credit' in f)
            if neg_impact:
                suggestions.append("🔑 Credit history is your biggest obstacle — start by paying ALL bills on time for 6+ months.")
            else:
                suggestions.append("📊 Build your credit history by getting a secured credit card and using it responsibly.")
 
        # Income advice
        if any('income' in f for f in feature_names):
            suggestions.append("💰 Low income vs loan amount is a concern — consider applying for a smaller loan or adding a co-applicant with higher income.")
        else:
            suggestions.append("💰 Increase your income or reduce the loan amount to improve your debt-to-income ratio.")
 
        # Score-based advice
        if score < 500:
            suggestions.append("⚠️ Score below 500: Focus on clearing any defaults or late payments before reapplying in 6 months.")
        elif score < 650:
            suggestions.append("📅 Score between 500-650: Consistent on-time payments for 3-6 months can significantly boost your score.")
        else:
            suggestions.append("🔄 Score near approval range — a small loan amount reduction or co-applicant may tip the decision in your favor.")
 
    return f"{intro}\n\n💡 Personalized Suggestions:\n" + "\n".join(suggestions)
 
 
def answer_user_question(question: str, context: dict) -> str:
    """
    Answer user questions about their loan using rule-based logic.
    Fast, reliable, no external dependencies.
    """
    result = context.get("result", "Unknown")
    score = context.get("score", 0)
    features = context.get("features", [])
    q = question.lower()
 
    # Score-related questions
    if any(word in q for word in ["score", "credit score", "number"]):
        rating = "excellent" if score >= 750 else "good" if score >= 650 else "fair" if score >= 550 else "poor"
        return f"Your credit score is {score}/900, which is considered {rating}. Scores above 700 generally have higher approval rates."
 
    # Approval/rejection reason
    if any(word in q for word in ["why", "reason", "rejected", "approved", "denied"]):
        if features:
            top = features[0]['feature']
            impact = features[0]['impact']
            direction = "positively" if impact > 0 else "negatively"
            return f"Your loan was {result} mainly because '{top}' influenced the decision {direction} (impact: {impact:.3f}). Other key factors were: {', '.join([f['feature'] for f in features[1:3]])}."
        return f"Your loan was {result} based on a combination of your credit history, income, and loan amount."
 
    # Improvement questions
    if any(word in q for word in ["improve", "increase", "better", "boost", "raise"]):
        if score < 650:
            return "To improve your chances: 1) Pay all bills on time for 6+ months, 2) Reduce existing debt, 3) Avoid applying for multiple loans simultaneously."
        return "To improve further: 1) Keep credit utilization below 30%, 2) Maintain stable employment, 3) Consider paying down existing loans."
 
    # Feature-specific questions
    if any(word in q for word in ["income", "salary", "earning"]):
        return "Income is a major factor. Lenders look at your debt-to-income ratio. A higher income relative to your loan amount improves approval chances."
 
    if any(word in q for word in ["history", "credit history", "past"]):
        return "Credit history shows your track record of repaying debts. A score of 1.0 (good history) significantly increases approval probability."
 
    if any(word in q for word in ["amount", "loan amount", "borrow"]):
        return "A smaller loan amount relative to your income improves approval chances. Try reducing the loan amount by 10-20% if rejected."
 
    if any(word in q for word in ["reapply", "apply again", "try again", "when"]):
        if result == "Rejected":
            return "We recommend waiting 3-6 months before reapplying. Use that time to improve your credit score and reduce existing debts."
        return "Your loan is approved — no need to reapply. Contact your lender to proceed with the next steps."
 
    # Default response
    return f"Your loan was {result} with a credit score of {score}. The top factors were: {', '.join([f['feature'] for f in features[:3]])}. Ask me about your score, reasons, or how to improve!"
 
 
# ============================================================================
# LIFESPAN
# ============================================================================
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_and_data()
    logger.info("Model loaded successfully")
    yield
    logger.info("Shutting down...")
 
# ============================================================================
# APP INIT
# ============================================================================
 
app = FastAPI(
    title="Loan Approval Prediction API",
    description="ML-powered API with smart AI financial assistant",
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
    model_config = ConfigDict(json_schema_extra={"example": {"Gender": "Male", "Married": "Yes", "Dependents": "2", "Education": "Graduate", "Self_Employed": "No", "ApplicantIncome": 50000, "CoapplicantIncome": 0, "LoanAmount": 200000, "Loan_Amount_Term": 360, "Credit_History": 1.0, "Property_Area": "Urban"}})
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
        "user": {"username": "user", "email": "user@example.com", "password": os.getenv("USER_PASSWORD", "user123"), "disabled": False}
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
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = authenticate_user(token_data.username, token_validation=True)
    if user is False:
        raise credentials_exception
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
        for name, transformer, cols in getattr(preprocessor, "transformers_", []):
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
        print("Loading saved model and preprocessor...")
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
            if df is None:
                raise Exception("Failed to load loan_data.csv")
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
    global MODEL, PREPROCESSOR, X_TRAIN, FEATURE_NAMES
    if MODEL is None or PREPROCESSOR is None:
        load_model_and_data()
 
# ============================================================================
# ENDPOINTS
# ============================================================================
 
@app.post("/register", response_model=Token)
async def register_user(user: UserCreate):
    try:
        logger.info(f"User registration attempt: {user.username}")
        access_token = create_access_token(data={"sub": user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Registration failed: {str(e)}")
 
@app.post("/login", response_model=Token)
async def login_user(user: UserLogin):
    try:
        logger.info(f"Login attempt: {user.username}")
        authenticated_user = authenticate_user(user.username, user.password)
        if not authenticated_user:
            logger.warning(f"Failed login attempt: {user.username}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
        access_token = create_access_token(data={"sub": authenticated_user["username"]}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        logger.info(f"Successful login: {user.username}")
        return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error for {user.username}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Login failed due to server error")
 
@app.get("/", response_model=HealthResponse)
async def health_check():
    return {"status": "success", "message": "API Running"}
 
@app.get("/health")
async def detailed_health():
    return {
        "api_status": "running",
        "model_loaded": MODEL is not None,
        "preprocessor_loaded": PREPROCESSOR is not None,
        "training_samples": X_TRAIN.shape[0] if X_TRAIN is not None else 0
    }
 
@app.post("/predict", response_model=PredictionResponse)
async def predict_loan(request: LoanApplicationRequest, current_user: dict = Depends(get_current_user)):
    try:
        logger.info(f"Prediction request from user: {current_user['username']}")
        ensure_model_loaded()
        global FEATURE_NAMES, LATEST_PREDICTION_CONTEXT
        if FEATURE_NAMES is None:
            raw_names = _get_transformed_feature_names(PREPROCESSOR)
            FEATURE_NAMES = [_prettify_feature_name(n) for n in raw_names] if raw_names else None
        input_data = request.model_dump()
        input_df = pd.DataFrame([input_data])
        X_input = PREPROCESSOR.transform(input_df)
        result = predict_with_credit_score(MODEL, X_input)
        explanation = explain_prediction(model=MODEL, X_input=X_input, X_background=X_TRAIN, feature_names=FEATURE_NAMES)
        approval_label = "Approved" if result['prediction'] == 1 else "Rejected"
        credit_score = result['credit_score']
        top_features = explanation['top_features']
        LATEST_PREDICTION_CONTEXT = {"result": approval_label, "score": credit_score, "features": top_features}
        ai_suggestions = generate_suggestions(result=approval_label, score=credit_score, features=top_features)
        logger.info(f"Prediction completed for user {current_user['username']}: approved={result['prediction']}")
        return {"loan_approved": result['prediction'], "approval_probability": result['probability'], "credit_score": credit_score, "top_features": top_features, "ai_suggestions": ai_suggestions}
    except Exception as e:
        logger.error(f"Prediction failed for user {current_user['username']}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
 
@app.post("/ask", response_model=UserQuestionResponse)
async def ask_question(request: UserQuestionRequest, current_user: dict = Depends(get_current_user)):
    try:
        logger.info(f"Question from user {current_user['username']}: {request.question}")
        context_available = LATEST_PREDICTION_CONTEXT["result"] is not None
        if not context_available:
            return {"answer": "Please make a loan prediction first, then I can answer questions about your results.", "context_available": False}
        answer = answer_user_question(question=request.question, context=LATEST_PREDICTION_CONTEXT)
        return {"answer": answer, "context_available": True}
    except Exception as e:
        logger.error(f"Question failed for user {current_user['username']}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Question answering failed: {str(e)}")
 
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return {"error": "Invalid input", "details": str(exc)}
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
 
