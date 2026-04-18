"""
FastAPI Backend — AI Credit Scoring & Loan Approval System
Lightweight backend using Hugging Face Inference API for AI suggestions.
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
    load_data, preprocess_data, train_model,
    predict_with_credit_score, explain_prediction
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001"
)
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# Hugging Face Inference API token
# Models are defined inside query_hf_api() with retry logic across flan-t5-large → flan-t5-base
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# ── Globals ───────────────────────────────────────────────────────────────────
MODEL = None
PREPROCESSOR = None
X_TRAIN = None
FEATURE_NAMES = None
LATEST_PREDICTION_CONTEXT = {"result": None, "score": None, "features": []}

# ── AI Assistant ──────────────────────────────────────────────────────────────
# Strategy: Try HF Inference API with retry logic.
# If unavailable, fall back to a rich rule-based system that gives real answers.
# Models tried in order: flan-t5-large → flan-t5-base → rule-based
# flan-t5 is text2text: always warm, no instruct tags needed, responds fast.

HF_MODELS = [
    "https://api-inference.huggingface.co/models/google/flan-t5-large",
    "https://api-inference.huggingface.co/models/google/flan-t5-base",
]

import time

def query_hf_api(prompt: str) -> str:
    """
    Query HF Inference API with retry logic across multiple models.
    Uses flan-t5 (text2text) — always-on, no instruct format needed.
    Retries up to 2 times per model on 503 (model loading).
    """
    if not HF_API_TOKEN:
        logger.warning("HF_API_TOKEN not set — using rule-based fallback.")
        return None

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200, "temperature": 0.7}
    }

    for model_url in HF_MODELS:
        for attempt in range(3):  # 3 attempts per model
            try:
                response = requests.post(model_url, headers=headers, json=payload, timeout=25)

                if response.status_code == 200:
                    data = response.json()
                    # flan-t5 returns: [{"generated_text": "..."}]
                    if isinstance(data, list) and data:
                        text = data[0].get("generated_text", "").strip()
                        if text and len(text) > 15:
                            logger.info(f"HF API success with {model_url.split('/')[-1]}")
                            return text
                    logger.warning(f"HF returned empty text (model: {model_url.split('/')[-1]})")
                    break  # empty response — try next model

                elif response.status_code == 503:
                    wait = 3 * (attempt + 1)  # 3s, 6s, 9s
                    logger.warning(f"Model loading (attempt {attempt+1}/3), waiting {wait}s...")
                    time.sleep(wait)
                    continue  # retry same model

                elif response.status_code == 401:
                    logger.error("HF API: Invalid token. Check HF_API_TOKEN in Render env vars.")
                    return None  # No point retrying with wrong token

                else:
                    logger.warning(f"HF API status {response.status_code}: {response.text[:150]}")
                    break  # unexpected error — try next model

            except requests.Timeout:
                logger.warning(f"HF API timeout (attempt {attempt+1}/3, model: {model_url.split('/')[-1]})")
                if attempt < 2:
                    time.sleep(2)
                    continue
                break
            except Exception as e:
                logger.error(f"HF API exception: {e}")
                break

    logger.warning("All HF models failed — using rule-based fallback.")
    return None


def generate_suggestions(result: str, score: int, features: list) -> str:
    """
    Generate personalised financial advice after a loan decision.
    Tries HF API first; always returns a meaningful answer regardless.
    """
    try:
        top_feature = features[0]["feature"].replace("_", " ") if features else "credit history"
        feature_list = ", ".join(
            [f["feature"].replace("_", " ") for f in features[:3]]
        ) if features else "credit history, income, loan amount"
        decision = "approved" if result == "Approved" else "rejected"

        # flan-t5 works best with direct instruction prompts
        prompt = (
            f"A loan application was {decision}. Credit score: {score} out of 900. "
            f"Key factors: {feature_list}. "
            f"Give 3 numbered actionable tips to improve loan approval chances."
        )

        answer = query_hf_api(prompt)
        if answer and len(answer.strip()) > 20:
            return answer.strip()

        # Always return a real, helpful answer as fallback
        return _smart_suggestions(result, score, features)

    except Exception as e:
        logger.error(f"generate_suggestions failed: {e}")
        return _smart_suggestions(result, score, features)


def answer_user_question(question: str, context: dict) -> str:
    """
    Answer a user's question about their loan result.
    HF API first, then a smart rule-based system that gives real answers.
    """
    try:
        result = context.get("result", "Unknown")
        score = context.get("score", "N/A")
        features = context.get("features", [])
        feature_list = ", ".join(
            [f["feature"].replace("_", " ") for f in features[:3]]
        ) if features else "credit history, income, loan amount"
        decision = "approved" if result == "Approved" else "rejected"

        prompt = (
            f"Loan was {decision}. Credit score: {score}/900. "
            f"Key factors: {feature_list}. "
            f"Question: {question} "
            f"Answer in 2 sentences."
        )

        answer = query_hf_api(prompt)
        if answer and len(answer.strip()) > 20:
            return answer.strip()

        # Smart fallback — actually answers the question properly
        return _smart_qa(question, result, score, features)

    except Exception as e:
        logger.error(f"answer_user_question failed: {e}")
        return _smart_qa(question, context.get("result", "Unknown"),
                         context.get("score", 0), context.get("features", []))


def _smart_suggestions(result: str, score: int, features: list) -> str:
    """
    Rich rule-based suggestions that give genuinely useful advice
    based on the actual prediction result and top features.
    """
    top_features = [f["feature"].lower() for f in features[:3]] if features else []

    if result == "Approved":
        tip3 = "Consider setting up automatic payments to protect your strong repayment record going forward."
        if any("income" in f for f in top_features):
            tip3 = "Your income was a strong positive factor — maintaining or growing it will keep you in good standing."
        return (
            f"1. Great news — your credit score of {score} helped secure this approval. "
            f"Keep paying bills on time to maintain this advantage.\n"
            f"2. Avoid opening new credit cards or loans in the next 3 months, "
            f"as multiple credit inquiries can lower your score.\n"
            f"3. {tip3}"
        )

    # Rejected — give targeted advice based on what actually hurt them
    tips = []
    if any("credit" in f for f in top_features):
        tips.append(
            "Your credit history was the biggest factor — start building it by paying every bill on time, "
            "even small ones. Consistency over 6–12 months makes a real difference."
        )
    if any("income" in f or "applicant" in f for f in top_features):
        tips.append(
            "Your income relative to the loan amount was flagged. Consider applying for a smaller loan, "
            "adding a co-applicant with steady income, or waiting until your income increases."
        )
    if any("loan" in f for f in top_features):
        tips.append(
            "The loan amount requested was identified as a risk factor. "
            "A smaller loan amount or a longer repayment term would reduce the lender's perceived risk."
        )

    # Fill up to 3 tips with general advice if needed
    general = [
        "Clear any outstanding debts where possible — a lower debt-to-income ratio significantly improves approval odds.",
        "Check your credit report for any errors or outdated negative entries and dispute them if needed.",
        "Wait at least 6 months before reapplying, using that time to strengthen your financial profile.",
    ]
    for g in general:
        if len(tips) >= 3:
            break
        tips.append(g)

    numbered = "\n".join([f"{i+1}. {t}" for i, t in enumerate(tips[:3])])
    return f"Your score of {score} needs improvement. Here's what to focus on:\n{numbered}"


def _smart_qa(question: str, result: str, score, features: list) -> str:
    """
    Smart rule-based Q&A that actually answers common questions properly.
    Matches question intent and returns a relevant, specific answer.
    """
    q = question.lower()
    decision = "approved" if result == "Approved" else "rejected"
    top_features = [f["feature"].replace("_", " ").lower() for f in features[:3]] if features else []
    top_name = top_features[0] if top_features else "credit history"

    # Why approved/rejected?
    if any(w in q for w in ["why", "reason", "because", "what caused", "what made"]):
        if result == "Approved":
            return (
                f"Your loan was approved mainly because of your strong {top_name}, "
                f"which pushed your credit score to {score}/900. "
                f"Lenders saw you as a low-risk borrower based on your financial profile."
            )
        return (
            f"Your loan was rejected largely due to concerns around your {top_name}. "
            f"With a score of {score}/900, the model identified your application as higher risk. "
            f"Improving your {top_name} over the next few months would meaningfully increase your chances."
        )

    # How to improve score?
    if any(w in q for w in ["improve", "increase", "boost", "better", "raise", "higher"]):
        return (
            f"To raise your credit score from {score}, focus on three things: "
            f"pay every bill on time without exception, reduce any existing debt balances, "
            f"and avoid applying for new credit in the short term. "
            f"Consistent positive behaviour over 6–12 months typically leads to noticeable improvement."
        )

    # What is the most important factor?
    if any(w in q for w in ["important", "factor", "key", "main", "biggest", "top"]):
        second = top_features[1] if len(top_features) > 1 else "income level"
        return (
            f"The most influential factor in your result was your {top_name}, "
            f"followed closely by your {second}. "
            f"These two factors carried the most weight in the model's decision."
        )

    # What is credit score?
    if any(w in q for w in ["credit score", "what is", "mean", "range", "300", "900"]):
        tier = "good" if score >= 700 else "fair" if score >= 550 else "needs improvement"
        return (
            f"Your credit score is {score} out of 900. That's considered {tier}. "
            f"Scores above 700 typically qualify for the best loan terms, "
            f"while scores below 550 usually lead to rejection unless other factors are very strong."
        )

    # What should I do next?
    if any(w in q for w in ["next", "now", "do", "step", "action", "reapply"]):
        if result == "Approved":
            return (
                f"Your loan was approved — the next step is to review the loan terms carefully "
                f"and ensure the repayment schedule fits your monthly budget. "
                f"Making every payment on time will also keep your score of {score} healthy."
            )
        return (
            f"Start by working on your {top_name} over the next 3–6 months. "
            f"Once your score improves past 650, consider reapplying — "
            f"you'll likely see a much better outcome."
        )

    # Default — still give a real answer
    return (
        f"Your loan was {decision} with a credit score of {score}/900. "
        f"The top factors influencing this were: {', '.join(top_features) if top_features else 'credit history, income, and loan amount'}. "
        f"Feel free to ask something more specific — like why it was {decision}, "
        f"how to improve your score, or what the most important factor was."
    )


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_and_data()
    logger.info("ML model loaded successfully")
    yield
    logger.info("Shutting down...")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Credit Scoring API",
    description="ML-powered loan approval prediction with AI financial assistant",
    version="3.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
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

# ── Auth ──────────────────────────────────────────────────────────────────────
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def authenticate_user(username: str, password: str = None, token_validation: bool = False):
    users_db = {
        "admin": {"username": "admin", "email": "admin@example.com",
                  "password": os.getenv("ADMIN_PASSWORD", "admin123"), "disabled": False},
        "user":  {"username": "user",  "email": "user@example.com",
                  "password": os.getenv("USER_PASSWORD",  "user123"),  "disabled": False},
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
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"}
    )
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

# ── Utilities ─────────────────────────────────────────────────────────────────
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

def _prettify(raw: str) -> str:
    if raw is None:
        return raw
    return str(raw).replace("num__", "").replace("cat__", "").replace("onehot__", "")

def load_model_and_data():
    global MODEL, PREPROCESSOR, X_TRAIN, FEATURE_NAMES
    mp, pp, xp = "loan_model.pkl", "loan_preprocessor.pkl", "loan_x_train.pkl"
    if os.path.exists(mp) and os.path.exists(pp):
        print("Loading saved model...")
        with open(mp, "rb") as f: MODEL = pickle.load(f)
        with open(pp, "rb") as f: PREPROCESSOR = pickle.load(f)
        raw = _get_transformed_feature_names(PREPROCESSOR)
        FEATURE_NAMES = [_prettify(n) for n in raw] if raw else None
        if os.path.exists(xp):
            with open(xp, "rb") as f: X_TRAIN = pickle.load(f)
        else:
            df = load_data("loan_data.csv")
            X_TRAIN, _, _ = preprocess_data(df)
            if hasattr(X_TRAIN, 'toarray'): X_TRAIN = X_TRAIN.toarray()
    else:
        print("Training new model...")
        df = load_data("loan_data.csv")
        if df is None: raise Exception("Failed to load loan_data.csv")
        X_TRAIN, y_train, PREPROCESSOR = preprocess_data(df)
        raw = _get_transformed_feature_names(PREPROCESSOR)
        FEATURE_NAMES = [_prettify(n) for n in raw] if raw else None
        if hasattr(X_TRAIN, 'toarray'): X_TRAIN = X_TRAIN.toarray()
        MODEL = train_model(X_TRAIN, y_train)
        with open(mp, "wb") as f: pickle.dump(MODEL, f)
        with open(pp, "wb") as f: pickle.dump(PREPROCESSOR, f)
        with open(xp, "wb") as f: pickle.dump(X_TRAIN, f)
        print("Model trained and saved!")

def ensure_model_loaded():
    global MODEL, PREPROCESSOR
    if MODEL is None or PREPROCESSOR is None:
        load_model_and_data()

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/register", response_model=Token)
async def register_user(user: UserCreate):
    try:
        token = create_access_token({"sub": user.username}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        return {"access_token": token, "token_type": "bearer"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Registration failed: {e}")

@app.post("/login", response_model=Token)
async def login_user(user: UserLogin):
    try:
        logger.info(f"Login attempt: {user.username}")
        auth = authenticate_user(user.username, user.password)
        if not auth:
            logger.warning(f"Failed login: {user.username}")
            raise HTTPException(status_code=401, detail="Incorrect username or password",
                                headers={"WWW-Authenticate": "Bearer"})
        token = create_access_token({"sub": auth["username"]}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        logger.info(f"Successful login: {user.username}")
        return {"access_token": token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Login failed due to a server error")

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
        "ai_assistant": "Mistral-7B via HF Inference API" if HF_API_TOKEN else "Rule-based fallback (no token set)"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_loan(request: LoanApplicationRequest, current_user: dict = Depends(get_current_user)):
    try:
        logger.info(f"Prediction request from: {current_user['username']}")
        ensure_model_loaded()

        global FEATURE_NAMES, LATEST_PREDICTION_CONTEXT
        if FEATURE_NAMES is None:
            raw = _get_transformed_feature_names(PREPROCESSOR)
            FEATURE_NAMES = [_prettify(n) for n in raw] if raw else None

        input_df = pd.DataFrame([request.model_dump()])
        X_input = PREPROCESSOR.transform(input_df)
        result = predict_with_credit_score(MODEL, X_input)
        explanation = explain_prediction(
            model=MODEL, X_input=X_input,
            X_background=X_TRAIN, feature_names=FEATURE_NAMES
        )

        approval_label = "Approved" if result['prediction'] == 1 else "Rejected"
        credit_score = result['credit_score']
        top_features = explanation['top_features']

        LATEST_PREDICTION_CONTEXT = {
            "result": approval_label, "score": credit_score, "features": top_features
        }

        ai_suggestions = generate_suggestions(
            result=approval_label, score=credit_score, features=top_features
        )

        logger.info(f"Prediction done for {current_user['username']}: {approval_label} (score {credit_score})")
        return {
            "loan_approved": result['prediction'],
            "approval_probability": result['probability'],
            "credit_score": credit_score,
            "top_features": top_features,
            "ai_suggestions": ai_suggestions,
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
            return {
                "answer": "It looks like you haven't run a prediction yet. "
                          "Submit your loan details first and then ask me anything about your result.",
                "context_available": False
            }
        answer = answer_user_question(
            question=request.question,
            context=LATEST_PREDICTION_CONTEXT
        )
        return {"answer": answer, "context_available": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Question failed: {e}")

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return {"error": "Invalid input", "details": str(exc)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")