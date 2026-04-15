# AI Credit Scoring System

A full-stack loan approval prediction system with JWT authentication, XGBoost ML model, SHAP explainability, and a FREE AI financial assistant powered by Hugging Face Flan-T5.

## 🌐 Live Demo

| Service | URL |
|---------|-----|
| 🎨 Frontend | [ai-credit-scoring-system-omega.vercel.app](https://ai-credit-scoring-system-omega.vercel.app) |
| ⚙️ Backend API | [ai-credit-scoring-system.onrender.com](https://ai-credit-scoring-system.onrender.com) |
| 📚 API Docs | [ai-credit-scoring-system.onrender.com/docs](https://ai-credit-scoring-system.onrender.com/docs) |

## 🚀 Features

### Backend (FastAPI)
- ✅ **JWT Authentication** - Secure login/registration with token-based auth
- ✅ **XGBoost ML Model** - Gradient boosting for superior accuracy
- ✅ **SHAP Explainability** - Top 3 features affecting each decision
- ✅ **AI Financial Assistant** - Hugging Face Flan-T5 via Inference API (no local model)
- ✅ **POST /ask endpoint** - Users can ask questions about their loan result
- ✅ **Lightweight design** - No torch/transformers installed, fast cold start
- ✅ **Comprehensive Logging** - Request/response logging with error handling
- ✅ **Input Validation** - Pydantic schemas with detailed error messages

### Frontend (React + Tailwind CSS)
- ✅ **Modern UI** - Beautiful, responsive design with Tailwind CSS
- ✅ **Authentication Flow** - Login/register with JWT token management
- ✅ **Real-time Predictions** - Live API integration with loading states
- ✅ **Feature Explanations** - Visual SHAP feature importance bars
- ✅ **AI Suggestions Card** - Numbered, actionable financial advice
- ✅ **AI Q&A Chat** - Ask questions about your loan decision with suggested prompts
- ✅ **Error Handling** - User-friendly error messages and validation

### Machine Learning
- ✅ **XGBoost Classifier** - ~85% accuracy on loan approval dataset
- ✅ **Credit Score Generation** - 300–900 range scoring system
- ✅ **Feature Engineering** - One-hot encoding and mean imputation
- ✅ **Model Persistence** - Auto saves/loads trained models

### AI Assistant (FREE - No Paid APIs)
- ✅ **Hugging Face Inference API** - `google/flan-t5-base` called via REST (no download)
- ✅ **generate_suggestions()** - Personalised advice after every prediction
- ✅ **answer_user_question()** - Context-aware Q&A about loan result
- ✅ **Fallback system** - Rule-based suggestions if API is unavailable
- ✅ **Singleton context** - Latest prediction stored for /ask endpoint

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python |
| ML Model | XGBoost, scikit-learn, SHAP |
| AI Assistant | Hugging Face Inference API (flan-t5-base) |
| Frontend | React, Tailwind CSS, Heroicons |
| Auth | JWT (python-jose) |
| Deployment | Render (backend) + Vercel (frontend) |

## 📦 Installation

### Backend Setup
```bash
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

## 🔐 Authentication

Demo credentials:
- **Username**: `admin` / **Password**: `admin123`
- **Username**: `user` / **Password**: `user123`

## 📡 API Endpoints

### Auth
```
POST /register   - Register new user
POST /login      - Login, returns JWT token
```

### ML + AI
```
POST /predict    - Loan prediction + AI suggestions (requires auth)
POST /ask        - Ask AI about your loan result (requires auth)
GET  /health     - Health check with model and AI status
GET  /           - Basic health check
```

### /predict Request
```json
{
  "Gender": "Male", "Married": "Yes", "Dependents": "0",
  "Education": "Graduate", "Self_Employed": "No",
  "ApplicantIncome": 50000, "CoapplicantIncome": 0,
  "LoanAmount": 200000, "Loan_Amount_Term": 360,
  "Credit_History": 1.0, "Property_Area": "Urban"
}
```

### /predict Response
```json
{
  "loan_approved": 1,
  "approval_probability": 0.92,
  "credit_score": 852,
  "top_features": [
    { "feature": "Credit_History", "impact": 0.25, "value": 1.0 },
    { "feature": "ApplicantIncome", "impact": 0.18, "value": 50000.0 },
    { "feature": "LoanAmount", "impact": 0.12, "value": 200000.0 }
  ],
  "ai_suggestions": "1. Keep making timely payments. 2. Maintain stable income. 3. Avoid excessive debt."
}
```

### /ask Request & Response
```json
// Request
{ "question": "Why was my loan approved?" }

// Response
{
  "answer": "Your loan was approved due to strong credit history and stable income.",
  "context_available": true
}
```

## 🎯 Usage

1. **Login** with demo credentials
2. **Enter loan details** and click Get Prediction
3. **View results** — approval status, credit score, risk tier
4. **Read AI suggestions** — numbered actionable financial advice
5. **Ask questions** — use suggested prompts or type your own
6. **View key factors** — SHAP feature importance bars

## 📁 File Structure

```
.
├── frontend/
│   └── src/
│       └── App.js           # React app with AI suggestions + Q&A UI
├── main.py                  # FastAPI + AI assistant (HF Inference API)
├── loan_approval_model.py   # XGBoost ML model functions
├── loan_data.csv            # Training dataset
├── requirements.txt         # Python dependencies (no torch/transformers)
├── render.yaml              # Render deployment config
├── Procfile                 # Railway deployment config
├── .env.example             # Environment variables template
└── README.md
```

## 🔧 Environment Variables

### Backend (Render)
| Key | Description |
|-----|-------------|
| `SECRET_KEY` | JWT signing secret |
| `ADMIN_PASSWORD` | Admin user password |
| `USER_PASSWORD` | Regular user password |
| `ALLOWED_ORIGINS` | Comma-separated frontend URLs |
| `HF_API_TOKEN` | Hugging Face API token (free at huggingface.co) |

### Frontend (Vercel)
| Key | Description |
|-----|-------------|
| `REACT_APP_API_URL` | Your Render backend URL |

## 🚀 Deployment

### Backend → Render
```bash
# Build: pip install -r requirements.txt
# Start: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Frontend → Vercel
```bash
# Root directory: frontend
# Build: npm install && npm run build
```

### Keep Render Awake (Free)
Use [UptimeRobot](https://uptimerobot.com) to ping your backend every 14 minutes — prevents cold starts on Render's free tier.

## 📝 API Docs

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 📄 License

MIT