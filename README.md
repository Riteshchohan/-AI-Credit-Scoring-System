# AI Credit Scoring System

A comprehensive loan approval prediction system with JWT authentication, modern UI, XGBoost-powered ML model, and a FREE AI financial assistant powered by Hugging Face Flan-T5.

## 🌐 Live Demo

| Service | URL |
|---------|-----|
| 🎨 Frontend | [ai-credit-scoring-system-omega.vercel.app](https://ai-credit-scoring-system-omega.vercel.app) |
| ⚙️ Backend API | [ai-credit-scoring-system.onrender.com](https://ai-credit-scoring-system.onrender.com) |
| 📚 API Docs | [ai-credit-scoring-system.onrender.com/docs](https://ai-credit-scoring-system.onrender.com/docs) |

## 🚀 Features

### Backend (FastAPI)
- ✅ **JWT Authentication** - Secure login/registration with token-based auth
- ✅ **XGBoost ML Model** - Improved accuracy over RandomForest
- ✅ **SHAP Explainability** - Feature importance for predictions
- ✅ **AI Financial Assistant** - FREE Hugging Face Flan-T5 powered suggestions
- ✅ **Q&A Endpoint** - Ask questions about your loan decision
- ✅ **Comprehensive Logging** - Request/response logging with error handling
- ✅ **Input Validation** - Pydantic schemas with detailed error messages
- ✅ **Model Persistence** - Automatic saving/loading of trained models

### Frontend (React + Tailwind CSS)
- ✅ **Modern UI** - Beautiful, responsive design with Tailwind CSS
- ✅ **Authentication Flow** - Login/register forms with JWT token management
- ✅ **Real-time Predictions** - Live API integration with loading states
- ✅ **Feature Explanations** - Visual display of SHAP feature importance
- ✅ **AI Suggestions Display** - Shows AI-generated financial advice
- ✅ **Error Handling** - User-friendly error messages and validation

### Machine Learning
- ✅ **XGBoost Classifier** - Gradient boosting for superior accuracy
- ✅ **Credit Score Generation** - 300-900 range scoring system
- ✅ **Feature Engineering** - One-hot encoding and imputation
- ✅ **Model Evaluation** - Comprehensive metrics (accuracy, precision, recall)

### AI Assistant (FREE - No Paid APIs)
- ✅ **Hugging Face Flan-T5** - google/flan-t5-base (fallback: flan-t5-small)
- ✅ **Financial Suggestions** - Actionable advice after every prediction
- ✅ **Q&A System** - Answer user questions about their loan decision
- ✅ **Singleton Pattern** - Model loads only once for performance
- ✅ **Fallback System** - Rule-based suggestions if AI model unavailable

## 🛠️ Tech Stack

- **Backend**: FastAPI, XGBoost, SHAP, scikit-learn, Python
- **AI Assistant**: Hugging Face Transformers, google/flan-t5-base
- **Frontend**: React, Tailwind CSS, JavaScript
- **Authentication**: JWT (JSON Web Tokens)
- **ML**: XGBoost, pandas, numpy
- **Deployment**: Render (backend) + Vercel (frontend)

## 📦 Installation

### Backend Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

## 🔐 Authentication

The system uses JWT authentication. Demo credentials:
- **Username**: `admin` / **Password**: `admin123`
- **Username**: `user` / **Password**: `user123`

## 📡 API Endpoints

### Authentication
```bash
# Register new user
POST /register
{
  "username": "newuser",
  "email": "user@example.com",
  "password": "securepass"
}

# Login
POST /login
{
  "username": "admin",
  "password": "admin123"
}
```

### Predictions (Requires Authentication)
```bash
# Make prediction
POST /predict
Authorization: Bearer <jwt_token>
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "0",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 50000,
  "CoapplicantIncome": 0,
  "LoanAmount": 200000,
  "Loan_Amount_Term": 360,
  "Credit_History": 1.0,
  "Property_Area": "Urban"
}
```

**Response:**
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
  "ai_suggestions": "Your loan was approved. To maintain your standing: 1. Keep making timely payments. 2. Avoid excessive new debt. 3. Maintain stable income."
}
```

### AI Q&A (Requires Authentication)
```bash
# Ask a question about your loan result
POST /ask
Authorization: Bearer <jwt_token>
{
  "question": "Why was my loan approved and how can I get a better credit score?"
}
```

**Response:**
```json
{
  "answer": "Your loan was approved due to a strong credit history and stable income. To improve your credit score further, continue making timely payments and keep your debt-to-income ratio low.",
  "context_available": true
}
```

### Health Check
```bash
GET /health
```
```json
{
  "api_status": "running",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "training_samples": 614,
  "ai_assistant_loaded": true
}
```

## 🎯 Usage

1. **Start Backend**: `uvicorn main:app --reload --host 127.0.0.1 --port 8000`
2. **Start Frontend**: `cd frontend && npm start`
3. **Login** with demo credentials
4. **Enter loan details** and get instant predictions
5. **View AI suggestions** for personalized financial advice
6. **Ask questions** about your loan decision via `/ask`

## 📊 Model Performance

XGBoost model trained on loan approval dataset:
- **Accuracy**: ~85%
- **Precision**: High for loan approvals
- **Recall**: Balanced for both classes
- **Features**: Credit history, income, loan amount, etc.

## 🤖 AI Assistant Details

The AI assistant uses Google's Flan-T5 model from Hugging Face (completely free, no API key needed):

- **Model**: `google/flan-t5-base` (fallback: `google/flan-t5-small`)
- **Task**: Text-to-text generation
- **Loaded**: Once at startup (singleton pattern for performance)
- **Endpoints**: `/predict` (suggestions) and `/ask` (Q&A)
- **Fallback**: Rule-based suggestions if model unavailable

## 📁 File Structure

```
.
├── frontend/                    # React frontend app
│   ├── package.json
│   ├── public/
│   └── src/
│       └── App.js               # Main React component
├── main.py                      # FastAPI application + AI assistant
├── loan_approval_model.py       # ML model functions
├── loan_data.csv                # Training data
├── requirements.txt             # Python dependencies
├── render.yaml                  # Render deployment config
├── Procfile                     # Railway deployment config
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🚀 Deployment

### Backend (Render)
1. Push the project to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables:

| Key | Value |
|-----|-------|
| `SECRET_KEY` | your-long-random-secret |
| `ADMIN_PASSWORD` | your-admin-password |
| `USER_PASSWORD` | your-user-password |
| `ALLOWED_ORIGINS` | your-vercel-frontend-url |

### Frontend (Vercel)
1. Go to [vercel.com](https://vercel.com) → Add New Project
2. Import your GitHub repo
3. Set root directory to `frontend`
4. Add environment variable:

| Key | Value |
|-----|-------|
| `REACT_APP_API_URL` | your-render-backend-url |

5. Deploy

## 🔧 Environment Variables

Copy `.env.example` to `.env` and fill in your values:
```bash
SECRET_KEY=your-super-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30
ADMIN_PASSWORD=your-secure-password
USER_PASSWORD=your-secure-password
ALLOWED_ORIGINS=https://your-app.vercel.app
```

## 📝 API Documentation

Access interactive API docs at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Input Fields

| Field | Type | Example |
|-------|------|---------|
| `Gender` | string | "Male" / "Female" |
| `Married` | string | "Yes" / "No" |
| `Dependents` | string | "0" / "1" / "2" / "3+" |
| `Education` | string | "Graduate" / "Undergraduate" |
| `Self_Employed` | string | "Yes" / "No" |
| `ApplicantIncome` | float | 50000 |
| `CoapplicantIncome` | float | 0 |
| `LoanAmount` | float | 200000 |
| `Loan_Amount_Term` | float | 360 |
| `Credit_History` | float | 1.0 (0.0 or 1.0) |
| `Property_Area` | string | "Urban" / "Semiurban" / "Rural" |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details
