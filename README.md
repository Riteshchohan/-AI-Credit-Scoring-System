# AI Credit Scoring System

A comprehensive loan approval prediction system with JWT authentication, modern UI, and XGBoost-powered ML model.

## 🌐 Live Demo

| Service | URL |
|---------|-----|
| 🎨 Frontend | [ai-credit-scoring-system-rho.vercel.app](https://ai-credit-scoring-system-rho.vercel.app) |
| ⚙️ Backend API | [ai-credit-scoring-system.onrender.com](https://ai-credit-scoring-system.onrender.com) |
| 📚 API Docs | [ai-credit-scoring-system.onrender.com/docs](https://ai-credit-scoring-system.onrender.com/docs) |

## 🚀 Features

### Backend (FastAPI)
- ✅ **JWT Authentication** - Secure login/registration with token-based auth
- ✅ **XGBoost ML Model** - Improved accuracy over RandomForest
- ✅ **SHAP Explainability** - Feature importance for predictions
- ✅ **Comprehensive Logging** - Request/response logging with error handling
- ✅ **Input Validation** - Pydantic schemas with detailed error messages
- ✅ **Model Persistence** - Automatic saving/loading of trained models

### Frontend (React + Tailwind CSS)
- ✅ **Modern UI** - Beautiful, responsive design with Tailwind CSS
- ✅ **Authentication Flow** - Login/register forms with JWT token management
- ✅ **Real-time Predictions** - Live API integration with loading states
- ✅ **Feature Explanations** - Visual display of SHAP feature importance
- ✅ **Error Handling** - User-friendly error messages and validation

### Machine Learning
- ✅ **XGBoost Classifier** - Gradient boosting for superior accuracy
- ✅ **Credit Score Generation** - 300-900 range scoring system
- ✅ **Feature Engineering** - One-hot encoding and imputation
- ✅ **Model Evaluation** - Comprehensive metrics (accuracy, precision, recall)

## 🛠️ Tech Stack

- **Backend**: FastAPI, XGBoost, SHAP, scikit-learn, Python
- **Frontend**: React, Tailwind CSS, JavaScript
- **Authentication**: JWT (JSON Web Tokens)
- **ML**: XGBoost, pandas, numpy
- **Deployment**: Ready for Render/Railway (backend) and Vercel (frontend)

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

### API Endpoints

#### Authentication
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

#### Predictions (Requires Authentication)
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

## 🎯 Usage

1. **Start Backend**: `uvicorn main:app --reload --host 127.0.0.1 --port 8000`
2. **Start Frontend**: `cd frontend && npm start`
3. **Login** with demo credentials
4. **Enter loan details** and get instant predictions
5. **View explanations** for model decisions

## 📊 Model Performance

XGBoost model trained on loan approval dataset:
- **Accuracy**: ~85%
- **Precision**: High for loan approvals
- **Recall**: Balanced for both classes
- **Features**: Credit history, income, loan amount, etc.

## 🚀 Deployment

### Backend (Render/Railway)
```bash
# Set environment variables
SECRET_KEY=your-production-secret-key
# ... other config

# Deploy command
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Frontend (Vercel)
```bash
# Build for production
npm run build

# Update API_URL in production build
# Set environment variable: REACT_APP_API_URL=https://your-api-domain.com
```

## 🔧 Improvements Made

### 1. JWT Authentication
- Added secure login/registration endpoints
- Protected prediction API with Bearer token authentication
- Token expiration and validation

### 2. Modern UI with Tailwind CSS
- Replaced basic CSS with Tailwind utility classes
- Responsive grid layouts and modern design
- Improved form styling and user experience
- Added loading states and error handling

### 3. Enhanced Logging & Error Handling
- Comprehensive request/response logging
- Structured error responses
- Graceful failure handling in ML pipeline
- User-friendly error messages

### 4. XGBoost Model Upgrade
- Replaced RandomForest with XGBoost for better accuracy
- Improved hyperparameters (n_estimators=200, max_depth=6)
- Better handling of feature importance with SHAP
- Enhanced model evaluation metrics

## 📝 API Documentation

Access interactive API docs at: `http://localhost:8000/docs`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details
```json
{
  "status": "success",
  "message": "API Running"
}
```

### 2. Detailed Health Status
**GET** `/health`
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "api_status": "running",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "training_samples": 614
}
```

### 3. Predict Loan Approval
**POST** `/predict`

**Request Example:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Response:**
```json
{
  "loan_approved": 1,
  "approval_probability": 0.92,
  "credit_score": 852,
  "top_features": [
    {
      "feature": "Feature_8",
      "impact": 0.25,
      "value": 1.0
    },
    {
      "feature": "Feature_3",
      "impact": 0.18,
      "value": 1.0
    },
    {
      "feature": "Feature_0",
      "impact": 0.12,
      "value": 50000.0
    }
  ]
}
```

## API Documentation

Once the server is running, access the interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## File Structure

```
.
├── frontend/                    # React frontend app
│   ├── package.json
│   ├── public/
│   └── src/
├── main.py                      # FastAPI application
├── loan_approval_model.py       # ML model functions
├── loan_data.csv               # Training data
├── requirements.txt             # Python dependencies
├── loan_model.pkl               # Saved trained model (auto-created)
├── loan_preprocessor.pkl        # Saved preprocessor (auto-created)
└── README.md                    # This file
```

## Model Training

The API automatically trains the model on first run if saved models don't exist:

1. Loads `loan_data.csv`
2. Preprocesses data (handles missing values, encoding)
3. Trains RandomForestClassifier
4. Saves model for future use

To retrain from scratch, delete `loan_model.pkl` and `loan_preprocessor.pkl`.

## Response Fields

| Field | Description |
|-------|-------------|
| `loan_approved` | 0=Rejected, 1=Approved |
| `approval_probability` | Confidence score (0.0-1.0) |
| `credit_score` | Score range 300-900 |
| `top_features` | Top 3 features affecting decision with impact values |

## Input Fields

All fields are required:

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
| `Credit_History` | float | 1.0 (0.0-1.0) |
| `Property_Area` | string | "Urban" / "Semiurban" / "Rural" |

## Deployment

### Backend (Render or Railway)
1. Push the project to GitHub.
2. Add a new service on Render or Railway.
3. Point the service to this repository.
4. Set the start command to:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
5. Make sure `requirements.txt` is selected as the dependency file.
6. Add the dataset file `loan_data.csv` to the repo so the model can train on first deploy.

### Frontend (Vercel)
1. Create a new Vercel project and connect it to this repository.
2. Set the root directory to `frontend`.
3. Configure the build command:
   ```bash
   npm install && npm run build
   ```
4. Set the publish directory to `build`.
5. Deploy and visit the Vercel URL.

> Tip: If the frontend and backend are deployed separately, you may need to configure CORS in the FastAPI app or set the frontend API URL to the deployed backend address.

## Development

To modify the API:

1. Update request/response schemas in `main.py` (classes with `BaseModel`)
2. Add new endpoints as decorated functions (`@app.get()`, `@app.post()`)
3. Restart the server
4. Test via http://localhost:8000/docs

## License

MIT
