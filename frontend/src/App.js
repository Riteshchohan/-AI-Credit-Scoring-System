import { useState } from 'react';
import {
  ArrowRightOnRectangleIcon,
  ChartBarIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  LockClosedIcon,
  ShieldCheckIcon,
  SparklesIcon,
  UserCircleIcon,
  XCircleIcon,
  ChatBubbleLeftRightIcon,
  PaperAirplaneIcon,
  LightBulbIcon,
} from '@heroicons/react/24/outline';
 
//const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const API_URL = process.env.REACT_APP_API_URL || 'https://ai-credit-scoring-system.onrender.com';
 
const initialForm = {
  ApplicantIncome: '',
  LoanAmount: '',
  Credit_History: '1.0',
  Dependents: '0'
};
 
const getRiskLevel = (approved, score) => {
  if (approved !== 1) return { level: 'High', color: 'text-red-600', bg: 'bg-red-50' };
  if (score >= 700) return { level: 'Low', color: 'text-green-600', bg: 'bg-green-50' };
  if (score >= 600) return { level: 'Medium', color: 'text-yellow-600', bg: 'bg-yellow-50' };
  return { level: 'High', color: 'text-red-600', bg: 'bg-red-50' };
};
 
function BrandMark({ compact = false }) {
  return (
    <div className="flex items-center gap-3">
      <div className="relative h-10 w-10 shrink-0">
        <div className="absolute inset-0 rounded-2xl bg-brand-700 shadow-sm" />
        <div className="absolute inset-[3px] rounded-[14px] bg-white/10" />
        <div className="absolute left-[6px] top-[6px] h-5 w-5 rounded-xl bg-accent-400 shadow-sm" />
        <div className="absolute bottom-[7px] right-[7px] h-4 w-4 rounded-lg bg-white/80" />
      </div>
      {!compact && (
        <div className="leading-tight">
          <div className="text-base font-extrabold tracking-tight text-brand-900">
            Credit scoring
          </div>
          <div className="text-xs font-medium text-slate-600">
            Loan decisioning with explainability
          </div>
        </div>
      )}
    </div>
  );
}
 
function StatCard({ icon: Icon, label, value, tone = 'neutral', sub }) {
  const toneStyles = {
    neutral: 'bg-white',
    good: 'bg-emerald-50/60 border-emerald-100',
    warn: 'bg-amber-50/60 border-amber-100',
    bad: 'bg-rose-50/60 border-rose-100',
    brand: 'bg-brand-50/70 border-brand-100',
  };
  return (
    <div className={`rounded-2xl border p-4 ${toneStyles[tone] || toneStyles.neutral}`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-xs font-semibold text-slate-600">{label}</div>
          <div className="mt-1 text-2xl font-extrabold tracking-tight text-slate-900">{value}</div>
          {sub && <div className="mt-1 text-xs text-slate-600">{sub}</div>}
        </div>
        {Icon && (
          <div className="rounded-xl bg-white/80 p-2 shadow-sm ring-1 ring-slate-200/60">
            <Icon className="h-5 w-5 text-brand-700" />
          </div>
        )}
      </div>
    </div>
  );
}
 
function FeatureRow({ feature, impact }) {
  const magnitude = Math.min(1, Math.abs(Number(impact)) / 0.5);
  const positive = Number(impact) >= 0;
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-sm font-semibold text-slate-900">{feature}</div>
          <div className="mt-1 text-xs text-slate-600">
            Impact: <span className={positive ? 'text-emerald-700 font-semibold' : 'text-rose-700 font-semibold'}>
              {positive ? '+' : ''}{Number(impact).toFixed(3)}
            </span>
          </div>
        </div>
        <div className="w-32">
          <div className="h-2 w-full rounded-full bg-slate-100">
            <div
              className={`h-2 rounded-full ${positive ? 'bg-emerald-400' : 'bg-rose-400'}`}
              style={{ width: `${Math.round(magnitude * 100)}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
 
// ── AI Suggestions Card ──────────────────────────────────────────────────────
function AISuggestionsCard({ suggestions }) {
  if (!suggestions) return null;
 
  // Split numbered suggestions into individual points if possible
  const lines = suggestions
    .split(/(?=\d\.\s)/)
    .map(s => s.trim())
    .filter(Boolean);
 
  return (
    <div className="surface rounded-3xl p-6 border border-brand-100 bg-gradient-to-br from-brand-50/60 to-white">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-extrabold tracking-tight text-slate-900">
            AI Financial Advice
          </h2>
          <p className="mt-1 text-sm text-slate-600">
            Personalised suggestions based on your application.
          </p>
        </div>
        <div className="rounded-xl bg-brand-100 p-2 ring-1 ring-brand-200">
          <LightBulbIcon className="h-5 w-5 text-brand-700" />
        </div>
      </div>
 
      <div className="mt-4 grid gap-3">
        {lines.length > 1 ? (
          lines.map((line, idx) => (
            <div key={idx} className="flex items-start gap-3 rounded-2xl border border-slate-100 bg-white p-3">
              <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-brand-100 text-xs font-bold text-brand-700">
                {idx + 1}
              </div>
              <p className="text-sm text-slate-700 leading-relaxed">
                {line.replace(/^\d\.\s*/, '')}
              </p>
            </div>
          ))
        ) : (
          <p className="text-sm text-slate-700 leading-relaxed">{suggestions}</p>
        )}
      </div>
    </div>
  );
}
 
// ── AI Q&A Card ──────────────────────────────────────────────────────────────
function AIQACard({ token, apiUrl }) {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [qaLoading, setQaLoading] = useState(false);
  const [qaError, setQaError] = useState('');
 
  const suggestedQuestions = [
    'Why was my loan approved or rejected?',
    'How can I improve my credit score?',
    'What is the most important factor in my result?',
  ];
 
  const askQuestion = async (q) => {
    const text = q || question;
    if (!text.trim()) return;
    setQaError('');
    setAnswer('');
    setQaLoading(true);
    try {
      const response = await fetch(`${apiUrl}/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ question: text })
      });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || 'Failed to get answer');
      }
      const data = await response.json();
      setAnswer(data.answer);
    } catch (err) {
      setQaError(err.message || 'Unable to get answer. Please try again.');
    } finally {
      setQaLoading(false);
    }
  };
 
  return (
    <div className="surface rounded-3xl p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-extrabold tracking-tight text-slate-900">
            Ask AI Assistant
          </h2>
          <p className="mt-1 text-sm text-slate-600">
            Ask anything about your loan decision.
          </p>
        </div>
        <div className="rounded-xl bg-slate-100 p-2 ring-1 ring-slate-200">
          <ChatBubbleLeftRightIcon className="h-5 w-5 text-slate-700" />
        </div>
      </div>
 
      {/* Suggested questions */}
      <div className="mt-4 flex flex-wrap gap-2">
        {suggestedQuestions.map((q, idx) => (
          <button
            key={idx}
            onClick={() => { setQuestion(q); askQuestion(q); }}
            className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:border-brand-300 hover:bg-brand-50 hover:text-brand-700 transition-colors"
          >
            {q}
          </button>
        ))}
      </div>
 
      {/* Input */}
      <div className="mt-4 flex gap-2">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && askQuestion()}
          placeholder="Type your question..."
          className="input flex-1 text-sm"
        />
        <button
          onClick={() => askQuestion()}
          disabled={qaLoading || !question.trim()}
          className="btn-primary flex items-center gap-1 px-4 py-2 text-sm disabled:opacity-50"
        >
          {qaLoading ? (
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
          ) : (
            <PaperAirplaneIcon className="h-4 w-4" />
          )}
        </button>
      </div>
 
      {/* Error */}
      {qaError && (
        <div className="mt-3 flex items-center gap-2 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
          <ExclamationTriangleIcon className="h-4 w-4 shrink-0" />
          {qaError}
        </div>
      )}
 
      {/* Answer */}
      {answer && (
        <div className="mt-3 rounded-2xl border border-emerald-100 bg-emerald-50/60 p-4">
          <div className="flex items-center gap-2 mb-2">
            <SparklesIcon className="h-4 w-4 text-brand-700" />
            <span className="text-xs font-semibold text-brand-700">AI Answer</span>
          </div>
          <p className="text-sm text-slate-700 leading-relaxed">{answer}</p>
        </div>
      )}
    </div>
  );
}
 
// ── Main App ─────────────────────────────────────────────────────────────────
function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [token, setToken] = useState('');
  const [authMode, setAuthMode] = useState('login');
  const [authForm, setAuthForm] = useState({ username: '', password: '', email: '' });
  const [formData, setFormData] = useState(initialForm);
  const [loading, setLoading] = useState(false);
  const [authLoading, setAuthLoading] = useState(false);
  const [error, setError] = useState('');
  const [authError, setAuthError] = useState('');
  const [result, setResult] = useState(null);
 
  const handleAuthChange = (event) => {
    const { name, value } = event.target;
    setAuthForm((prev) => ({ ...prev, [name]: value }));
  };
 
  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };
 
  const handleAuth = async (event) => {
    event.preventDefault();
    setAuthError('');
    setAuthLoading(true);
    try {
      const endpoint = authMode === 'login' ? '/login' : '/register';
      const payload = authMode === 'login'
        ? { username: authForm.username, password: authForm.password }
        : { username: authForm.username, password: authForm.password, email: authForm.email };
 
      try { await fetch(`${API_URL}/`, { method: 'GET' }); } catch (_) {}
 
      const response = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
 
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || `${authMode === 'login' ? 'Login' : 'Registration'} failed`);
      }
 
      const data = await response.json();
      setToken(data.access_token);
      setIsAuthenticated(true);
      setAuthForm({ username: '', password: '', email: '' });
    } catch (err) {
      setAuthError(err.message === 'Failed to fetch'
        ? 'Server is waking up, please wait 30 seconds and try again.'
        : err.message || `Unable to ${authMode}`);
    } finally {
      setAuthLoading(false);
    }
  };
 
  const handleLogout = () => {
    setIsAuthenticated(false);
    setToken('');
    setResult(null);
    setFormData(initialForm);
  };
 
  const buildPayload = () => ({
    Gender: 'Male',
    Married: 'Yes',
    Dependents: formData.Dependents,
    Education: 'Graduate',
    Self_Employed: 'No',
    ApplicantIncome: Number(formData.ApplicantIncome),
    CoapplicantIncome: 0,
    LoanAmount: Number(formData.LoanAmount),
    Loan_Amount_Term: 360,
    Credit_History: Number(formData.Credit_History),
    Property_Area: 'Urban'
  });
 
  const handleSubmit = async (event) => {
    event.preventDefault();
    setError('');
    setResult(null);
    setLoading(true);
    try {
      const payload = buildPayload();
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(payload)
      });
 
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || 'Prediction request failed');
      }
 
      const data = await response.json();
      setResult({
        approval: data.loan_approved === 1 ? 'Approved' : 'Rejected',
        probability: data.approval_probability,
        creditScore: data.credit_score,
        risk: getRiskLevel(data.loan_approved, data.credit_score),
        topFeatures: data.top_features,
        aiSuggestions: data.ai_suggestions || null
      });
    } catch (err) {
      setError(err.message || 'Unable to contact backend');
    } finally {
      setLoading(false);
    }
  };
 
  // ── Login / Register Screen ────────────────────────────────────────────────
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-brand-50/30 flex">
        <div className="m-auto w-full max-w-5xl grid grid-cols-1 lg:grid-cols-2 gap-0 overflow-hidden rounded-3xl shadow-2xl shadow-slate-200/60 border border-slate-200/60">
 
          {/* Left — form */}
          <div className="bg-white p-10 flex flex-col justify-center">
            <BrandMark />
            <h1 className="mt-8 text-3xl font-extrabold tracking-tight text-slate-900">
              {authMode === 'login' ? 'Welcome back' : 'Create account'}
            </h1>
            <p className="mt-2 text-sm text-slate-600">
              {authMode === 'login'
                ? 'Sign in to generate a credit score and see the most important factors behind the decision.'
                : 'Create your account to get started with AI-powered loan analysis.'}
            </p>
 
            <form onSubmit={handleAuth} className="mt-8 grid gap-4">
              <div className="grid gap-2">
                <label className="text-xs font-semibold text-slate-700">Username</label>
                <div className="relative">
                  <UserCircleIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                  <input name="username" type="text" required value={authForm.username} onChange={handleAuthChange} placeholder="admin" className="input pl-9" />
                </div>
              </div>
 
              {authMode === 'register' && (
                <div className="grid gap-2">
                  <label className="text-xs font-semibold text-slate-700">Email</label>
                  <input name="email" type="email" required value={authForm.email} onChange={handleAuthChange} placeholder="you@example.com" className="input" />
                </div>
              )}
 
              <div className="grid gap-2">
                <label className="text-xs font-semibold text-slate-700">Password</label>
                <div className="relative">
                  <LockClosedIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                  <input name="password" type="password" required value={authForm.password} onChange={handleAuthChange} placeholder="••••••••" className="input pl-9" />
                </div>
              </div>
 
              {authError && (
                <div className="flex items-start gap-3 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
                  <ExclamationTriangleIcon className="mt-0.5 h-5 w-5 shrink-0" />
                  <div>{authError}</div>
                </div>
              )}
 
              <button type="submit" disabled={authLoading} className="btn-primary mt-2">
                {authLoading ? 'Please wait…' : authMode === 'login' ? 'Sign in' : 'Create account'}
                <ArrowRightOnRectangleIcon className="h-4 w-4" />
              </button>
            </form>
 
            <p className="mt-6 text-center text-sm text-slate-600">
              {authMode === 'login' ? "Don't have an account?" : 'Already have an account?'}{' '}
              <button onClick={() => setAuthMode(authMode === 'login' ? 'register' : 'login')} className="font-semibold text-brand-700 hover:underline">
                {authMode === 'login' ? 'Sign up' : 'Sign in'}
              </button>
            </p>
 
            {authMode === 'login' && (
              <div className="mt-6 rounded-2xl border border-slate-200 bg-slate-50 p-4">
                <div className="text-xs font-semibold text-slate-700 mb-3">Demo credentials</div>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="rounded-xl bg-white border border-slate-200 p-3">
                    <div className="font-semibold text-slate-800">admin</div>
                    <div className="text-slate-500">admin123</div>
                  </div>
                  <div className="rounded-xl bg-white border border-slate-200 p-3">
                    <div className="font-semibold text-slate-800">user</div>
                    <div className="text-slate-500">user123</div>
                  </div>
                </div>
              </div>
            )}
          </div>
 
          {/* Right — preview panel */}
          <div className="hidden lg:flex flex-col justify-center bg-gradient-to-br from-brand-700 to-brand-900 p-10 text-white">
            <div className="chip bg-white/10 text-white ring-1 ring-white/20 w-fit mb-6">
              <SparklesIcon className="mr-1 h-4 w-4" /> Smart scoring
            </div>
            <h2 className="text-2xl font-extrabold tracking-tight">Professional, modern and consistent branding</h2>
            <p className="mt-3 text-sm text-white/70">Clean spacing, refined typography, and an accessible color system that matches your reference (teal + lime).</p>
            <div className="mt-8 grid gap-3">
              {[
                { label: 'Approval outcome', value: 'Approved / Rejected', sub: 'Clear status at a glance' },
                { label: 'Risk tier', value: 'Low / Med / High' },
                { label: 'Confidence', value: '0–100%' },
              ].map((item, i) => (
                <div key={i} className="rounded-2xl bg-white/10 p-4 ring-1 ring-white/10">
                  <div className="text-xs text-white/60">{item.label}</div>
                  <div className="mt-1 text-lg font-extrabold">{item.value}</div>
                  {item.sub && <div className="mt-0.5 text-xs text-white/50">{item.sub}</div>}
                </div>
              ))}
              <div className="rounded-2xl bg-white/10 p-4 ring-1 ring-white/10">
                <div className="text-xs text-white/60">AI Suggestions</div>
                <div className="mt-1 text-sm font-semibold">Personalised financial advice</div>
                <div className="mt-0.5 text-xs text-white/50">Powered by Hugging Face Flan-T5</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
 
  // ── Main Dashboard ─────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-brand-50/20">
      <header className="sticky top-0 z-10 border-b border-slate-200/60 bg-white/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <BrandMark />
          <button onClick={handleLogout} className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 transition-colors">
            <ArrowRightOnRectangleIcon className="h-4 w-4" />
            Sign out
          </button>
        </div>
      </header>
 
      <main className="mx-auto max-w-7xl px-6 py-10">
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-5">
 
          {/* Left — form */}
          <section className="lg:col-span-2">
            <div className="surface rounded-3xl p-6">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h2 className="text-lg font-extrabold tracking-tight text-slate-900">Loan application</h2>
                  <p className="mt-1 text-sm text-slate-600">Enter a few details. We'll return approval, credit score and top factors.</p>
                </div>
                <div className="chip bg-accent-100 text-accent-900 ring-1 ring-accent-200">
                  <SparklesIcon className="mr-1 h-4 w-4" /> Pro UI
                </div>
              </div>
 
              <form onSubmit={handleSubmit} className="mt-6 grid gap-4">
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <div className="grid gap-2">
                    <label htmlFor="ApplicantIncome" className="text-xs font-semibold text-slate-700">Annual income</label>
                    <input id="ApplicantIncome" name="ApplicantIncome" type="number" required className="input" placeholder="50000" value={formData.ApplicantIncome} onChange={handleChange} />
                  </div>
                  <div className="grid gap-2">
                    <label htmlFor="LoanAmount" className="text-xs font-semibold text-slate-700">Loan amount</label>
                    <input id="LoanAmount" name="LoanAmount" type="number" required className="input" placeholder="200000" value={formData.LoanAmount} onChange={handleChange} />
                  </div>
                  <div className="grid gap-2">
                    <label htmlFor="Credit_History" className="text-xs font-semibold text-slate-700">Credit history</label>
                    <select id="Credit_History" name="Credit_History" className="select" value={formData.Credit_History} onChange={handleChange}>
                      <option value="1.0">Good (1.0)</option>
                      <option value="0.0">Poor (0.0)</option>
                    </select>
                  </div>
                  <div className="grid gap-2">
                    <label htmlFor="Dependents" className="text-xs font-semibold text-slate-700">Dependents</label>
                    <select id="Dependents" name="Dependents" className="select" value={formData.Dependents} onChange={handleChange}>
                      <option value="0">0</option>
                      <option value="1">1</option>
                      <option value="2">2</option>
                      <option value="3+">3+</option>
                    </select>
                  </div>
                </div>
 
                {error && (
                  <div className="flex items-start gap-3 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
                    <ExclamationTriangleIcon className="mt-0.5 h-5 w-5" />
                    <div>{error}</div>
                  </div>
                )}
 
                <button type="submit" disabled={loading} className="btn-primary">
                  {loading ? 'Analyzing…' : 'Get prediction'}
                  <SparklesIcon className="h-4 w-4" />
                </button>
 
                <div className="rounded-2xl border border-slate-200 bg-white/60 p-4">
                  <div className="text-xs font-semibold text-slate-700">What you'll get</div>
                  <div className="mt-3 grid gap-2 text-sm text-slate-700">
                    <div className="flex items-center gap-2"><CheckCircleIcon className="h-5 w-5 text-emerald-600" /> Approval status & risk tier</div>
                    <div className="flex items-center gap-2"><ChartBarIcon className="h-5 w-5 text-brand-700" /> Credit score and probability</div>
                    <div className="flex items-center gap-2"><ShieldCheckIcon className="h-5 w-5 text-slate-600" /> Key factors with impacts</div>
                    <div className="flex items-center gap-2"><LightBulbIcon className="h-5 w-5 text-amber-500" /> AI financial suggestions</div>
                    <div className="flex items-center gap-2"><ChatBubbleLeftRightIcon className="h-5 w-5 text-brand-600" /> Ask AI about your result</div>
                  </div>
                </div>
              </form>
            </div>
          </section>
 
          {/* Right — results */}
          <section className="lg:col-span-3">
            {!result ? (
              <div className="surface flex h-full min-h-[420px] flex-col justify-center rounded-3xl p-8 text-center">
                <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-brand-50 ring-1 ring-brand-100">
                  <ChartBarIcon className="h-7 w-7 text-brand-700" />
                </div>
                <h3 className="mt-4 text-xl font-extrabold tracking-tight text-slate-900">Results will appear here</h3>
                <p className="mt-2 text-sm text-slate-600">Submit the form to generate a prediction, credit score, AI suggestions, and top factors.</p>
              </div>
            ) : (
              <div className="grid gap-6">
 
                {/* Decision summary */}
                <div className="surface rounded-3xl p-6">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <h2 className="text-lg font-extrabold tracking-tight text-slate-900">Decision summary</h2>
                      <p className="mt-1 text-sm text-slate-600">Instant prediction based on your inputs.</p>
                    </div>
                    <div className={`chip ${result.approval === 'Approved' ? 'bg-emerald-100 text-emerald-900 ring-1 ring-emerald-200' : 'bg-rose-100 text-rose-900 ring-1 ring-rose-200'}`}>
                      {result.approval === 'Approved' ? <CheckCircleIcon className="mr-1 h-4 w-4" /> : <XCircleIcon className="mr-1 h-4 w-4" />}
                      {result.approval}
                    </div>
                  </div>
                  <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
                    <StatCard icon={ShieldCheckIcon} label="Risk level" value={result.risk.level} tone={result.risk.level === 'Low' ? 'good' : result.risk.level === 'Medium' ? 'warn' : 'bad'} sub="Higher risk may lower approval odds" />
                    <StatCard icon={ChartBarIcon} label="Approval probability" value={`${Math.round(result.probability * 100)}%`} tone="brand" sub="Model confidence" />
                    <StatCard icon={SparklesIcon} label="Credit score" value={result.creditScore} tone="neutral" sub="Range 300–900" />
                    <StatCard icon={result.approval === 'Approved' ? CheckCircleIcon : XCircleIcon} label="Loan status" value={result.approval} tone={result.approval === 'Approved' ? 'good' : 'bad'} sub="Final predicted outcome" />
                  </div>
                </div>
 
                {/* Key factors */}
                <div className="surface rounded-3xl p-6">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <h2 className="text-lg font-extrabold tracking-tight text-slate-900">Key factors</h2>
                      <p className="mt-1 text-sm text-slate-600">The most influential features for this decision.</p>
                    </div>
                    <div className="chip bg-slate-50 text-slate-800 ring-1 ring-slate-200">Top {result.topFeatures?.length || 0}</div>
                  </div>
                  <div className="mt-6 grid gap-3">
                    {result.topFeatures?.map((f, idx) => (
                      <FeatureRow key={idx} feature={f.feature} impact={f.impact} />
                    ))}
                  </div>
                </div>
 
                {/* AI Suggestions */}
                {result.aiSuggestions && (
                  <AISuggestionsCard suggestions={result.aiSuggestions} />
                )}
 
                {/* AI Q&A */}
                <AIQACard token={token} apiUrl={API_URL} />
 
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
 
export default App;