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
 
const API_URL = process.env.REACT_APP_API_URL || 'https://ai-credit-scoring-system.onrender.com';
 
const initialForm = {
  ApplicantIncome: '',
  LoanAmount: '',
  Credit_History: '1.0',
  Dependents: '0',
};
 
const getRiskLevel = (approved, score) => {
  if (approved !== 1) return { level: 'High', color: 'text-red-600', bg: 'bg-red-50' };
  if (score >= 700) return { level: 'Low', color: 'text-green-600', bg: 'bg-green-50' };
  if (score >= 600) return { level: 'Medium', color: 'text-yellow-600', bg: 'bg-yellow-50' };
  return { level: 'High', color: 'text-red-600', bg: 'bg-red-50' };
};
 
// ── Brand Mark ────────────────────────────────────────────────────────────────
function BrandMark({ compact = false }) {
  return (
    <div className="flex items-center gap-2 sm:gap-3">
      <div className="relative h-8 w-8 sm:h-10 sm:w-10 shrink-0">
        <div className="absolute inset-0 rounded-xl sm:rounded-2xl bg-teal-600 shadow-sm" />
        <div className="absolute inset-[3px] rounded-lg sm:rounded-[14px] bg-white/10" />
        <div className="absolute left-[5px] top-[5px] sm:left-[6px] sm:top-[6px] h-4 w-4 sm:h-5 sm:w-5 rounded-lg sm:rounded-xl bg-lime-400 shadow-sm" />
        <div className="absolute bottom-[5px] right-[5px] sm:bottom-[7px] sm:right-[7px] h-3 w-3 sm:h-4 sm:w-4 rounded-md sm:rounded-lg bg-white/80" />
      </div>
      {!compact && (
        <div className="leading-tight min-w-0">
          <div className="text-sm sm:text-base font-extrabold tracking-tight text-teal-900 truncate">
            Credit scoring
          </div>
          <div className="text-[10px] sm:text-xs font-medium text-slate-500 truncate">
            Loan decisioning with explainability
          </div>
        </div>
      )}
    </div>
  );
}
 
// ── Stat Card ─────────────────────────────────────────────────────────────────
function StatCard({ icon: Icon, label, value, tone = 'neutral', sub }) {
  const toneStyles = {
    neutral: 'bg-white border-slate-200',
    good: 'bg-emerald-50 border-emerald-100',
    warn: 'bg-amber-50 border-amber-100',
    bad: 'bg-rose-50 border-rose-100',
    brand: 'bg-teal-50 border-teal-100',
  };
  return (
    <div className={`rounded-2xl border p-3 sm:p-4 ${toneStyles[tone] || toneStyles.neutral}`}>
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="text-[10px] sm:text-xs font-semibold text-slate-500 uppercase tracking-wide truncate">{label}</div>
          <div className="mt-1 text-xl sm:text-2xl font-extrabold tracking-tight text-slate-900 truncate">{value}</div>
          {sub && <div className="mt-0.5 text-[10px] sm:text-xs text-slate-500 truncate">{sub}</div>}
        </div>
        {Icon && (
          <div className="rounded-lg sm:rounded-xl bg-white/80 p-1.5 sm:p-2 shadow-sm ring-1 ring-slate-200/60 shrink-0">
            <Icon className="h-4 w-4 sm:h-5 sm:w-5 text-teal-600" />
          </div>
        )}
      </div>
    </div>
  );
}
 
// ── Feature Row ───────────────────────────────────────────────────────────────
function FeatureRow({ feature, impact }) {
  const magnitude = Math.min(1, Math.abs(Number(impact)) / 0.5);
  const positive = Number(impact) >= 0;
  return (
    <div className="rounded-xl sm:rounded-2xl border border-slate-200 bg-white p-3 sm:p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="text-xs sm:text-sm font-semibold text-slate-900 truncate">{feature}</div>
          <div className="mt-0.5 text-[10px] sm:text-xs text-slate-500">
            Impact:{' '}
            <span className={positive ? 'text-emerald-700 font-semibold' : 'text-rose-700 font-semibold'}>
              {positive ? '+' : ''}{Number(impact).toFixed(3)}
            </span>
          </div>
        </div>
        <div className="w-20 sm:w-32 shrink-0">
          <div className="h-1.5 sm:h-2 w-full rounded-full bg-slate-100">
            <div
              className={`h-1.5 sm:h-2 rounded-full transition-all ${positive ? 'bg-emerald-400' : 'bg-rose-400'}`}
              style={{ width: `${Math.round(magnitude * 100)}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
 
// ── AI Suggestions Card ───────────────────────────────────────────────────────
function AISuggestionsCard({ suggestions }) {
  if (!suggestions) return null;
  const lines = suggestions.split(/(?=\d\.\s)/).map(s => s.trim()).filter(Boolean);
 
  return (
    <div className="rounded-2xl sm:rounded-3xl border border-teal-100 bg-gradient-to-br from-teal-50/60 to-white p-4 sm:p-6 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-base sm:text-lg font-extrabold tracking-tight text-slate-900">AI Financial Advice</h2>
          <p className="mt-0.5 text-xs sm:text-sm text-slate-500">Personalised suggestions based on your application.</p>
        </div>
        <div className="shrink-0 rounded-lg sm:rounded-xl bg-teal-100 p-1.5 sm:p-2 ring-1 ring-teal-200">
          <LightBulbIcon className="h-4 w-4 sm:h-5 sm:w-5 text-teal-700" />
        </div>
      </div>
      <div className="mt-4 grid gap-2 sm:gap-3">
        {lines.length > 1 ? (
          lines.map((line, idx) => (
            <div key={idx} className="flex items-start gap-2 sm:gap-3 rounded-xl sm:rounded-2xl border border-slate-100 bg-white p-2.5 sm:p-3">
              <div className="mt-0.5 flex h-5 w-5 sm:h-6 sm:w-6 shrink-0 items-center justify-center rounded-full bg-teal-100 text-[10px] sm:text-xs font-bold text-teal-700">
                {idx + 1}
              </div>
              <p className="text-xs sm:text-sm text-slate-700 leading-relaxed">{line.replace(/^\d\.\s*/, '')}</p>
            </div>
          ))
        ) : (
          <p className="text-xs sm:text-sm text-slate-700 leading-relaxed">{suggestions}</p>
        )}
      </div>
    </div>
  );
}
 
// ── AI Q&A Card ───────────────────────────────────────────────────────────────
function AIQACard({ token, apiUrl }) {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [qaLoading, setQaLoading] = useState(false);
  const [qaError, setQaError] = useState('');
 
  const suggestedQuestions = [
    'Why was my loan approved or rejected?',
    'How can I improve my credit score?',
    'What is the most important factor?',
  ];
 
  const askQuestion = async (q) => {
    const text = (q || question).trim();
    if (!text) return;
    setQaError('');
    setAnswer('');
    setQaLoading(true);
    try {
      const response = await fetch(`${apiUrl}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ question: text }),
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
    <div className="rounded-2xl sm:rounded-3xl border border-slate-200 bg-white p-4 sm:p-6 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-base sm:text-lg font-extrabold tracking-tight text-slate-900">Ask AI Assistant</h2>
          <p className="mt-0.5 text-xs sm:text-sm text-slate-500">Ask anything about your loan decision.</p>
        </div>
        <div className="shrink-0 rounded-lg sm:rounded-xl bg-slate-100 p-1.5 sm:p-2 ring-1 ring-slate-200">
          <ChatBubbleLeftRightIcon className="h-4 w-4 sm:h-5 sm:w-5 text-slate-600" />
        </div>
      </div>
 
      {/* Suggested questions — wraps nicely on mobile */}
      <div className="mt-3 sm:mt-4 flex flex-wrap gap-1.5 sm:gap-2">
        {suggestedQuestions.map((q, idx) => (
          <button
            key={idx}
            onClick={() => { setQuestion(q); askQuestion(q); }}
            className="rounded-full border border-slate-200 bg-slate-50 px-2.5 sm:px-3 py-1 sm:py-1.5 text-[10px] sm:text-xs font-medium text-slate-600 hover:border-teal-300 hover:bg-teal-50 hover:text-teal-700 transition-colors active:scale-95"
          >
            {q}
          </button>
        ))}
      </div>
 
      {/* Input row */}
      <div className="mt-3 sm:mt-4 flex gap-2">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && askQuestion()}
          placeholder="Type your question…"
          className="flex-1 min-w-0 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs sm:text-sm text-slate-900 placeholder-slate-400 focus:border-teal-400 focus:bg-white focus:outline-none focus:ring-2 focus:ring-teal-100 transition"
        />
        <button
          onClick={() => askQuestion()}
          disabled={qaLoading || !question.trim()}
          className="shrink-0 flex items-center justify-center rounded-xl bg-teal-600 px-3 sm:px-4 py-2 text-white hover:bg-teal-700 disabled:opacity-40 disabled:cursor-not-allowed transition active:scale-95"
        >
          {qaLoading
            ? <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
            : <PaperAirplaneIcon className="h-4 w-4" />}
        </button>
      </div>
 
      {qaError && (
        <div className="mt-3 flex items-start gap-2 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2.5 text-xs sm:text-sm text-rose-800">
          <ExclamationTriangleIcon className="mt-0.5 h-4 w-4 shrink-0" />
          {qaError}
        </div>
      )}
 
      {answer && (
        <div className="mt-3 rounded-xl sm:rounded-2xl border border-emerald-100 bg-emerald-50 p-3 sm:p-4">
          <div className="flex items-center gap-1.5 mb-1.5">
            <SparklesIcon className="h-3.5 w-3.5 sm:h-4 sm:w-4 text-teal-600" />
            <span className="text-[10px] sm:text-xs font-semibold text-teal-700 uppercase tracking-wide">AI Answer</span>
          </div>
          <p className="text-xs sm:text-sm text-slate-700 leading-relaxed">{answer}</p>
        </div>
      )}
    </div>
  );
}
 
// ── Main App ──────────────────────────────────────────────────────────────────
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
 
  const handleAuthChange = (e) => {
    const { name, value } = e.target;
    setAuthForm(prev => ({ ...prev, [name]: value }));
  };
 
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };
 
  const handleAuth = async (e) => {
    e.preventDefault();
    setAuthError('');
    setAuthLoading(true);
    try {
      const endpoint = authMode === 'login' ? '/login' : '/register';
      const payload = authMode === 'login'
        ? { username: authForm.username, password: authForm.password }
        : { username: authForm.username, password: authForm.password, email: authForm.email };
 
      try { await fetch(`${API_URL}/`, { method: 'GET' }); } catch (_) {}
 
      const res = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail || `${authMode === 'login' ? 'Login' : 'Registration'} failed`);
      }
      const data = await res.json();
      setToken(data.access_token);
      setIsAuthenticated(true);
      setAuthForm({ username: '', password: '', email: '' });
    } catch (err) {
      setAuthError(err.message === 'Failed to fetch'
        ? 'Server is waking up — please wait 30 seconds and try again.'
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
 
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setResult(null);
    setLoading(true);
    try {
      const payload = {
        Gender: 'Male', Married: 'Yes', Dependents: formData.Dependents,
        Education: 'Graduate', Self_Employed: 'No',
        ApplicantIncome: Number(formData.ApplicantIncome),
        CoapplicantIncome: 0, LoanAmount: Number(formData.LoanAmount),
        Loan_Amount_Term: 360, Credit_History: Number(formData.Credit_History),
        Property_Area: 'Urban',
      };
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail || 'Prediction request failed');
      }
      const data = await res.json();
      setResult({
        approval: data.loan_approved === 1 ? 'Approved' : 'Rejected',
        probability: data.approval_probability,
        creditScore: data.credit_score,
        risk: getRiskLevel(data.loan_approved, data.credit_score),
        topFeatures: data.top_features,
        aiSuggestions: data.ai_suggestions || null,
      });
    } catch (err) {
      setError(err.message || 'Unable to contact backend');
    } finally {
      setLoading(false);
    }
  };
 
  // shared input classes
  const inputCls = "w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm text-slate-900 placeholder-slate-400 focus:border-teal-400 focus:bg-white focus:outline-none focus:ring-2 focus:ring-teal-100 transition";
  const selectCls = inputCls;
 
  // ── Login / Register ────────────────────────────────────────────────────────
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-100 to-teal-50 flex items-center justify-center p-4 sm:p-6">
        <div className="w-full max-w-4xl overflow-hidden rounded-2xl sm:rounded-3xl shadow-2xl shadow-slate-200/80 border border-slate-200/60 grid grid-cols-1 lg:grid-cols-2">
 
          {/* Form side */}
          <div className="bg-white p-6 sm:p-8 lg:p-10 flex flex-col justify-center">
            <BrandMark />
            <h1 className="mt-6 sm:mt-8 text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900">
              {authMode === 'login' ? 'Welcome back' : 'Create account'}
            </h1>
            <p className="mt-1.5 sm:mt-2 text-xs sm:text-sm text-slate-500">
              {authMode === 'login'
                ? 'Sign in to generate a credit score and see the key factors behind the decision.'
                : 'Create your account to start AI-powered loan analysis.'}
            </p>
 
            <form onSubmit={handleAuth} className="mt-6 sm:mt-8 grid gap-3 sm:gap-4">
              <div className="grid gap-1.5">
                <label className="text-xs font-semibold text-slate-600">Username</label>
                <div className="relative">
                  <UserCircleIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400 pointer-events-none" />
                  <input name="username" type="text" required value={authForm.username} onChange={handleAuthChange} placeholder="admin" className={`${inputCls} pl-9`} />
                </div>
              </div>
 
              {authMode === 'register' && (
                <div className="grid gap-1.5">
                  <label className="text-xs font-semibold text-slate-600">Email</label>
                  <input name="email" type="email" required value={authForm.email} onChange={handleAuthChange} placeholder="you@example.com" className={inputCls} />
                </div>
              )}
 
              <div className="grid gap-1.5">
                <label className="text-xs font-semibold text-slate-600">Password</label>
                <div className="relative">
                  <LockClosedIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400 pointer-events-none" />
                  <input name="password" type="password" required value={authForm.password} onChange={handleAuthChange} placeholder="••••••••" className={`${inputCls} pl-9`} />
                </div>
              </div>
 
              {authError && (
                <div className="flex items-start gap-2.5 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2.5 text-xs sm:text-sm text-rose-800">
                  <ExclamationTriangleIcon className="mt-0.5 h-4 w-4 shrink-0" />
                  {authError}
                </div>
              )}
 
              <button
                type="submit"
                disabled={authLoading}
                className="mt-1 flex w-full items-center justify-center gap-2 rounded-xl bg-teal-600 px-4 py-2.5 sm:py-3 text-sm font-semibold text-white shadow-sm hover:bg-teal-700 disabled:opacity-50 disabled:cursor-not-allowed transition active:scale-[0.98]"
              >
                {authLoading ? (
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                ) : (
                  <>
                    {authMode === 'login' ? 'Sign in' : 'Create account'}
                    <ArrowRightOnRectangleIcon className="h-4 w-4" />
                  </>
                )}
              </button>
            </form>
 
            <p className="mt-5 sm:mt-6 text-center text-xs sm:text-sm text-slate-500">
              {authMode === 'login' ? "Don't have an account?" : 'Already have an account?'}{' '}
              <button onClick={() => { setAuthMode(authMode === 'login' ? 'register' : 'login'); setAuthError(''); }} className="font-semibold text-teal-700 hover:underline">
                {authMode === 'login' ? 'Sign up' : 'Sign in'}
              </button>
            </p>
 
            {authMode === 'login' && (
              <div className="mt-5 sm:mt-6 rounded-xl sm:rounded-2xl border border-slate-200 bg-slate-50 p-3 sm:p-4">
                <div className="text-[10px] sm:text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2 sm:mb-3">Demo credentials</div>
                <div className="grid grid-cols-2 gap-2 sm:gap-3 text-xs">
                  {[['admin', 'admin123'], ['user', 'user123']].map(([u, p]) => (
                    <button
                      key={u}
                      onClick={() => setAuthForm({ username: u, password: p, email: '' })}
                      className="rounded-lg sm:rounded-xl bg-white border border-slate-200 p-2 sm:p-3 text-left hover:border-teal-300 hover:bg-teal-50 transition active:scale-95"
                    >
                      <div className="font-semibold text-slate-800">{u}</div>
                      <div className="text-slate-400">{p}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
 
          {/* Preview side — hidden on mobile, visible on lg */}
          <div className="hidden lg:flex flex-col justify-center bg-gradient-to-br from-teal-700 to-teal-900 p-10 text-white">
            <div className="inline-flex items-center gap-1.5 rounded-full bg-white/10 px-3 py-1.5 text-xs font-semibold ring-1 ring-white/20 w-fit mb-6">
              <SparklesIcon className="h-3.5 w-3.5" /> Smart scoring
            </div>
            <h2 className="text-2xl font-extrabold tracking-tight leading-snug">
              Professional, modern<br />and consistent branding
            </h2>
            <p className="mt-3 text-sm text-white/60 leading-relaxed">
              Clean spacing, refined typography, and an accessible color system that matches your reference (teal + lime).
            </p>
            <div className="mt-8 grid gap-3">
              {[
                { label: 'Approval outcome', value: 'Approved / Rejected', sub: 'Clear status at a glance' },
                { label: 'Risk tier', value: 'Low / Med / High', sub: 'Confidence score 0–100%' },
                { label: 'AI Suggestions', value: 'Personalised advice', sub: 'Powered by Hugging Face Flan-T5' },
              ].map((item, i) => (
                <div key={i} className="rounded-2xl bg-white/10 p-4 ring-1 ring-white/10">
                  <div className="text-xs text-white/50">{item.label}</div>
                  <div className="mt-1 text-base font-extrabold">{item.value}</div>
                  {item.sub && <div className="mt-0.5 text-xs text-white/40">{item.sub}</div>}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }
 
  // ── Dashboard ───────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-teal-50/30">
 
      {/* Header */}
      <header className="sticky top-0 z-20 border-b border-slate-200/60 bg-white/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 sm:px-6 py-3 sm:py-4">
          <BrandMark />
          <button
            onClick={handleLogout}
            className="flex items-center gap-1.5 sm:gap-2 rounded-lg sm:rounded-xl border border-slate-200 bg-white px-3 sm:px-4 py-1.5 sm:py-2 text-xs sm:text-sm font-semibold text-slate-700 hover:bg-slate-50 transition active:scale-95"
          >
            <ArrowRightOnRectangleIcon className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
            <span className="hidden xs:inline">Sign out</span>
          </button>
        </div>
      </header>
 
      <main className="mx-auto max-w-7xl px-4 sm:px-6 py-6 sm:py-10">
        {/* On mobile: single column stacked. On lg: 2-col side-by-side */}
        <div className="grid grid-cols-1 gap-5 sm:gap-6 lg:gap-8 lg:grid-cols-5">
 
          {/* ── Form panel ── */}
          <section className="lg:col-span-2">
            <div className="rounded-2xl sm:rounded-3xl border border-slate-200 bg-white p-4 sm:p-6 shadow-sm">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h2 className="text-base sm:text-lg font-extrabold tracking-tight text-slate-900">Loan application</h2>
                  <p className="mt-0.5 text-xs sm:text-sm text-slate-500">Enter details for instant AI-powered analysis.</p>
                </div>
                <span className="shrink-0 inline-flex items-center gap-1 rounded-full bg-lime-100 px-2.5 py-1 text-[10px] sm:text-xs font-semibold text-lime-800 ring-1 ring-lime-200">
                  <SparklesIcon className="h-3 w-3 sm:h-3.5 sm:w-3.5" /> Pro UI
                </span>
              </div>
 
              <form onSubmit={handleSubmit} className="mt-4 sm:mt-6 grid gap-3 sm:gap-4">
                {/* 2-col on sm+, 1-col on mobile */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
                  <div className="grid gap-1.5">
                    <label htmlFor="ApplicantIncome" className="text-xs font-semibold text-slate-600">Annual income</label>
                    <input id="ApplicantIncome" name="ApplicantIncome" type="number" required className={inputCls} placeholder="50000" value={formData.ApplicantIncome} onChange={handleChange} />
                  </div>
                  <div className="grid gap-1.5">
                    <label htmlFor="LoanAmount" className="text-xs font-semibold text-slate-600">Loan amount</label>
                    <input id="LoanAmount" name="LoanAmount" type="number" required className={inputCls} placeholder="200000" value={formData.LoanAmount} onChange={handleChange} />
                  </div>
                  <div className="grid gap-1.5">
                    <label htmlFor="Credit_History" className="text-xs font-semibold text-slate-600">Credit history</label>
                    <select id="Credit_History" name="Credit_History" className={selectCls} value={formData.Credit_History} onChange={handleChange}>
                      <option value="1.0">Good (1.0)</option>
                      <option value="0.0">Poor (0.0)</option>
                    </select>
                  </div>
                  <div className="grid gap-1.5">
                    <label htmlFor="Dependents" className="text-xs font-semibold text-slate-600">Dependents</label>
                    <select id="Dependents" name="Dependents" className={selectCls} value={formData.Dependents} onChange={handleChange}>
                      <option value="0">0</option>
                      <option value="1">1</option>
                      <option value="2">2</option>
                      <option value="3+">3+</option>
                    </select>
                  </div>
                </div>
 
                {error && (
                  <div className="flex items-start gap-2.5 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2.5 text-xs sm:text-sm text-rose-800">
                    <ExclamationTriangleIcon className="mt-0.5 h-4 w-4 shrink-0" />
                    {error}
                  </div>
                )}
 
                <button
                  type="submit"
                  disabled={loading}
                  className="flex w-full items-center justify-center gap-2 rounded-xl bg-teal-600 px-4 py-2.5 sm:py-3 text-sm font-semibold text-white hover:bg-teal-700 disabled:opacity-50 disabled:cursor-not-allowed transition active:scale-[0.98] shadow-sm"
                >
                  {loading ? (
                    <>
                      <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                      Analyzing…
                    </>
                  ) : (
                    <>
                      Get prediction
                      <SparklesIcon className="h-4 w-4" />
                    </>
                  )}
                </button>
 
                {/* What you'll get — hidden on mobile to save space */}
                <div className="hidden sm:block rounded-2xl border border-slate-200 bg-slate-50 p-4">
                  <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">What you'll get</div>
                  <div className="grid gap-2 text-xs sm:text-sm text-slate-600">
                    {[
                      [CheckCircleIcon, 'text-emerald-600', 'Approval status & risk tier'],
                      [ChartBarIcon, 'text-teal-600', 'Credit score and probability'],
                      [ShieldCheckIcon, 'text-slate-500', 'Key factors with impacts'],
                      [LightBulbIcon, 'text-amber-500', 'AI financial suggestions'],
                      [ChatBubbleLeftRightIcon, 'text-teal-600', 'Ask AI about your result'],
                    ].map(([Icon, color, label], i) => (
                      <div key={i} className="flex items-center gap-2">
                        <Icon className={`h-4 w-4 sm:h-5 sm:w-5 shrink-0 ${color}`} />
                        {label}
                      </div>
                    ))}
                  </div>
                </div>
              </form>
            </div>
          </section>
 
          {/* ── Results panel ── */}
          <section className="lg:col-span-3">
            {!result ? (
              <div className="flex h-full min-h-[280px] sm:min-h-[420px] flex-col items-center justify-center rounded-2xl sm:rounded-3xl border border-slate-200 bg-white p-6 sm:p-8 text-center shadow-sm">
                <div className="flex h-12 w-12 sm:h-14 sm:w-14 items-center justify-center rounded-2xl bg-teal-50 ring-1 ring-teal-100">
                  <ChartBarIcon className="h-6 w-6 sm:h-7 sm:w-7 text-teal-600" />
                </div>
                <h3 className="mt-4 text-lg sm:text-xl font-extrabold tracking-tight text-slate-900">Results will appear here</h3>
                <p className="mt-2 text-xs sm:text-sm text-slate-500 max-w-xs">
                  Submit the form to generate a prediction, credit score, AI suggestions, and top factors.
                </p>
              </div>
            ) : (
              <div className="grid gap-4 sm:gap-6">
 
                {/* Decision summary */}
                <div className="rounded-2xl sm:rounded-3xl border border-slate-200 bg-white p-4 sm:p-6 shadow-sm">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <h2 className="text-base sm:text-lg font-extrabold tracking-tight text-slate-900">Decision summary</h2>
                      <p className="mt-0.5 text-xs sm:text-sm text-slate-500">Instant prediction based on your inputs.</p>
                    </div>
                    <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-semibold ring-1 shrink-0 ${result.approval === 'Approved' ? 'bg-emerald-100 text-emerald-800 ring-emerald-200' : 'bg-rose-100 text-rose-800 ring-rose-200'}`}>
                      {result.approval === 'Approved'
                        ? <CheckCircleIcon className="h-3.5 w-3.5" />
                        : <XCircleIcon className="h-3.5 w-3.5" />}
                      {result.approval}
                    </span>
                  </div>
                  {/* 2-col on xs+, 1-col on very small */}
                  <div className="mt-4 sm:mt-6 grid grid-cols-2 gap-2 sm:gap-4">
                    <StatCard icon={ShieldCheckIcon} label="Risk level" value={result.risk.level} tone={result.risk.level === 'Low' ? 'good' : result.risk.level === 'Medium' ? 'warn' : 'bad'} sub="Approval risk indicator" />
                    <StatCard icon={ChartBarIcon} label="Probability" value={`${Math.round(result.probability * 100)}%`} tone="brand" sub="Model confidence" />
                    <StatCard icon={SparklesIcon} label="Credit score" value={result.creditScore} tone="neutral" sub="Range 300–900" />
                    <StatCard icon={result.approval === 'Approved' ? CheckCircleIcon : XCircleIcon} label="Loan status" value={result.approval} tone={result.approval === 'Approved' ? 'good' : 'bad'} sub="Final outcome" />
                  </div>
                </div>
 
                {/* Key factors */}
                <div className="rounded-2xl sm:rounded-3xl border border-slate-200 bg-white p-4 sm:p-6 shadow-sm">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <h2 className="text-base sm:text-lg font-extrabold tracking-tight text-slate-900">Key factors</h2>
                      <p className="mt-0.5 text-xs sm:text-sm text-slate-500">Most influential features for this decision.</p>
                    </div>
                    <span className="shrink-0 inline-flex items-center rounded-full bg-slate-100 px-2.5 py-1 text-[10px] sm:text-xs font-semibold text-slate-600 ring-1 ring-slate-200">
                      Top {result.topFeatures?.length || 0}
                    </span>
                  </div>
                  <div className="mt-4 sm:mt-6 grid gap-2 sm:gap-3">
                    {result.topFeatures?.map((f, idx) => (
                      <FeatureRow key={idx} feature={f.feature} impact={f.impact} />
                    ))}
                  </div>
                </div>
 
                {/* AI Suggestions */}
                {result.aiSuggestions && <AISuggestionsCard suggestions={result.aiSuggestions} />}
 
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