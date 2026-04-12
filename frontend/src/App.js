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
} from '@heroicons/react/24/outline';

//const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000'; //
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

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [token, setToken] = useState('');
  const [authMode, setAuthMode] = useState('login'); // 'login' or 'register'
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

      const response = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
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
      setAuthError(err.message || `Unable to ${authMode}`);
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
        topFeatures: data.top_features
      });
    } catch (err) {
      setError(err.message || 'Unable to contact backend');
    } finally {
      setLoading(false);
    }
  };

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen px-4 py-10 sm:px-6 lg:px-8">
        <div className="mx-auto grid w-full max-w-5xl grid-cols-1 items-stretch gap-6 lg:grid-cols-2">
          <div className="surface rounded-3xl p-7 sm:p-9">
            <BrandMark />

            <div className="mt-8">
              <div className="inline-flex items-center gap-2 rounded-full bg-brand-50 px-3 py-1 text-xs font-semibold text-brand-800 ring-1 ring-brand-100">
                <ShieldCheckIcon className="h-4 w-4" />
                Secure decisioning with explainability
              </div>

              <h1 className="mt-4 text-3xl font-extrabold tracking-tight text-slate-900 sm:text-4xl">
                {authMode === 'login' ? 'Welcome back' : 'Create your account'}
              </h1>
              <p className="mt-2 text-sm leading-6 text-slate-600">
                {authMode === 'login'
                  ? 'Sign in to generate a credit score and see the most important factors behind the decision.'
                  : 'Register to access predictions and a clear breakdown of what influenced the outcome.'}
              </p>
            </div>

            <form className="mt-8 grid gap-4" onSubmit={handleAuth}>
              <div className="grid gap-2">
                <label className="text-xs font-semibold text-slate-700">Username</label>
                <div className="relative">
                  <UserCircleIcon className="pointer-events-none absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-slate-400" />
                  <input
                    name="username"
                    type="text"
                    required
                    className="input pl-10"
                    placeholder="admin"
                    value={authForm.username}
                    onChange={handleAuthChange}
                  />
                </div>
              </div>

              {authMode === 'register' && (
                <div className="grid gap-2">
                  <label className="text-xs font-semibold text-slate-700">Email</label>
                  <input
                    name="email"
                    type="email"
                    required
                    className="input"
                    placeholder="name@example.com"
                    value={authForm.email}
                    onChange={handleAuthChange}
                  />
                </div>
              )}

              <div className="grid gap-2">
                <label className="text-xs font-semibold text-slate-700">Password</label>
                <div className="relative">
                  <LockClosedIcon className="pointer-events-none absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-slate-400" />
                  <input
                    name="password"
                    type="password"
                    required
                    className="input pl-10"
                    placeholder={authMode === 'login' ? 'admin123' : 'Minimum 6 characters'}
                    value={authForm.password}
                    onChange={handleAuthChange}
                  />
                </div>
              </div>

              {authError && (
                <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
                  {authError}
                </div>
              )}

              <button type="submit" disabled={authLoading} className="btn-primary">
                {authLoading ? 'Processing…' : (authMode === 'login' ? 'Sign in' : 'Create account')}
                <ArrowRightOnRectangleIcon className="h-4 w-4" />
              </button>

              <div className="text-center text-sm text-slate-600">
                {authMode === 'login' ? "Don't have an account?" : 'Already have an account?'}{' '}
                <button
                  type="button"
                  onClick={() => setAuthMode(authMode === 'login' ? 'register' : 'login')}
                  className="font-semibold text-brand-800 hover:text-brand-900"
                >
                  {authMode === 'login' ? 'Sign up' : 'Sign in'}
                </button>
              </div>

              <div className="mt-2 grid gap-3 rounded-2xl border border-slate-200 bg-white/60 p-4">
                <div className="text-xs font-semibold text-slate-700">Demo credentials</div>
                <div className="grid grid-cols-1 gap-2 text-xs text-slate-600 sm:grid-cols-2">
                  <div className="rounded-xl bg-slate-50 px-3 py-2 ring-1 ring-slate-200">
                    <div className="font-semibold text-slate-900">admin</div>
                    <div>admin123</div>
                  </div>
                  <div className="rounded-xl bg-slate-50 px-3 py-2 ring-1 ring-slate-200">
                    <div className="font-semibold text-slate-900">user</div>
                    <div>user123</div>
                  </div>
                </div>
              </div>
            </form>
          </div>

          <div className="surface hidden rounded-3xl p-7 sm:p-9 lg:block">
            <div className="flex items-center justify-between">
              <div className="chip bg-accent-100 text-accent-900 ring-1 ring-accent-200">
                <SparklesIcon className="mr-1 h-4 w-4" />
                Smart scoring
              </div>
              <div className="chip bg-brand-50 text-brand-900 ring-1 ring-brand-100">
                <ChartBarIcon className="mr-1 h-4 w-4" />
                Explainable AI
              </div>
            </div>

            <div className="mt-8">
              <h2 className="text-xl font-extrabold tracking-tight text-slate-900">
                Professional, modern and consistent branding
              </h2>
              <p className="mt-2 text-sm leading-6 text-slate-600">
                Clean spacing, refined typography, and an accessible color system that matches your reference (teal + lime).
              </p>
            </div>

            <div className="mt-7 grid gap-4">
              <StatCard icon={CheckCircleIcon} label="Approval outcome" value="Approved / Rejected" tone="brand" sub="Clear status at a glance" />
              <div className="grid grid-cols-2 gap-4">
                <StatCard icon={ShieldCheckIcon} label="Risk tier" value="Low / Med / High" tone="neutral" />
                <StatCard icon={ChartBarIcon} label="Confidence" value="0–100%" tone="neutral" />
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white p-5">
                <div className="text-xs font-semibold text-slate-700">Top factors</div>
                <div className="mt-3 grid gap-3">
                  <FeatureRow feature="Credit_History" impact={0.214} />
                  <FeatureRow feature="ApplicantIncome" impact={0.132} />
                  <FeatureRow feature="LoanAmount" impact={-0.087} />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/70 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-4 sm:px-6 lg:px-8">
          <BrandMark compact />
          <div className="flex items-center gap-3">
            <div className="hidden items-center gap-2 rounded-2xl bg-slate-50 px-3 py-2 text-xs font-semibold text-slate-700 ring-1 ring-slate-200 sm:flex">
              <UserCircleIcon className="h-4 w-4 text-slate-500" />
              Signed in
            </div>
            <button onClick={handleLogout} className="btn-secondary">
              Logout
              <ArrowRightOnRectangleIcon className="h-4 w-4" />
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
          <section className="surface rounded-3xl p-6 lg:col-span-2">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-lg font-extrabold tracking-tight text-slate-900">
                  Loan application
                </h2>
                <p className="mt-1 text-sm text-slate-600">
                  Enter a few details. We’ll return approval, credit score and the top factors.
                </p>
              </div>
              <div className="chip bg-accent-100 text-accent-900 ring-1 ring-accent-200">
                <SparklesIcon className="mr-1 h-4 w-4" />
                Pro UI
              </div>
            </div>

            <form onSubmit={handleSubmit} className="mt-6 grid gap-4">
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div className="grid gap-2">
                  <label htmlFor="ApplicantIncome" className="text-xs font-semibold text-slate-700">
                    Annual income
                  </label>
                  <input
                    id="ApplicantIncome"
                    name="ApplicantIncome"
                    type="number"
                    required
                    className="input"
                    placeholder="50000"
                    value={formData.ApplicantIncome}
                    onChange={handleChange}
                  />
                </div>

                <div className="grid gap-2">
                  <label htmlFor="LoanAmount" className="text-xs font-semibold text-slate-700">
                    Loan amount
                  </label>
                  <input
                    id="LoanAmount"
                    name="LoanAmount"
                    type="number"
                    required
                    className="input"
                    placeholder="200000"
                    value={formData.LoanAmount}
                    onChange={handleChange}
                  />
                </div>

                <div className="grid gap-2">
                  <label htmlFor="Credit_History" className="text-xs font-semibold text-slate-700">
                    Credit history
                  </label>
                  <select
                    id="Credit_History"
                    name="Credit_History"
                    className="select"
                    value={formData.Credit_History}
                    onChange={handleChange}
                  >
                    <option value="1.0">Good (1.0)</option>
                    <option value="0.0">Poor (0.0)</option>
                  </select>
                </div>

                <div className="grid gap-2">
                  <label htmlFor="Dependents" className="text-xs font-semibold text-slate-700">
                    Dependents
                  </label>
                  <select
                    id="Dependents"
                    name="Dependents"
                    className="select"
                    value={formData.Dependents}
                    onChange={handleChange}
                  >
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
                <div className="text-xs font-semibold text-slate-700">What you’ll get</div>
                <div className="mt-3 grid gap-2 text-sm text-slate-700">
                  <div className="flex items-center gap-2">
                    <CheckCircleIcon className="h-5 w-5 text-emerald-600" />
                    Approval status & risk tier
                  </div>
                  <div className="flex items-center gap-2">
                    <ChartBarIcon className="h-5 w-5 text-brand-700" />
                    Credit score and probability
                  </div>
                  <div className="flex items-center gap-2">
                    <ShieldCheckIcon className="h-5 w-5 text-slate-600" />
                    Key factors with impacts
                  </div>
                </div>
              </div>
            </form>
          </section>

          <section className="lg:col-span-3">
            {!result ? (
              <div className="surface flex h-full min-h-[420px] flex-col justify-center rounded-3xl p-8 text-center">
                <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-brand-50 ring-1 ring-brand-100">
                  <ChartBarIcon className="h-7 w-7 text-brand-700" />
                </div>
                <h3 className="mt-4 text-xl font-extrabold tracking-tight text-slate-900">
                  Results will appear here
                </h3>
                <p className="mt-2 text-sm text-slate-600">
                  Submit the form to generate a prediction, credit score, and top factors.
                </p>
              </div>
            ) : (
              <div className="grid gap-6">
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
                    <StatCard
                      icon={ShieldCheckIcon}
                      label="Risk level"
                      value={result.risk.level}
                      tone={result.risk.level === 'Low' ? 'good' : result.risk.level === 'Medium' ? 'warn' : 'bad'}
                      sub="Higher risk may lower approval odds"
                    />
                    <StatCard
                      icon={ChartBarIcon}
                      label="Approval probability"
                      value={`${Math.round(result.probability * 100)}%`}
                      tone="brand"
                      sub="Model confidence"
                    />
                    <StatCard
                      icon={SparklesIcon}
                      label="Credit score"
                      value={result.creditScore}
                      tone="neutral"
                      sub="Range 300–900"
                    />
                    <StatCard
                      icon={result.approval === 'Approved' ? CheckCircleIcon : XCircleIcon}
                      label="Loan status"
                      value={result.approval}
                      tone={result.approval === 'Approved' ? 'good' : 'bad'}
                      sub="Final predicted outcome"
                    />
                  </div>
                </div>

                <div className="surface rounded-3xl p-6">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <h2 className="text-lg font-extrabold tracking-tight text-slate-900">Key factors</h2>
                      <p className="mt-1 text-sm text-slate-600">
                        The most influential features for this decision.
                      </p>
                    </div>
                    <div className="chip bg-slate-50 text-slate-800 ring-1 ring-slate-200">
                      Top {result.topFeatures?.length || 0}
                    </div>
                  </div>

                  <div className="mt-6 grid gap-3">
                    {result.topFeatures?.map((f, idx) => (
                      <FeatureRow key={idx} feature={f.feature} impact={f.impact} />
                    ))}
                  </div>
                </div>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}

export default App;
