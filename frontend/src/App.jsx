import { useState, useId, useEffect } from 'react'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:4000'

const SAMPLE_PRESETS = [
  {
    title: 'Billing Inquiry',
    subject: 'Double charged on monthly subscription invoice',
    body: 'I was billed twice on my credit card for invoice #9942 this month. Please review and issue a refund for the duplicate charge.',
  },
  {
    title: 'VPN / IT Issue',
    subject: 'VPN client drops connection every 10 minutes',
    body: 'My Cisco AnyConnect VPN client repeatedly disconnects while working on the office laptop. Error: Gateway unreachable.',
  },
  {
    title: 'HR Policy',
    subject: 'Request for annual PTO policy and holiday calendar',
    body: 'Where can I find the official company holiday schedule and updated parental leave documentation for this year?',
  },
  {
    title: 'Software Crash',
    subject: 'Application crashes immediately upon startup',
    body: 'After updating to the latest patch this morning, the desktop client crashes with a fatal unhandled exception.',
  },
]

function App() {
  const [subject, setSubject] = useState('')
  const [body, setBody] = useState('')
  const [loading, setLoading] = useState(false)
  const [validationError, setValidationError] = useState('')
  const [apiError, setApiError] = useState('')
  const [prediction, setPrediction] = useState(null)

  // Supabase ticket history state
  const [tickets, setTickets] = useState([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState('')

  const subjectId = useId()
  const bodyId = useId()

  const fetchTickets = async () => {
    setHistoryLoading(true)
    setHistoryError('')
    try {
      const response = await fetch(`${API_BASE_URL}/api/tickets`)
      const data = await response.json().catch(() => ({}))

      if (!response.ok) {
        const errorMsg = data.error || `Server responded with status ${response.status}`
        const errorDetails = typeof data.details === 'string' ? data.details : (data.details ? JSON.stringify(data.details) : '')
        setHistoryError(errorDetails ? `${errorMsg}: ${errorDetails}` : errorMsg)
        return
      }

      setTickets(Array.isArray(data) ? data : [])
    } catch (err) {
      setHistoryError(
        `Unable to fetch ticket history from ${API_BASE_URL}. (${err.message})`
      )
    } finally {
      setHistoryLoading(false)
    }
  }

  useEffect(() => {
    let ignore = false
    const load = async () => {
      setHistoryLoading(true)
      setHistoryError('')
      try {
        const response = await fetch(`${API_BASE_URL}/api/tickets`)
        const data = await response.json().catch(() => ({}))
        if (ignore) return

        if (!response.ok) {
          const errorMsg = data.error || `Server responded with status ${response.status}`
          const errorDetails = typeof data.details === 'string' ? data.details : (data.details ? JSON.stringify(data.details) : '')
          setHistoryError(errorDetails ? `${errorMsg}: ${errorDetails}` : errorMsg)
          return
        }

        setTickets(Array.isArray(data) ? data : [])
      } catch (err) {
        if (!ignore) {
          setHistoryError(`Unable to fetch ticket history from ${API_BASE_URL}. (${err.message})`)
        }
      } finally {
        if (!ignore) {
          setHistoryLoading(false)
        }
      }
    }

    load()
    return () => {
      ignore = true
    }
  }, [])

  const handlePredict = async (e) => {
    if (e) e.preventDefault()

    const cleanSubject = subject.trim()
    const cleanBody = body.trim()

    // 1. Client-side validation: at least one non-whitespace field required
    if (!cleanSubject && !cleanBody) {
      setValidationError('Please enter a ticket subject or description before predicting.')
      setPrediction(null)
      setApiError('')
      return
    }

    setValidationError('')
    setApiError('')
    setLoading(true)

    try {
      const response = await fetch(`${API_BASE_URL}/api/tickets/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          subject: cleanSubject || undefined,
          body: cleanBody || undefined,
        }),
      })

      const data = await response.json().catch(() => ({}))

      if (!response.ok) {
        // Detailed error message from backend, ML service, or database
        const errorMsg = data.error || `Server responded with status ${response.status}`
        const errorDetails = typeof data.details === 'string' ? data.details : (data.details ? JSON.stringify(data.details) : '')
        setApiError(errorDetails ? `${errorMsg}: ${errorDetails}` : errorMsg)
        setPrediction(null)
        return
      }

      setPrediction(data)

      // Refresh ticket history to display the newly saved ticket
      fetchTickets()
    } catch (err) {
      setApiError(
        `Unable to reach backend at ${API_BASE_URL}. Ensure the Express server is running on port 4000. (${err.message})`
      )
      setPrediction(null)
    } finally {
      setLoading(false)
    }
  }

  const handlePresetClick = (preset) => {
    setSubject(preset.subject)
    setBody(preset.body)
    setValidationError('')
    setApiError('')
  }

  const handleClear = () => {
    setSubject('')
    setBody('')
    setValidationError('')
    setApiError('')
    setPrediction(null)
  }

  // Format confidence to percentage string
  const formatConfidence = (conf) => {
    if (typeof conf !== 'number') return '0%'
    return `${(conf * 100).toFixed(1)}%`
  }

  // Format date helper
  const formatDate = (isoString) => {
    if (!isoString) return '—'
    try {
      const d = new Date(isoString)
      return d.toLocaleString(undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
    } catch {
      return isoString
    }
  }

  // Get sorted list of probabilities for the breakdown
  const sortedProbabilities = prediction?.probabilities
    ? Object.entries(prediction.probabilities)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 5)
    : []

  return (
    <div className="app-container">
      {/* Top Header */}
      <header className="app-header">
        <div className="header-brand">
          <div className="brand-logo">
            <span className="logo-glyph">RD</span>
          </div>
          <div>
            <h1 className="brand-title">RouteDesk</h1>
            <p className="brand-tagline">AI-Powered Support Ticket Triage</p>
          </div>
        </div>
        <div className="header-meta">
          <span className="badge badge-live">
            <span className="live-dot"></span> ML Baseline Active
          </span>
          <span className="badge badge-db">
            <span className="db-dot"></span> Supabase Connected
          </span>
          <span className="backend-url-tag">Gateway: {API_BASE_URL}</span>
        </div>
      </header>

      {/* Main Grid Content */}
      <main className="dashboard-grid">
        {/* Left Column: Ticket Input Form */}
        <section className="card card-form">
          <div className="card-header">
            <h2>New Support Ticket</h2>
            <p className="card-subtitle">
              Enter customer ticket details below to classify the target department.
            </p>
          </div>

          {/* Quick Presets */}
          <div className="presets-container">
            <span className="presets-label">Quick test templates:</span>
            <div className="presets-chips">
              {SAMPLE_PRESETS.map((p, idx) => (
                <button
                  key={idx}
                  type="button"
                  className="preset-chip"
                  onClick={() => handlePresetClick(p)}
                  disabled={loading}
                >
                  {p.title}
                </button>
              ))}
            </div>
          </div>

          <form onSubmit={handlePredict} className="ticket-form" noValidate>
            {/* Subject Input */}
            <div className="form-group">
              <label htmlFor={subjectId} className="form-label">
                Ticket Subject <span className="label-optional">(optional if description provided)</span>
              </label>
              <input
                id="ticket-subject-input"
                name="subject"
                type="text"
                className="form-input"
                placeholder="e.g., Cannot access billing invoice or portal"
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                disabled={loading}
              />
            </div>

            {/* Description Textarea */}
            <div className="form-group">
              <label htmlFor={bodyId} className="form-label">
                Ticket Description / Body <span className="label-optional">(optional if subject provided)</span>
              </label>
              <textarea
                id="ticket-body-input"
                name="body"
                className="form-textarea"
                rows="6"
                placeholder="Paste the full customer message, error logs, or problem description here..."
                value={body}
                onChange={(e) => setBody(e.target.value)}
                disabled={loading}
              />
            </div>

            {/* Validation Error Banner */}
            {validationError && (
              <div id="validation-error" className="banner banner-warning" role="alert">
                <span className="banner-icon">!</span>
                <div>
                  <strong>Input required:</strong> {validationError}
                </div>
              </div>
            )}

            {/* API / ML / DB Error Banner */}
            {apiError && (
              <div id="error-banner" className="banner banner-error" role="alert">
                <span className="banner-icon">✕</span>
                <div>
                  <strong>Service Error:</strong> {apiError}
                </div>
              </div>
            )}

            {/* Form Actions */}
            <div className="form-actions">
              <button
                id="predict-queue-btn"
                type="submit"
                className={`btn btn-primary ${loading ? 'btn-loading' : ''}`}
                disabled={loading}
              >
                {loading ? (
                  <>
                    <span className="spinner" aria-hidden="true"></span>
                    <span>Classifying & Archiving Ticket...</span>
                  </>
                ) : (
                  <>
                    <span className="btn-icon">⚡</span>
                    <span>Predict & Save Queue</span>
                  </>
                )}
              </button>

              <button
                id="clear-form-btn"
                type="button"
                className="btn btn-secondary"
                onClick={handleClear}
                disabled={loading || (!subject && !body && !prediction)}
              >
                Clear
              </button>
            </div>
          </form>
        </section>

        {/* Right Column: Prediction Results & Model Disclaimer */}
        <div className="results-column">
          {/* Prediction Result Card */}
          <section id="prediction-result" className="card card-result">
            <div className="card-header">
              <h2>Routing Recommendation</h2>
              <p className="card-subtitle">Predicted destination queue & model confidence</p>
            </div>

            {prediction ? (
              <div className="result-content">
                <div className="hero-queue-box">
                  <span className="queue-label-tag">Target Queue</span>
                  <div className="queue-name" id="predicted-queue-name">
                    {prediction.queue}
                  </div>

                  <div className="confidence-meter-container">
                    <div className="confidence-meta">
                      <span className="confidence-title">Model Confidence</span>
                      <span className="confidence-value" id="predicted-confidence-value">
                        {formatConfidence(prediction.confidence)}
                      </span>
                    </div>
                    <div className="meter-track">
                      <div
                        className="meter-fill"
                        style={{
                          width: `${Math.min(100, Math.max(5, (prediction.confidence || 0) * 100))}%`,
                        }}
                      ></div>
                    </div>
                  </div>
                </div>

                {/* Candidate Queues Breakdown */}
                {sortedProbabilities.length > 0 && (
                  <div className="breakdown-section">
                    <h3 className="breakdown-title">Top Candidate Queues</h3>
                    <ul className="breakdown-list">
                      {sortedProbabilities.map(([qName, prob]) => (
                        <li
                          key={qName}
                          className={`breakdown-item ${qName === prediction.queue ? 'item-winner' : ''}`}
                        >
                          <span className="candidate-name">{qName}</span>
                          <span className="candidate-prob">{(prob * 100).toFixed(1)}%</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ) : (
              <div className="placeholder-state">
                <div className="placeholder-icon">📋</div>
                <h3>No Ticket Evaluated Yet</h3>
                <p>
                  Fill in the ticket subject or description on the left and click <strong>Predict & Save Queue</strong> to view the routing prediction.
                </p>
              </div>
            )}
          </section>

          {/* ML Disclaimer & Model Transparency Card */}
          <section id="ml-disclaimer" className="card card-disclaimer">
            <div className="disclaimer-header">
              <span className="disclaimer-icon">ℹ️</span>
              <h3>ML Model Information & Transparency</h3>
            </div>
            <p className="disclaimer-text">
              Predictions are machine-learning generated using a <strong>TF-IDF + Logistic Regression</strong> baseline trained on the Hugging Face customer support tickets dataset.
            </p>
            <p className="disclaimer-subtext">
              Routing recommendations may occasionally be inaccurate or misclassified. Always verify before routing critical or high-priority escalations.
            </p>
            <div className="disclaimer-tags">
              <span className="tech-tag">TF-IDF N-grams</span>
              <span className="tech-tag">Logistic Regression</span>
              <span className="tech-tag">10 Target Queues</span>
              <span className="tech-tag">Supabase Archive</span>
            </div>
          </section>
        </div>
      </main>

      {/* Ticket History Section */}
      <section className="card card-history" id="ticket-history-section">
        <div className="history-header">
          <div>
            <div className="history-title-row">
              <h2>Ticket History</h2>
              <span className="badge badge-count" id="tickets-count-badge">
                {tickets.length} {tickets.length === 1 ? 'ticket' : 'tickets'}
              </span>
            </div>
            <p className="card-subtitle">
              Stored predictions retrieved newest first from Supabase (<span className="table-name-tag">public.tickets</span>).
            </p>
          </div>

          <button
            id="refresh-history-btn"
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={fetchTickets}
            disabled={historyLoading}
            title="Refresh history from database"
          >
            {historyLoading ? (
              <>
                <span className="spinner spinner-sm" aria-hidden="true"></span>
                <span>Refreshing...</span>
              </>
            ) : (
              <>
                <span className="btn-icon">↻</span>
                <span>Refresh History</span>
              </>
            )}
          </button>
        </div>

        {/* History Error Banner */}
        {historyError && (
          <div id="history-error-banner" className="banner banner-warning" role="alert">
            <span className="banner-icon">!</span>
            <div>
              <strong>Database status:</strong> {historyError}
            </div>
          </div>
        )}

        {/* History List or Table */}
        {historyLoading && tickets.length === 0 ? (
          <div className="history-loading-state">
            <span className="spinner" aria-hidden="true"></span>
            <p>Loading ticket history from database...</p>
          </div>
        ) : tickets.length === 0 ? (
          <div className="placeholder-state history-empty-state">
            <div className="placeholder-icon">🗄️</div>
            <h3>No Saved Tickets Yet</h3>
            <p>
              Submit a support ticket above. Once classified, it will be automatically archived into your Supabase database and displayed here.
            </p>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="tickets-table" id="tickets-history-table">
              <thead>
                <tr>
                  <th scope="col" className="th-id">ID</th>
                  <th scope="col" className="th-date">Logged At</th>
                  <th scope="col" className="th-subject">Subject</th>
                  <th scope="col" className="th-body">Description Preview</th>
                  <th scope="col" className="th-queue">Predicted Queue</th>
                  <th scope="col" className="th-confidence">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {tickets.map((t) => (
                  <tr key={t.id || `${t.created_at}-${t.subject}`}>
                    <td className="td-id">
                      <span className="ticket-id-badge">#{t.id}</span>
                    </td>
                    <td className="td-date">{formatDate(t.created_at)}</td>
                    <td className="td-subject">
                      <div className="ticket-subject-cell" title={t.subject || ''}>
                        {t.subject || <span className="cell-muted">—</span>}
                      </div>
                    </td>
                    <td className="td-body">
                      <div className="ticket-body-cell" title={t.body || ''}>
                        {t.body || <span className="cell-muted">—</span>}
                      </div>
                    </td>
                    <td className="td-queue">
                      <span className="queue-pill" title={t.predicted_queue}>
                        {t.predicted_queue}
                      </span>
                    </td>
                    <td className="td-confidence">
                      <span className="confidence-pill">
                        {formatConfidence(t.confidence)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Footer */}
      <footer className="app-footer">
        <p>RouteDesk Ticket Routing System &bull; 3-Tier Architecture (React + Express + FastAPI + Supabase)</p>
      </footer>
    </div>
  )
}

export default App
