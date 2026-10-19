import { useState, useId, useEffect } from 'react'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:4000'

const SAMPLE_PRESETS = [
  {
    title: 'Billing Issue',
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

const AVAILABLE_QUEUES = [
  'Billing and Payments',
  'Customer Service',
  'General Inquiry',
  'Human Resources',
  'IT Support',
  'Product Support',
  'Returns and Exchanges',
  'Sales and Pre-Sales',
  'Service Outages and Maintenance',
  'Technical Support',
]

// Provisional UI heuristic threshold for low-confidence routing (not a validated decision boundary)
const LOW_CONFIDENCE_THRESHOLD = 0.40

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

  // System health verification state (live verified against real health endpoints)
  const [systemHealth, setSystemHealth] = useState({
    backend: null,
    ml_classifier: null,
    ml_embeddings: null,
    supabase: null,
  })

  // Ticket prediction feedback state (Step 9)
  const [feedbackSubmittingMap, setFeedbackSubmittingMap] = useState({})
  const [feedbackErrorMap, setFeedbackErrorMap] = useState({})
  const [reassigningTicketId, setReassigningTicketId] = useState(null)
  const [selectedQueueMap, setSelectedQueueMap] = useState({})

  // Semantic duplicate detection state
  const [duplicateDetection, setDuplicateDetection] = useState(null)

  // Model evaluation metrics state (Step 8: Model Evaluation & Explainability)
  const [modelMetrics, setModelMetrics] = useState(null)
  const [metricsLoading, setMetricsLoading] = useState(false)
  const [metricsError, setMetricsError] = useState('')

  const subjectId = useId()
  const bodyId = useId()

  const fetchHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`)
      if (response.ok) {
        const data = await response.json().catch(() => ({}))
        setSystemHealth({
          backend: data.services?.backend?.status || 'online',
          ml_classifier: data.services?.ml_classifier?.status || 'offline',
          ml_embeddings: data.services?.ml_embeddings?.status || 'offline',
          supabase: data.services?.supabase?.status || 'offline',
        })
        return
      }
      setSystemHealth({ backend: 'degraded', ml_classifier: 'offline', ml_embeddings: 'offline', supabase: 'offline' })
    } catch {
      setSystemHealth({ backend: 'offline', ml_classifier: 'offline', ml_embeddings: 'offline', supabase: 'offline' })
    }
  }

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

  const fetchMetrics = async () => {
    setMetricsLoading(true)
    setMetricsError('')
    try {
      const response = await fetch(`${API_BASE_URL}/api/tickets/metrics`)
      const data = await response.json().catch(() => ({}))

      if (!response.ok) {
        const errorMsg = data.error || `Server responded with status ${response.status}`
        const errorDetails = typeof data.details === 'string' ? data.details : (data.details ? JSON.stringify(data.details) : '')
        setMetricsError(errorDetails ? `${errorMsg}: ${errorDetails}` : errorMsg)
        return
      }

      setModelMetrics(data)
    } catch (err) {
      setMetricsError(`Unable to fetch model metrics: ${err.message}`)
    } finally {
      setMetricsLoading(false)
    }
  }

  useEffect(() => {
    let ignore = false
    const load = async () => {
      setHistoryLoading(true)
      setHistoryError('')
      setMetricsLoading(true)
      setMetricsError('')

      try {
        const [ticketsRes, metricsRes, healthRes] = await Promise.allSettled([
          fetch(`${API_BASE_URL}/api/tickets`),
          fetch(`${API_BASE_URL}/api/tickets/metrics`),
          fetch(`${API_BASE_URL}/health`),
        ])

        if (ignore) return

        // Handle health response
        if (healthRes.status === 'fulfilled' && healthRes.value.ok) {
          const hData = await healthRes.value.json().catch(() => ({}))
          setSystemHealth({
            backend: hData.services?.backend?.status || 'online',
            ml_classifier: hData.services?.ml_classifier?.status || 'offline',
            ml_embeddings: hData.services?.ml_embeddings?.status || 'offline',
            supabase: hData.services?.supabase?.status || 'offline',
          })
        } else {
          setSystemHealth({ backend: 'offline', ml_classifier: 'offline', ml_embeddings: 'offline', supabase: 'offline' })
        }

        // Handle tickets response
        if (ticketsRes.status === 'fulfilled') {
          const tData = await ticketsRes.value.json().catch(() => ({}))
          if (ticketsRes.value.ok) {
            setTickets(Array.isArray(tData) ? tData : [])
          } else {
            const errorMsg = tData.error || `Server error (${ticketsRes.value.status})`
            setHistoryError(errorMsg)
          }
        } else {
          setHistoryError(`Unable to reach backend at ${API_BASE_URL}`)
        }

        // Handle metrics response
        if (metricsRes.status === 'fulfilled') {
          const mData = await metricsRes.value.json().catch(() => ({}))
          if (metricsRes.value.ok) {
            setModelMetrics(mData)
          } else {
            const errorMsg = mData.error || `Metrics error (${metricsRes.value.status})`
            setMetricsError(errorMsg)
          }
        } else {
          setMetricsError(`Unable to reach metrics service`)
        }
      } catch (err) {
        if (!ignore) {
          setHistoryError(err.message)
          setMetricsError(err.message)
        }
      } finally {
        if (!ignore) {
          setHistoryLoading(false)
          setMetricsLoading(false)
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
        setDuplicateDetection(null)
        return
      }

      setPrediction(data)
      setDuplicateDetection(data.duplicate_detection || null)

      // Refresh ticket history to display the newly saved ticket
      fetchTickets()
    } catch (err) {
      setApiError(
        `Unable to reach backend at ${API_BASE_URL}. Ensure the Express server is running on port 4000. (${err.message})`
      )
      setPrediction(null)
      setDuplicateDetection(null)
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
    setDuplicateDetection(null)
  }

  // Handle agent triage feedback submission (Step 9)
  const handleSubmitFeedback = async (ticketId, isCorrect, correctedQueue) => {
    if (!ticketId) return

    setFeedbackSubmittingMap((prev) => ({ ...prev, [ticketId]: true }))
    setFeedbackErrorMap((prev) => ({ ...prev, [ticketId]: '' }))

    try {
      const response = await fetch(`${API_BASE_URL}/api/tickets/${ticketId}/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          is_correct: isCorrect,
          corrected_queue: isCorrect ? null : correctedQueue,
        }),
      })

      const data = await response.json().catch(() => ({}))

      if (!response.ok) {
        const errorMsg = data.error || `Server responded with status ${response.status}`
        const errorDetails = typeof data.details === 'string' ? data.details : (data.details ? JSON.stringify(data.details) : '')
        setFeedbackErrorMap((prev) => ({ ...prev, [ticketId]: errorDetails ? `${errorMsg}: ${errorDetails}` : errorMsg }))
        return
      }

      // 1. Update tickets state locally with the new feedback
      setTickets((prev) =>
        prev.map((t) =>
          t.id === ticketId
            ? {
                ...t,
                ticket_feedback: data.feedback,
              }
            : t
        )
      )

      // 2. Also update prediction hero ticket if it matches
      setPrediction((prev) => {
        if (prev && prev.ticket && prev.ticket.id === ticketId) {
          return {
            ...prev,
            ticket: {
              ...prev.ticket,
              ticket_feedback: data.feedback,
            },
          }
        }
        return prev
      })

      // 3. Close inline reassignment form
      setReassigningTicketId(null)
    } catch (err) {
      setFeedbackErrorMap((prev) => ({
        ...prev,
        [ticketId]: `Network error: ${err.message}`,
      }))
    } finally {
      setFeedbackSubmittingMap((prev) => ({ ...prev, [ticketId]: false }))
    }
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
            <span className="logo-glyph">RouteDesk</span>
          </div>
          <h1 className="brand-tagline">AI-Powered Support Ticket Triage</h1>
        </div>
        <div className="header-meta">
          <span
            className={`badge ${
              systemHealth.ml_classifier === 'online'
                ? 'badge-live'
                : systemHealth.ml_classifier === 'offline'
                ? 'badge-warning'
                : 'badge-neutral'
            }`}
            id="header-ml-badge"
          >
            <span
              className={`status-dot ${
                systemHealth.ml_classifier === 'online' ? 'live-dot' : 'warn-dot'
              }`}
            ></span>
            {systemHealth.ml_classifier === 'online'
              ? 'Classifier Online'
              : systemHealth.ml_classifier === 'offline'
              ? 'Classifier Offline'
              : 'Checking Classifier...'}
          </span>

          <span
            className={`badge ${
              systemHealth.ml_embeddings === 'online'
                ? 'badge-live'
                : systemHealth.ml_embeddings === 'offline'
                ? 'badge-warning'
                : 'badge-neutral'
            }`}
            id="header-embeddings-badge"
          >
            <span
              className={`status-dot ${
                systemHealth.ml_embeddings === 'online' ? 'live-dot' : 'warn-dot'
              }`}
            ></span>
            {systemHealth.ml_embeddings === 'online'
              ? 'Duplicate Detection Ready'
              : systemHealth.ml_embeddings === 'offline'
              ? 'Duplicate Detection Offline'
              : 'Checking Duplicates...'}
          </span>

          <span
            className={`badge ${
              systemHealth.supabase === 'online'
                ? 'badge-live'
                : systemHealth.supabase === 'offline'
                ? 'badge-warning'
                : 'badge-neutral'
            }`}
            id="header-supabase-badge"
          >
            <span
              className={`status-dot ${
                systemHealth.supabase === 'online' ? 'live-dot' : 'warn-dot'
              }`}
            ></span>
            {systemHealth.supabase === 'online'
              ? 'Database Connected'
              : systemHealth.supabase === 'offline'
              ? 'Database Offline'
              : 'Checking Database...'}
          </span>
        </div>
      </header>

      {/* Main Grid Content */}
      <main className="dashboard-grid">
        {/* Left Column: Ticket Input Form */}
        <section className="card card-form">
          <div className="card-header">
            <h2>New Support Ticket</h2>
            <p className="card-subtitle">
              Enter a ticket's subject or description to get a recommended support team.
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
                Ticket Subject <span className="label-optional">(optional if description is provided)</span>
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
                Ticket Description <span className="label-optional">(optional if subject is provided)</span>
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
                    <span>Classifying & saving ticket...</span>
                  </>
                ) : (
                  <>
                    <span className="btn-icon">⚡</span>
                    <span>Predict & Save Ticket</span>
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
              <p className="card-subtitle">
                The model recommends a support team and provides a confidence score for its prediction.
              </p>
            </div>

            {prediction ? (
              <div className="result-content">
                <div className="hero-queue-box">
                  <span className="queue-label-tag">Recommended Support Team</span>
                  <div className="queue-name" id="predicted-queue-name">
                    {prediction.queue}
                  </div>

                  <div className="confidence-meter-container">
                    <div className="confidence-meta">
                      <span className="confidence-title">Model Confidence</span>
                      <div className="confidence-value-group">
                        <span className="confidence-value" id="predicted-confidence-value">
                          {formatConfidence(prediction.confidence)}
                        </span>
                        {prediction.confidence < LOW_CONFIDENCE_THRESHOLD ? (
                          <span
                            className="confidence-review-tag tag-warning"
                            title="Predictions below 40% confidence need human review. The 40% threshold is a provisional review threshold, not a scientifically validated boundary."
                          >
                            Needs review (&lt;40%)
                          </span>
                        ) : (
                          <span
                            className="confidence-review-tag tag-normal"
                            title="Standard routing confidence (≥40%)"
                          >
                            Standard confidence (&ge;40%)
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="meter-track">
                      <div
                        className={`meter-fill ${
                          prediction.confidence < LOW_CONFIDENCE_THRESHOLD
                            ? 'meter-fill-warning'
                            : ''
                        }`}
                        style={{
                          width: `${Math.min(100, Math.max(5, (prediction.confidence || 0) * 100))}%`,
                        }}
                      ></div>
                    </div>

                    {prediction.confidence < LOW_CONFIDENCE_THRESHOLD && (
                      <div className="confidence-warning-alert" role="alert">
                        <strong>Needs review:</strong> This prediction has a confidence score below 40%. Please review it before assigning the ticket.
                        <div className="confidence-warning-subtext">
                          The 40% threshold is a provisional review threshold, not a scientifically validated decision boundary.
                        </div>
                      </div>
                    )}

                    <div className="confidence-explain-note" id="confidence-explain-note">
                      <span className="confidence-explain-icon">ℹ️</span>
                      <p className="confidence-explain-text">
                        The model's confidence score is not a calibrated probability and should not be interpreted as the exact chance that a prediction is correct.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Candidate Queues Breakdown */}
                {sortedProbabilities.length > 0 && (
                  <div className="breakdown-section">
                    <h3 className="breakdown-title">Other Possible Support Teams</h3>
                    <p className="breakdown-caption">
                      These are alternative model predictions, not confirmed answers.
                    </p>
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

                {/* Agent Triage Feedback Panel (Step 9) */}
                {prediction.ticket?.id && (() => {
                  const activeFeedback = tickets.find((t) => t.id === prediction.ticket.id)?.ticket_feedback || prediction.ticket.ticket_feedback
                  return (
                    <div className="prediction-feedback-panel" id="prediction-feedback-panel">
                      <div className="feedback-panel-header">
                        <span className="feedback-panel-icon">⚖️</span>
                        <div className="feedback-panel-title-group">
                          <h4>Feedback for Ticket #{prediction.ticket.id}</h4>
                          <span className="feedback-panel-caption">
                            Help evaluate the model by confirming or correcting this recommendation.
                          </span>
                        </div>
                      </div>

                      {activeFeedback ? (
                        <div className="feedback-status-badge-container">
                          {activeFeedback.is_correct ? (
                            <div className="badge-feedback-confirmed">
                              <span className="feedback-badge-icon">✓</span>
                              <span><strong>Confirmed:</strong> Recommended team is <em>{activeFeedback.predicted_queue}</em></span>
                            </div>
                          ) : (
                            <div className="badge-feedback-reassigned">
                              <span className="feedback-badge-icon">✗</span>
                              <span><strong>Correction Submitted:</strong> Assigned to <em>{activeFeedback.corrected_queue}</em> (Originally recommended: <em>{activeFeedback.predicted_queue}</em>)</span>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="feedback-action-container">
                          {reassigningTicketId === prediction.ticket.id ? (
                            <div className="reassign-control-group">
                              <label htmlFor={`hero-reassign-select-${prediction.ticket.id}`} className="reassign-label">
                                Select Correct Support Team:
                              </label>
                              <div className="reassign-input-row">
                                <select
                                  id={`hero-reassign-select-${prediction.ticket.id}`}
                                  className="form-select reassign-select"
                                  value={selectedQueueMap[prediction.ticket.id] || ''}
                                  onChange={(e) =>
                                    setSelectedQueueMap((prev) => ({
                                      ...prev,
                                      [prediction.ticket.id]: e.target.value,
                                    }))
                                  }
                                  disabled={feedbackSubmittingMap[prediction.ticket.id]}
                                >
                                  <option value="">Choose team...</option>
                                  {AVAILABLE_QUEUES.map((q) => (
                                    <option key={q} value={q} disabled={q === prediction.queue}>
                                      {q} {q === prediction.queue ? '(recommended)' : ''}
                                    </option>
                                  ))}
                                </select>

                                <button
                                  type="button"
                                  className="btn btn-primary btn-sm"
                                  onClick={() =>
                                    handleSubmitFeedback(
                                      prediction.ticket.id,
                                      false,
                                      selectedQueueMap[prediction.ticket.id]
                                    )
                                  }
                                  disabled={
                                    feedbackSubmittingMap[prediction.ticket.id] ||
                                    !selectedQueueMap[prediction.ticket.id]
                                  }
                                >
                                  {feedbackSubmittingMap[prediction.ticket.id] ? 'Saving...' : 'Confirm'}
                                </button>

                                <button
                                  type="button"
                                  className="btn btn-secondary btn-sm"
                                  onClick={() => setReassigningTicketId(null)}
                                  disabled={feedbackSubmittingMap[prediction.ticket.id]}
                                >
                                  Cancel
                                </button>
                              </div>
                            </div>
                          ) : (
                            <div className="feedback-buttons-row">
                              <button
                                type="button"
                                id="hero-feedback-correct-btn"
                                className="btn btn-feedback-correct"
                                onClick={() => handleSubmitFeedback(prediction.ticket.id, true, null)}
                                disabled={feedbackSubmittingMap[prediction.ticket.id]}
                                title="Confirm the recommended team is correct"
                              >
                                {feedbackSubmittingMap[prediction.ticket.id] ? 'Saving...' : '✓ Confirm Correct'}
                              </button>

                              <button
                                type="button"
                                id="hero-feedback-wrong-btn"
                                className="btn btn-feedback-wrong"
                                onClick={() => {
                                  setReassigningTicketId(prediction.ticket.id)
                                  setFeedbackErrorMap((prev) => ({ ...prev, [prediction.ticket.id]: '' }))
                                }}
                                disabled={feedbackSubmittingMap[prediction.ticket.id]}
                                title="Mark as incorrect and select the correct support team"
                              >
                                ✗ Mark Incorrect
                              </button>
                            </div>
                          )}

                          {feedbackErrorMap[prediction.ticket.id] && (
                            <div className="feedback-error-banner" role="alert">
                              {feedbackErrorMap[prediction.ticket.id]}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )
                })()}
              </div>
            ) : (
              <div className="placeholder-state">
                <div className="placeholder-icon">📋</div>
                <p>
                  No prediction yet. Enter a ticket and select <strong>Predict & Save Ticket</strong>.
                </p>
              </div>
            )}
          </section>

          {/* Similar Ticket Check Card */}
          <section id="duplicate-detection-result" className="card card-duplicates">
            <div className="card-header">
              <h2>Similar Ticket Check</h2>
              <p className="card-subtitle">
                Checks whether a previously submitted ticket has similar wording or meaning. Similarity threshold: 55%
              </p>
            </div>

            {duplicateDetection ? (
              <div className="duplicates-content">
                {duplicateDetection.error || duplicateDetection.warning ? (
                  <div className="duplicate-alert-box duplicate-error-box" id="duplicate-error-box">
                    <div className="duplicate-alert-header">
                      <span className="alert-icon-symbol">⚠️</span>
                      <div>
                        <strong>Similar Ticket Check Unavailable</strong>
                        <p className="alert-subtext">
                          {duplicateDetection.error || duplicateDetection.warning || 'The duplicate check could not be completed.'}
                        </p>
                      </div>
                    </div>
                  </div>
                ) : duplicateDetection.has_duplicates ? (
                  <div className="duplicate-alert-box" id="duplicate-alert-box">
                    <div className="duplicate-alert-header">
                      <span className="alert-icon-symbol">⚠️</span>
                      <div>
                        <strong>Similar Ticket Found</strong>
                        <p className="alert-subtext">
                          A previously submitted ticket appears similar to this one. Review the match to decide whether both tickets describe the same issue.
                        </p>
                      </div>
                    </div>

                    <div className="matched-tickets-list" id="matched-duplicates-list">
                      {duplicateDetection.duplicates
                        .filter((d) => d.similarity_score >= (duplicateDetection.threshold || 0.55))
                        .map((dup) => (
                          <div key={dup.id} className="matched-ticket-item">
                            <div className="matched-meta">
                              <span className="similarity-badge">
                                Similarity score: {(dup.similarity_score * 100).toFixed(1)}% (Similarity threshold: 55%)
                              </span>
                              <span className="ticket-ref-tag">Ticket #{dup.id}</span>
                              {dup.predicted_queue && (
                                <span className="matched-queue-tag">{dup.predicted_queue}</span>
                              )}
                            </div>
                            <h4 className="matched-subject">{dup.subject || 'No subject'}</h4>
                            <p className="matched-body-preview">{dup.body || 'No description'}</p>
                          </div>
                        ))}
                    </div>
                  </div>
                ) : (
                  <div className="duplicate-clean-box" id="duplicate-clean-box">
                    <span className="clean-icon-symbol">✓</span>
                    <div>
                      <strong>No Similar Ticket Found Above the Threshold</strong>
                      <p className="clean-subtext">
                        No existing ticket met the configured similarity threshold. Lower-scoring matches may still exist.
                        {duplicateDetection.duplicates && duplicateDetection.duplicates.length > 0 && (
                          <span> Highest similarity score observed: <strong>{(duplicateDetection.duplicates[0].similarity_score * 100).toFixed(1)}%</strong>.</span>
                        )}
                      </p>
                    </div>
                  </div>
                )}

                <div className="duplicate-advisory-note">
                  <span className="advisory-icon">ℹ️</span>
                  <p className="advisory-text">
                    Similar tickets are suggestions for human review. RouteDesk does not automatically reject or merge tickets.
                  </p>
                </div>
              </div>
            ) : (
              <div className="placeholder-state placeholder-sm">
                <div className="placeholder-icon">🔍</div>
                <p>
                  Similar-ticket results will appear after you submit a ticket.
                </p>
              </div>
            )}
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
              Previously submitted tickets, newest first.
            </p>
          </div>

          <button
            id="refresh-history-btn"
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={() => {
              fetchTickets()
              fetchHealth()
            }}
            disabled={historyLoading}
            title="Refresh ticket history from database"
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
              Submit a support ticket above. Once classified, it will be saved to your database and displayed here.
            </p>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="tickets-table" id="tickets-history-table">
              <thead>
                <tr>
                  <th scope="col" className="th-id">Ticket ID</th>
                  <th scope="col" className="th-date">Date and Time</th>
                  <th scope="col" className="th-subject">Subject</th>
                  <th scope="col" className="th-body">Description</th>
                  <th scope="col" className="th-queue">Recommended Team</th>
                  <th scope="col" className="th-confidence">
                    <div className="th-confidence-title">
                      <span>Confidence</span>
                      <span
                        className="info-icon"
                        title="Confidence scores show how strongly the model favours its prediction. They are not calibrated probabilities, so they should not be interpreted as the exact chance that a prediction is correct."
                        aria-label="Confidence score explanation"
                      >
                        ℹ️
                      </span>
                    </div>
                    <span className="confidence-legend">Below 40%: review recommended.</span>
                  </th>
                  <th scope="col" className="th-feedback">Feedback</th>
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
                      {typeof t.confidence === 'number' && t.confidence < LOW_CONFIDENCE_THRESHOLD ? (
                        <span
                          className="confidence-pill confidence-low"
                          title="Provisional review threshold (<40%): low confidence, human review recommended"
                        >
                          {formatConfidence(t.confidence)}
                        </span>
                      ) : (
                        <span className="confidence-pill">
                          {formatConfidence(t.confidence)}
                        </span>
                      )}
                    </td>
                    <td className="td-feedback">
                      {t.ticket_feedback ? (
                        <div className="table-feedback-result">
                          {t.ticket_feedback.is_correct ? (
                            <span className="badge badge-feedback-correct" title="Recommendation confirmed">
                              <span className="feedback-icon">✓</span> Confirmed
                            </span>
                          ) : (
                            <div className="table-feedback-corrected">
                              <span
                                className="badge badge-feedback-incorrect"
                                title={`Reassigned to ${t.ticket_feedback.corrected_queue} (Originally: ${t.ticket_feedback.predicted_queue})`}
                              >
                                <span className="feedback-icon">✗</span> Correction Submitted: {t.ticket_feedback.corrected_queue}
                              </span>
                              <span className="table-feedback-orig-label">Originally: {t.ticket_feedback.predicted_queue}</span>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="table-feedback-actions">
                          {reassigningTicketId === t.id ? (
                            <div className="table-reassign-form">
                              <select
                                className="table-reassign-select"
                                value={selectedQueueMap[t.id] || ''}
                                onChange={(e) =>
                                  setSelectedQueueMap((prev) => ({
                                    ...prev,
                                    [t.id]: e.target.value,
                                  }))
                                }
                                disabled={feedbackSubmittingMap[t.id]}
                                aria-label={`Select correct support team for ticket #${t.id}`}
                              >
                                <option value="">Select team...</option>
                                {AVAILABLE_QUEUES.map((q) => (
                                  <option key={q} value={q} disabled={q === t.predicted_queue}>
                                    {q} {q === t.predicted_queue ? '(recommended)' : ''}
                                  </option>
                                ))}
                              </select>
                              <div className="table-reassign-buttons">
                                <button
                                  type="button"
                                  className="btn btn-primary btn-xs"
                                  onClick={() =>
                                    handleSubmitFeedback(t.id, false, selectedQueueMap[t.id])
                                  }
                                  disabled={feedbackSubmittingMap[t.id] || !selectedQueueMap[t.id]}
                                  title="Save corrected team"
                                >
                                  {feedbackSubmittingMap[t.id] ? '...' : 'Save'}
                                </button>
                                <button
                                  type="button"
                                  className="btn btn-secondary btn-xs"
                                  onClick={() => setReassigningTicketId(null)}
                                  disabled={feedbackSubmittingMap[t.id]}
                                  title="Cancel reassignment"
                                >
                                  Cancel
                                </button>
                              </div>
                            </div>
                          ) : (
                            <div className="table-feedback-btn-group">
                              <button
                                type="button"
                                className="btn-feedback-pill btn-feedback-pill-correct"
                                onClick={() => handleSubmitFeedback(t.id, true, null)}
                                disabled={feedbackSubmittingMap[t.id]}
                                title={`Confirm recommendation for ticket #${t.id}`}
                              >
                                {feedbackSubmittingMap[t.id] ? '...' : 'Confirm Correct'}
                              </button>
                              <button
                                type="button"
                                className="btn-feedback-pill btn-feedback-pill-wrong"
                                onClick={() => {
                                  setReassigningTicketId(t.id)
                                  setFeedbackErrorMap((prev) => ({ ...prev, [t.id]: '' }))
                                }}
                                disabled={feedbackSubmittingMap[t.id]}
                                title={`Mark recommendation as incorrect for ticket #${t.id}`}
                              >
                                Mark Incorrect
                              </button>
                            </div>
                          )}

                          {feedbackErrorMap[t.id] && (
                            <div className="table-feedback-error-inline" role="alert">
                              {feedbackErrorMap[t.id]}
                            </div>
                          )}
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Model Performance Section */}
      <section className="card card-performance" id="model-performance-section">
        <div className="performance-header">
          <div>
            <div className="performance-title-row">
              <h2>Model Performance</h2>
              <span className="badge badge-ai" id="performance-eval-badge">
                <span className="ai-dot"></span> Evaluation on 4,236 test tickets
              </span>
            </div>
            <p className="card-subtitle">
              The baseline always predicts the most common support team. The classifier performs better than this simple benchmark on the held-out test set.
            </p>
          </div>

          <button
            id="refresh-metrics-btn"
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={() => {
              fetchMetrics()
              fetchHealth()
            }}
            disabled={metricsLoading}
            title="Refresh metrics from API"
          >
            {metricsLoading ? (
              <>
                <span className="spinner spinner-sm" aria-hidden="true"></span>
                <span>Loading Metrics...</span>
              </>
            ) : (
              <>
                <span className="btn-icon">↻</span>
                <span>Refresh Metrics</span>
              </>
            )}
          </button>
        </div>

        {metricsError && (
          <div id="metrics-error-banner" className="banner banner-warning" role="alert">
            <span className="banner-icon">!</span>
            <div>
              <strong>Metrics status:</strong> {metricsError}
            </div>
          </div>
        )}

        {/* Metrics KPI Grid */}
        <div className="metrics-summary-grid">
          <div className="metric-box">
            <span className="metric-label">Test Accuracy</span>
            <div className="metric-number-row">
              <span className="metric-primary-value" id="metric-test-accuracy">
                {modelMetrics?.test_evaluation?.accuracy
                  ? `${(modelMetrics.test_evaluation.accuracy * 100).toFixed(1)}%`
                  : '51.2%'}
              </span>
              <span
                className="metric-lift-tag"
                title="Absolute difference: 51.2% - 28.6% = 22.7 percentage points"
              >
                +22.7 percentage points vs baseline
              </span>
            </div>
            <span className="metric-caption">
              Majority baseline: {modelMetrics?.test_evaluation?.majority_baseline_accuracy ? `${(modelMetrics.test_evaluation.majority_baseline_accuracy * 100).toFixed(1)}%` : '28.6%'} ({modelMetrics?.test_evaluation?.majority_class || 'Technical Support'})
            </span>
          </div>

          <div className="metric-box">
            <span className="metric-label">Macro F1 Score</span>
            <div className="metric-number-row">
              <span className="metric-primary-value" id="metric-macro-f1">
                {modelMetrics?.test_evaluation?.macro_f1
                  ? `${(modelMetrics.test_evaluation.macro_f1 * 100).toFixed(1)}%`
                  : '52.2%'}
              </span>
              <span
                className="metric-lift-tag"
                title="Absolute difference: 52.2% - 4.4% = 47.8 percentage points"
              >
                +47.8 percentage points vs baseline
              </span>
            </div>
            <span className="metric-caption">
              Majority baseline: {modelMetrics?.test_evaluation?.majority_baseline_macro_f1 ? `${(modelMetrics.test_evaluation.majority_baseline_macro_f1 * 100).toFixed(1)}%` : '4.4%'} &bull; Macro F1 measures performance across all support teams by giving each team equal weight.
            </span>
          </div>
        </div>
      </section>

      {/* How RouteDesk Works Section */}
      <section className="card card-architecture" id="architecture-overview-section">
        <div className="card-header">
          <div className="architecture-title-row">
            <h2>How RouteDesk Works</h2>
            <span className="badge badge-neutral">Verified System Architecture</span>
          </div>
          <p className="card-subtitle">
            Verified technical roles, components, and end-to-end request flow.
          </p>
        </div>

        <div className="architecture-grid">
          {/* Component 1: Frontend */}
          <div className="arch-card">
            <div className="arch-card-header">
              <span className="arch-icon">💻</span>
              <div>
                <h3 className="arch-title">Frontend</h3>
                <span className="arch-tech-badge">React 19 &bull; Vite 8</span>
              </div>
            </div>
            <p className="arch-desc">
              Browser interface for submitting support tickets, viewing triage recommendations, checking similarity warnings, browsing history, and recording agent feedback. Communicates exclusively with the Express backend API.
            </p>
            <div className="arch-meta">
              <span className="arch-meta-label">Endpoints called:</span>
              <code className="arch-code">/api/tickets/predict, /api/tickets, /api/tickets/:id/feedback</code>
            </div>
          </div>

          {/* Component 2: Backend API */}
          <div className="arch-card">
            <div className="arch-card-header">
              <span className="arch-icon">⚙️</span>
              <div>
                <h3 className="arch-title">Backend API</h3>
                <span className="arch-tech-badge">Node.js &bull; Express 4</span>
              </div>
            </div>
            <p className="arch-desc">
              Central API gateway that validates ticket input, fetches recent tickets from PostgreSQL for comparison, orchestrates requests to the ML service, and records tickets and human review feedback.
            </p>
            <div className="arch-meta">
              <span className="arch-meta-label">Key responsibility:</span>
              <span className="arch-meta-text">Input validation, service orchestration, database access</span>
            </div>
          </div>

          {/* Component 3: ML Service */}
          <div className="arch-card">
            <div className="arch-card-header">
              <span className="arch-icon">🤖</span>
              <div>
                <h3 className="arch-title">ML Service</h3>
                <span className="arch-tech-badge">Python 3.14 &bull; FastAPI</span>
              </div>
            </div>
            <p className="arch-desc">
              Independent microservice hosting two distinct models: a Scikit-learn TF-IDF + Logistic Regression pipeline for 10-class queue prediction, and a Sentence-Transformers (all-MiniLM-L6-v2) model for cosine similarity calculation. Has no direct database connection.
            </p>
            <div className="arch-meta">
              <span className="arch-meta-label">Models used:</span>
              <span className="arch-meta-text">TF-IDF + Logistic Reg (Classifier), all-MiniLM-L6-v2 (Embeddings)</span>
            </div>
          </div>

          {/* Component 4: Database */}
          <div className="arch-card">
            <div className="arch-card-header">
              <span className="arch-icon">🗄️</span>
              <div>
                <h3 className="arch-title">Database</h3>
                <span className="arch-tech-badge">Supabase &bull; PostgreSQL</span>
              </div>
            </div>
            <p className="arch-desc">
              Managed cloud PostgreSQL database storing persisted tickets (<code className="arch-inline-code">tickets</code> table) and agent verification feedback (<code className="arch-inline-code">ticket_feedback</code> table). Accessed only by the Express backend API using service credentials.
            </p>
            <div className="arch-meta">
              <span className="arch-meta-label">Stored tables:</span>
              <code className="arch-code">public.tickets, public.ticket_feedback</code>
            </div>
          </div>
        </div>

        {/* Request Flow Sequence */}
        <div className="architecture-flow-box">
          <h4 className="flow-title">End-to-End Request Flow</h4>
          <div className="flow-steps">
            <div className="flow-step">
              <div className="step-number">1</div>
              <div className="step-content">
                <strong>Submit Ticket</strong>
                <p>User inputs subject and description in React frontend.</p>
              </div>
            </div>
            <div className="flow-arrow" aria-hidden="true">→</div>
            <div className="flow-step">
              <div className="step-number">2</div>
              <div className="step-content">
                <strong>Validate &amp; Query</strong>
                <p>Express backend validates fields and retrieves recent tickets from Supabase.</p>
              </div>
            </div>
            <div className="flow-arrow" aria-hidden="true">→</div>
            <div className="flow-step">
              <div className="step-number">3</div>
              <div className="step-content">
                <strong>Predict &amp; Compare</strong>
                <p>FastAPI classifies the queue and computes sentence embedding similarity.</p>
              </div>
            </div>
            <div className="flow-arrow" aria-hidden="true">→</div>
            <div className="flow-step">
              <div className="step-number">4</div>
              <div className="step-content">
                <strong>Store &amp; Present</strong>
                <p>Express saves ticket to Supabase and returns routing recommendation to dashboard.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="app-footer">
        <p>RouteDesk &bull; AI-assisted support ticket routing</p>
      </footer>
    </div>
  )
}

export default App
