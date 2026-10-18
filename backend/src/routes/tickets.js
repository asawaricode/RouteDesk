import { Router } from "express";

const router = Router();
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://127.0.0.1:8000";

/**
 * POST /api/tickets/predict
 *
 * Forwards ticket subject and body to the FastAPI ML service for classification.
 * Validates that at least one field has non-whitespace text.
 */
router.post("/predict", async (req, res) => {
  try {
    const { subject, body } = req.body || {};

    const cleanSubject = typeof subject === "string" ? subject.trim() : "";
    const cleanBody = typeof body === "string" ? body.trim() : "";

    // 1. Validation: at least one field must have non-whitespace text
    if (!cleanSubject && !cleanBody) {
      return res.status(400).json({
        error: "Invalid input: at least one of 'subject' or 'body' must contain non-whitespace text.",
      });
    }

    // 2. Call FastAPI ML service
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000); // 10-second timeout

    let mlResponse;
    try {
      mlResponse = await fetch(`${ML_SERVICE_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          subject: cleanSubject || null,
          body: cleanBody || null,
        }),
        signal: controller.signal,
      });
    } catch (networkErr) {
      clearTimeout(timeoutId);
      console.error(`[WARN] ML service unreachable at ${ML_SERVICE_URL}:`, networkErr.message);
      return res.status(503).json({
        error: "ML service is unavailable. Please ensure the ML service is running.",
        details: networkErr.message,
      });
    } finally {
      clearTimeout(timeoutId);
    }

    // 3. Handle ML service error responses
    if (!mlResponse.ok) {
      const errorData = await mlResponse.json().catch(() => ({}));
      const statusCode = mlResponse.status >= 500 ? 502 : mlResponse.status;
      return res.status(statusCode).json({
        error: "ML service returned an error.",
        details: errorData.detail || errorData,
      });
    }

    // 4. Return prediction result
    const data = await mlResponse.json();
    return res.status(200).json({
      queue: data.queue,
      confidence: data.confidence,
      probabilities: data.probabilities,
    });
  } catch (err) {
    console.error("[ERROR] Unexpected error in /api/tickets/predict:", err);
    return res.status(500).json({
      error: "An unexpected error occurred while processing the prediction request.",
    });
  }
});

export default router;
