import { Router } from "express";
import { supabase } from "../supabase.js";

const router = Router();
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://127.0.0.1:8000";

/**
 * Helper to fetch candidate tickets from Supabase for duplicate evaluation.
 * Returns up to `limit` most recent tickets.
 */
async function getCandidateTickets(limit = 100) {
  try {
    const { data: tickets, error } = await supabase
      .from("tickets")
      .select("id, subject, body, predicted_queue, confidence, created_at")
      .order("created_at", { ascending: false })
      .limit(limit);

    if (error) {
      console.warn("[WARN] Supabase error fetching candidate tickets:", error.message);
      return [];
    }

    return Array.isArray(tickets) ? tickets : [];
  } catch (err) {
    console.warn("[WARN] Unexpected error querying candidate tickets:", err.message);
    return [];
  }
}

/**
 * Helper to call the ML service /duplicates endpoint with target ticket text and candidates.
 */
async function checkSemanticDuplicates(subject, body, candidates, threshold = 0.55, topK = 3) {
  if (!candidates || candidates.length === 0) {
    return {
      has_duplicates: false,
      threshold,
      candidate_count: 0,
      duplicates: [],
    };
  }

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 8000); // 8s timeout

    const resp = await fetch(`${ML_SERVICE_URL}/duplicates`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        subject: subject || null,
        body: body || null,
        candidates,
        threshold,
        top_k: topK,
      }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (resp.ok) {
      return await resp.json();
    }

    const errData = await resp.json().catch(() => ({}));
    console.warn("[WARN] ML duplicates endpoint error:", resp.status, errData);
    return {
      has_duplicates: false,
      threshold,
      candidate_count: candidates.length,
      duplicates: [],
      error: errData.detail || "ML service duplicate detection error",
    };
  } catch (err) {
    console.warn("[WARN] Unable to reach ML service for duplicate detection:", err.message);
    return {
      has_duplicates: false,
      threshold,
      candidate_count: candidates.length,
      duplicates: [],
      warning: "Duplicate detection service unavailable",
    };
  }
}

/**
 * GET /api/tickets
 *
 * Retrieves saved tickets from Supabase, ordered newest first.
 */
router.get("/", async (_req, res) => {
  try {
    const { data: tickets, error } = await supabase
      .from("tickets")
      .select("*")
      .order("created_at", { ascending: false });

    if (error) {
      console.error("[ERROR] Failed to retrieve tickets from Supabase:", error.message);
      return res.status(500).json({
        error: "Failed to retrieve tickets from database.",
        details: error.message,
      });
    }

    return res.status(200).json(tickets || []);
  } catch (err) {
    console.error("[ERROR] Unexpected error in GET /api/tickets:", err);
    return res.status(500).json({
      error: "An unexpected error occurred while fetching tickets.",
    });
  }
});

/**
 * POST /api/tickets/duplicates
 *
 * Checks semantic duplicates for an incoming ticket against existing Supabase tickets
 * without creating or modifying any database records.
 */
router.post("/duplicates", async (req, res) => {
  try {
    const { subject, body, threshold = 0.55, top_k = 5 } = req.body || {};

    const cleanSubject = typeof subject === "string" ? subject.trim() : "";
    const cleanBody = typeof body === "string" ? body.trim() : "";

    if (!cleanSubject && !cleanBody) {
      return res.status(400).json({
        error: "Invalid input: at least one of 'subject' or 'body' must contain text.",
      });
    }

    const candidates = await getCandidateTickets(100);
    const result = await checkSemanticDuplicates(
      cleanSubject,
      cleanBody,
      candidates,
      parseFloat(threshold) || 0.55,
      parseInt(top_k, 10) || 5
    );

    return res.status(200).json(result);
  } catch (err) {
    console.error("[ERROR] Unexpected error in /api/tickets/duplicates:", err);
    return res.status(500).json({
      error: "An unexpected error occurred while checking for duplicates.",
    });
  }
});

/**
 * POST /api/tickets/predict
 *
 * 1. Forwards ticket subject and body to the FastAPI ML service for classification.
 * 2. Checks candidate tickets from Supabase for semantic duplicates (using all-MiniLM-L6-v2).
 * 3. Persists the ticket and prediction to Supabase public.tickets (no automatic rejection/merging).
 * 4. Returns queue prediction, model confidence, saved ticket, and duplicate detection results.
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

    // 2. Query existing tickets from Supabase for duplicate detection before inserting new one
    const candidateTickets = await getCandidateTickets(100);

    // 3. Call FastAPI ML service for queue classification
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

    // 4. Handle ML service error responses
    if (!mlResponse.ok) {
      const errorData = await mlResponse.json().catch(() => ({}));
      const statusCode = mlResponse.status >= 500 ? 502 : mlResponse.status;
      return res.status(statusCode).json({
        error: "ML service returned an error.",
        details: errorData.detail || errorData,
      });
    }

    const predictionData = await mlResponse.json();

    // 5. Evaluate semantic duplicates against candidate tickets
    const duplicateResults = await checkSemanticDuplicates(
      cleanSubject,
      cleanBody,
      candidateTickets,
      0.55,
      3
    );

    // 6. Persist ticket record and prediction to Supabase public.tickets
    const { data: savedTicket, error: insertError } = await supabase
      .from("tickets")
      .insert([
        {
          subject: cleanSubject,
          body: cleanBody,
          predicted_queue: predictionData.queue,
          confidence: predictionData.confidence,
        },
      ])
      .select()
      .single();

    if (insertError) {
      console.error("[ERROR] Failed to save ticket to Supabase:", insertError.message);
      return res.status(500).json({
        error: "Prediction succeeded, but failed to save ticket to database.",
        details: insertError.message,
        queue: predictionData.queue,
        confidence: predictionData.confidence,
        probabilities: predictionData.probabilities,
        duplicate_detection: duplicateResults,
      });
    }

    // 7. Return prediction result, duplicate detection, and saved ticket
    return res.status(200).json({
      queue: predictionData.queue,
      confidence: predictionData.confidence,
      probabilities: predictionData.probabilities,
      ticket: savedTicket,
      duplicate_detection: duplicateResults,
    });
  } catch (err) {
    console.error("[ERROR] Unexpected error in /api/tickets/predict:", err);
    return res.status(500).json({
      error: "An unexpected error occurred while processing the prediction request.",
    });
  }
});

export default router;
