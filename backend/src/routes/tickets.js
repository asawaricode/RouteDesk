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

export const VALID_QUEUES = [
  "Billing and Payments",
  "Customer Service",
  "General Inquiry",
  "Human Resources",
  "IT Support",
  "Product Support",
  "Returns and Exchanges",
  "Sales and Pre-Sales",
  "Service Outages and Maintenance",
  "Technical Support",
];

/**
 * GET /api/tickets
 *
 * Retrieves saved tickets from Supabase with their associated feedback records, ordered newest first.
 */
router.get("/", async (_req, res) => {
  try {
    const { data: tickets, error } = await supabase
      .from("tickets")
      .select("*, ticket_feedback(*)")
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
 * Helper to process ticket feedback submission.
 */
async function processFeedback(ticketIdParam, reqBody, res) {
  const ticketId = parseInt(ticketIdParam, 10);

  if (isNaN(ticketId) || ticketId <= 0) {
    return res.status(400).json({
      error: "Invalid ticket ID. Must be a positive integer.",
    });
  }

  const { is_correct, corrected_queue } = reqBody || {};

  if (typeof is_correct !== "boolean") {
    return res.status(400).json({
      error: "Invalid input: 'is_correct' must be a boolean (true or false).",
    });
  }

  // 1. Verify that parent ticket exists in public.tickets
  const { data: ticket, error: ticketError } = await supabase
    .from("tickets")
    .select("id, predicted_queue")
    .eq("id", ticketId)
    .single();

  if (ticketError || !ticket) {
    return res.status(404).json({
      error: `Ticket #${ticketId} not found in database.`,
      details: ticketError?.message,
    });
  }

  // 2. Prevent accidental duplicate feedback for the same ticket
  const { data: existingFeedback } = await supabase
    .from("ticket_feedback")
    .select("id, ticket_id, is_correct, corrected_queue, created_at")
    .eq("ticket_id", ticketId)
    .maybeSingle();

  if (existingFeedback) {
    return res.status(409).json({
      error: `Feedback has already been submitted for ticket #${ticketId}.`,
      feedback: existingFeedback,
    });
  }

  // 3. Validate corrected_queue consistency
  let finalCorrectedQueue = null;
  if (is_correct === true) {
    finalCorrectedQueue = null;
  } else {
    if (!corrected_queue || typeof corrected_queue !== "string" || !corrected_queue.trim()) {
      return res.status(400).json({
        error: "When 'is_correct' is false, 'corrected_queue' must be provided.",
        valid_queues: VALID_QUEUES,
      });
    }

    const trimmedQueue = corrected_queue.trim();
    if (!VALID_QUEUES.includes(trimmedQueue)) {
      return res.status(400).json({
        error: `Invalid corrected queue: '${trimmedQueue}'.`,
        valid_queues: VALID_QUEUES,
      });
    }
    finalCorrectedQueue = trimmedQueue;
  }

  // 4. Insert feedback into Supabase
  const { data: newFeedback, error: insertError } = await supabase
    .from("ticket_feedback")
    .insert([
      {
        ticket_id: ticketId,
        predicted_queue: ticket.predicted_queue,
        corrected_queue: finalCorrectedQueue,
        is_correct,
      },
    ])
    .select()
    .single();

  if (insertError) {
    if (insertError.code === "23505") {
      return res.status(409).json({
        error: `Feedback has already been submitted for ticket #${ticketId}.`,
        details: insertError.message,
      });
    }
    if (insertError.code === "23514") {
      return res.status(400).json({
        error: "Database constraint violation: corrected_queue must be null if correct, or non-empty if incorrect.",
        details: insertError.message,
      });
    }
    console.error("[ERROR] Failed to save feedback to Supabase:", insertError.message);
    return res.status(500).json({
      error: "Failed to record feedback in database.",
      details: insertError.message,
    });
  }

  return res.status(201).json({
    message: "Feedback submitted successfully.",
    feedback: newFeedback,
  });
}

/**
 * POST /api/tickets/:id/feedback
 *
 * Records support agent feedback for a ticket's predicted queue.
 */
router.post("/:id/feedback", async (req, res) => {
  try {
    return await processFeedback(req.params.id, req.body, res);
  } catch (err) {
    console.error("[ERROR] Unexpected error in POST /api/tickets/:id/feedback:", err);
    return res.status(500).json({
      error: "An unexpected error occurred while saving feedback.",
    });
  }
});

/**
 * POST /api/tickets/feedback
 *
 * Alias endpoint for recording feedback where ticket_id is supplied in the request body.
 */
router.post("/feedback", async (req, res) => {
  try {
    const ticketId = req.body?.ticket_id;
    return await processFeedback(ticketId, req.body, res);
  } catch (err) {
    console.error("[ERROR] Unexpected error in POST /api/tickets/feedback:", err);
    return res.status(500).json({
      error: "An unexpected error occurred while saving feedback.",
    });
  }
});

/**
 * GET /api/tickets/metrics
 *
 * Retrieves model evaluation metrics (accuracy, macro F1, test split evaluation)
 * from the ML service metrics endpoint.
 */
router.get("/metrics", async (_req, res) => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    const mlResponse = await fetch(`${ML_SERVICE_URL}/metrics`, {
      method: "GET",
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!mlResponse.ok) {
      const errBody = await mlResponse.json().catch(() => ({}));
      return res.status(mlResponse.status).json({
        error: "Failed to retrieve metrics from ML service.",
        details: errBody.detail || errBody,
      });
    }

    const data = await mlResponse.json();
    return res.status(200).json(data);
  } catch (err) {
    console.error("[ERROR] Failed to fetch model metrics from ML service:", err.message);
    return res.status(503).json({
      error: "ML service metrics endpoint unavailable.",
      details: err.message,
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
    const { subject, body, threshold = 0.55, top_k = 5, exclude_id } = req.body || {};

    const cleanSubject = typeof subject === "string" ? subject.trim() : "";
    const cleanBody = typeof body === "string" ? body.trim() : "";

    if (!cleanSubject && !cleanBody) {
      return res.status(400).json({
        error: "Invalid input: at least one of 'subject' or 'body' must contain text.",
      });
    }

    let candidates = await getCandidateTickets(100);
    if (exclude_id) {
      const excludeNum = parseInt(exclude_id, 10);
      if (!isNaN(excludeNum)) {
        candidates = candidates.filter((c) => c.id !== excludeNum);
      }
    }

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
