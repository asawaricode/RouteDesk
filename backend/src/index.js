import "dotenv/config";
import express from "express";
import cors from "cors";

import ticketsRouter from "./routes/tickets.js";

const app = express();
const PORT = process.env.PORT || 4000;

// ── Middleware ──────────────────────────────────────────────────────────────
app.use(cors());
app.use(express.json());

// ── Routes ──────────────────────────────────────────────────────────────────

import { supabase } from "./supabase.js";

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://127.0.0.1:8000";

/**
 * GET /health
 *
 * Verifies live connectivity across all architectural tiers:
 * - Express backend
 * - FastAPI ML Service (classifier + sentence-transformer embedding model)
 * - Supabase database (tickets table accessibility)
 */
app.get("/health", async (_req, res) => {
  let mlServiceOk = false;
  let classifierLoaded = false;
  let embeddingLoaded = false;
  let supabaseOk = false;

  try {
    const mlResp = await fetch(`${ML_SERVICE_URL}/health`, { signal: AbortSignal.timeout(2500) });
    if (mlResp.ok) {
      const mlData = await mlResp.json();
      mlServiceOk = true;
      classifierLoaded = !!mlData.model_loaded;
      embeddingLoaded = !!mlData.embedding_model_loaded;
    }
  } catch (err) {
    // ML service offline or unreachable
  }

  try {
    const { error } = await supabase.from("tickets").select("id").limit(1);
    supabaseOk = !error;
  } catch (err) {
    // Supabase offline or credentials invalid
  }

  const allHealthy = mlServiceOk && classifierLoaded && embeddingLoaded && supabaseOk;

  res.status(200).json({
    status: allHealthy ? "ok" : "degraded",
    service: "routedesk-backend",
    services: {
      backend: { status: "online" },
      ml_classifier: {
        status: classifierLoaded ? "online" : "offline",
        model: "TF-IDF + LogisticRegression",
      },
      ml_embeddings: {
        status: embeddingLoaded ? "online" : "offline",
        model: "all-MiniLM-L6-v2",
      },
      supabase: {
        status: supabaseOk ? "online" : "offline",
        table: "public.tickets",
      },
    },
    timestamp: new Date().toISOString(),
  });
});

/**
 * Ticket routing endpoints
 */
app.use("/api/tickets", ticketsRouter);


/**
 * Catch-all 404
 */
app.use((_req, res) => {
  res.status(404).json({ error: "Route not found" });
});

// ── Start ────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`[OK] Backend running at http://localhost:${PORT}`);
});
