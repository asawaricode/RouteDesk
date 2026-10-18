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

/**
 * GET /health
 * Health-check endpoint — returns service status and timestamp.
 */
app.get("/health", (_req, res) => {
  res.json({
    status: "ok",
    service: "routedesk-backend",
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
