# TriageIQ — Support Ticket Routing System

An AI-powered support ticket routing system built with a modern three-tier
architecture.

## Project Structure

```
TriageIQ/
├── frontend/          # React + Vite (port 5173)
│   ├── src/
│   ├── public/
│   ├── .env.example
│   └── package.json
│
├── backend/           # Node.js + Express API (port 4000)
│   ├── src/
│   │   └── index.js
│   ├── .env.example
│   └── package.json
│
└── ml-service/        # Python + FastAPI ML engine (port 8000)
    ├── app/
    │   ├── __init__.py
    │   └── main.py
    ├── .venv/         # not committed
    ├── .env.example
    └── requirements.txt
```

---

## Prerequisites

| Tool | Minimum version |
|------|----------------|
| Node.js | 20 |
| npm | 9 |
| Python | 3.11 |

---

## Quick Start

### 1. Clone & enter the repo

```bash
git clone <repo-url>
cd TriageIQ
```

### 2. Copy environment files

```bash
# Frontend
cp frontend/.env.example frontend/.env

# Backend
cp backend/.env.example backend/.env

# ML service
cp ml-service/.env.example ml-service/.env
```

---

### 3. Frontend — React + Vite

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

### 4. Backend — Node.js + Express

```bash
cd backend
npm install
npm run dev      # uses --watch (Node 22+) for hot-reload
# → http://localhost:4000
```

Health check:

```bash
curl http://localhost:4000/health
# {"status":"ok","service":"triageiq-backend","timestamp":"..."}
```

---

### 5. ML Service — Python + FastAPI

```bash
cd ml-service

# Create and activate virtual environment (first time only)
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# → http://localhost:8000
```

Health check:

```bash
curl http://localhost:8000/health
# {"status":"ok","service":"triageiq-ml-service","timestamp":"..."}
```

Interactive API docs: <http://localhost:8000/docs>

---

## Running All Services Simultaneously

Open **three terminal tabs** and run the commands above in each.

Or use a process manager such as [concurrently](https://github.com/open-cli-tools/concurrently):

```bash
# From the repo root (install once)
npm install -g concurrently

concurrently \
  "cd frontend && npm run dev" \
  "cd backend && npm run dev" \
  "cd ml-service && .venv/Scripts/activate && uvicorn app.main:app --reload --port 8000"
```

---

## Health Endpoints Summary

| Service | URL | Method |
|---------|-----|--------|
| Backend | `http://localhost:4000/health` | GET |
| ML Service | `http://localhost:8000/health` | GET |
| ML Service Docs | `http://localhost:8000/docs` | GET |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite 6 |
| Backend | Node.js 20+, Express 4, ES Modules |
| ML Service | Python 3.11+, FastAPI, Uvicorn |

---

## Development Notes

- The backend uses **ES Modules** (`"type": "module"` in `package.json`).
- The ML service uses the FastAPI **lifespan** context manager for startup/shutdown logic — load ML models there in a future phase.
- Never commit `.env` files. Use `.env.example` as a template.
