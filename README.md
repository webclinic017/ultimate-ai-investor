# ultimate-ai-investor
Google Colab notebook, PDF report, and back-test artefacts for the “Ultimate AI Investor” multi-model alpha-generation pipeline.

## 0 Quick Links
| Resource | Link |
|----------|------|
| **Live Colab (free GPU)** | `<PASTE-PUBLIC-COLAB-URL>` |
| **PDF White-Paper** | [`/docs/Ultimate_AI_Investor.pdf`](docs/Ultimate_AI_Investor.pdf) |
| **FastAPI Demo** | <https://ai-investor-demo.fly.dev/docs> |
| **Docker Hub** | `docker pull taehunkim/ai-investor:latest` |

---

## 1 Introduction

| Item | Detail |
|------|--------|
| **Context** | Modern quant strategies increasingly rely on **multi-model pipelines** fusing price, sentiment, and macro data. Few undergraduate projects still show the *full* workflow—data → α-generation → deployment—*and* remain reproducible on free cloud GPUs. |
| **Research Gap** | Prior work reports Sharpe *or* RMSE in isolation, omits realistic costs, and skips walk-forward tests. No open, **Yahoo-only** reference yet meets hedge-fund validation standards. |
| **Research Question** | *Can a fully open-source ensemble built on public Yahoo Finance OHLCV deliver Sharpe ≥ 2, RMSE ≤ 1.2 × 10⁻³, and orderly drawdowns after 10 bps round-trip costs?* |
| **Objectives** | 1️⃣ Build a **14-step modular pipeline** in Google Colab.<br>2️⃣ Ensemble TFT, GRU, LSTM, GNN & XGBoost with Bayesian HPO (Optuna prototype 30-50 trials; full sweep 180-500 planned).<br>3️⃣ Evaluate on **BTC-USD, AAPL, SPX** (via SPY ETF) from ≈ 2023-06 → 2025-05 with a **180-fold walk-forward baseline** and an **879-fold Optuna sweep**, both using 10 bps costs.<br>4️⃣ Package a **Dockerised FastAPI** micro-service for real-time inference. |
| **Key Contributions** | • **310-feature** library → 98 survive QA.<br>• Current OOS mean: **Sharpe 1.02**, **Sortino 1.52**, **Max DD –11.7 %**.<br>• Reproducible end-to-end in ≈ 18 min on a free Tesla T4—no paid APIs or feeds. |

---

## 2 Set-Up Instructions

### 2.1 Clone & install
```bash
git clone https://github.com/kevinkorea324/ultimate-ai-investor.git
cd ultimate-ai-investor
python -m venv venv && source venv/bin/activate      # (Windows) venv\Scripts\activate
pip install -r requirements.txt                      # or: pip install -e .
