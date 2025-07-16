# ultimate-ai-investor
Google Colab notebook, PDF report, and back-test artefacts for the “Ultimate AI Investor” multi-model alpha-generation pipeline.

## 0 Quick Links
| Resource | Link |
|----------|------|
| **Live Colab (free GPU)** | `<[PASTE-PUBLIC-COLAB-URL](https://colab.research.google.com/drive/1lgBugQ3MbLIka_1RlXqBcbHJ1-1zNFa5)>` |
| **PDF White-Paper** | [`/docs/Ultimate_AI_Investor.pdf`](docs/Ultimate_AI_Investor.pdf) |
| **FastAPI Demo** | <https://ai-investor-demo.fly.dev/docs> |
| **Docker Hub** | `docker pull taehunkim/ai-investor:latest` |

## 1 Introduction

| Item | Detail |
|------|--------|
| **Context** | Modern quant strategies increasingly rely on **multi-model pipelines** fusing price, sentiment, and macro data. Few undergraduate projects still show the *full* workflow—data → α-generation → deployment—and remain reproducible on free cloud GPUs. |
| **Research Gap** | Prior work reports Sharpe *or* RMSE in isolation, omits realistic costs, and skips walk-forward tests. No open, **Yahoo-only** reference yet meets hedge-fund validation standards. |
| **Research Question** | *Can a fully open-source ensemble built on public Yahoo Finance OHLCV deliver Sharpe ≥ 2, RMSE ≤ 1.2 × 10⁻³, and orderly drawdowns after 10 bps round-trip costs?* |
| **Objectives** | (1) Build a **14-step modular pipeline** in Google Colab.<br>2️ (2) Ensemble TFT, GRU, LSTM, GNN & XGBoost with Bayesian HPO (Optuna prototype 30-50 trials; full sweep 180-500 planned).<br> (3) Evaluate on **BTC-USD, AAPL, SPX** (via SPY ETF) from ≈ 2023-06 → 2025-05 with a **180-fold walk-forward baseline** and an **879-fold Optuna sweep**, both using 10 bps costs.<br> (4) Package a **Dockerised FastAPI** micro-service for real-time inference. |
| **Key Contributions** | • **310-feature** library → 98 survive QA. <br>• Current OOS mean: **Sharpe 1.02**, **Sortino 1.52**, **Max DD –11.7 %**. <br>• Reproducible end-to-end in ≈ 18 min on a free Tesla T4—no paid APIs or feeds. |

## 2 Set-Up Instructions

### 2.1 Clone & install

git clone https://github.com/kevinkorea324/ultimate-ai-investor.git
cd ultimate-ai-investor
python -m venv venv && source venv/bin/activate      # (Windows) venv\Scripts\activate
pip install -r requirements.txt                      # or: pip install -e .

### 2.2 Run the Colab notebook
1. Open **`Ultimate_AI_Investor.ipynb`** in Colab.  
2. Choose **Runtime → Run all** (≈ 18 min on a free GPU).

### 2.3 Docker one-liner

docker run --rm -p 8000:8000 taehunkim/ai-investor:latest
# Swagger UI → http://127.0.0.1:8000/docs

### 2.4 Command-line back-test
python cli/backtest.py --symbol BTC-USD --horizon 60

## 3 Pipeline Overview (14 Steps)

```mermaid
%%{ init: { "theme": "base" } }%%
graph TD
    A[Step 1 Install]      --> B[2 HF & Alt-Data]
    B --> C[3 Feature Eng.]
    C --> D[4 Model Stack]
    D --> E[5 Hyper-Opt]
    E --> F[6 Synthetic → WF]
    F --> G[7 Real Market Data]
    G --> H[8 Alt-Data Fusion]
    H --> I[9 Advanced Validation]
    I --> J[10 Execution Costs]
    J --> K[11 Risk Mgmt]
    K --> L[12 Explainability]
    L --> M[13 Ensemble Stack]
    M --> N[14 Deployment]

## 4 Results (OOS ≈ 2023-06 → 2025-05)

| Metric      | BTC-USD | AAPL | SPX\* | **Mean** |
|-------------|-------:|----:|------:|---------:|
| **Sharpe**  | 1.05   | 0.98 | *pending* | **1.02** |
| **Sortino** | 1.63   | 1.40 | *pending* | **1.52** |
| **Max DD**  | –11.4 % | –11.9 % | *pending* | **–11.7 %** |

\*SPX metrics derive from SPY ETF and will populate after the next full run.  

> Meets the drawdown target (< –12 %) but still trails Sharpe > 2 and RMSE ≤ 1.2 × 10⁻³; a 500-trial Optuna sweep is scheduled.

## 5 Repository Structure
```text
ultimate-ai-investor/
├─ pipeline/               # 14 notebooks / scripts
│  ├─ 01_install.ipynb
│  ├─ …
│  └─ 14_deploy.ipynb
├─ cli/                    # Command-line helpers
│  ├─ backtest.py
│  └─ predict.py
├─ fastapi_app/            # Production REST service
│  ├─ main.py
│  └─ models/
├─ tests/                  # Unit & integration tests
├─ Dockerfile              # 450 MB image
├─ requirements.txt
└─ README.md

## 6 Application Code Highlights

| File                       | Purpose                                                                      |
|----------------------------|------------------------------------------------------------------------------|
| `fastapi_app/main.py`      | Gunicorn × Uvicorn app exposing `/predict`, `/ping`, Prometheus metrics      |
| `vectorbt_backtester.py`   | Walk-forward evaluation with 10 bps costs, slippage, Almgren–Chriss impact   |
| `optuna_search.py`         | Bayesian HPO with ASHA early-stopping (`search_spaces.yaml`)                 |
| `stacking.py`              | Blends TFT, GRU, LSTM, GAT, XGB predictions via meta-XGB                     |

## 7 Testing

pytest -q      # 27 tests, all < 10 s

## 8 Deployment Matrix

| Stage           | Command                                   | Notes                          |
|-----------------|-------------------------------------------|--------------------------------|
| **Local FastAPI** | `uvicorn fastapi_app.main:app --reload` | Hot-reload for UI work         |
| **Docker**        | `docker build -t ai-investor .`         | 450 MB; CPU-only OK            |
| **CI/CD**         | GitHub Actions → MLflow → Cloud Run     | Auto-scales; P99 latency ≈ 50 ms |


## 10 Citation
If you use this work, please cite:
```bibtex
@article{kim2025ultimateai,
  title  = {Ultimate AI Investor — Multi-Model Alpha Generation Using Market, Sentiment, and Macroeconomic Signals},
  author = {Taehun Kim},
  year   = {2025},
  note   = {arXiv:XXXX.XXXXX}   % ← update once pre-print is live
}

