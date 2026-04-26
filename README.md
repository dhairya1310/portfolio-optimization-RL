# QuantRL: Deep Reinforcement Learning for Portfolio Optimization

A deep reinforcement learning system that learns dynamic asset allocation policies across major global equity indices (S&P 500, DJIA, Hang Seng Index). Three RL algorithms — DQN, A2C, and PPO — were trained on 30+ years of daily market data and evaluated against an equal-weight baseline on out-of-sample 2023–2024 data. Results are presented in an interactive Streamlit dashboard.

## Results

Evaluated on the 2023–2024 out-of-sample test period, starting from $100,000:

| Model | Final Value | Sharpe Ratio | Max Drawdown | Accuracy | Precision | Recall | F1 |
|-------|------------:|-------------:|-------------:|---------:|----------:|-------:|---:|
| **A2C** | **$129,914.59** | **1.88** | -10.28% | 0.5409 | 0.5484 | 0.5903 | 0.5686 |
| PPO   | $121,704.56 | 1.58 | -9.43%  | 0.5338 | 0.5409 | 0.5972 | 0.5677 |
| DQN   | $116,105.37 | 1.26 | -9.02%  | 0.5302 | 0.5380 | 0.5903 | 0.5629 |

**Key takeaway:** Continuous-action models (A2C, PPO) outperformed the discrete-action DQN across every financial metric. A2C delivered the strongest risk-adjusted return with a Sharpe of 1.88 and a 29.9% gain over the test period.

## Motivation

Traditional portfolio strategies — Modern Portfolio Theory, fixed-rule allocation, equal-weight — rely on static assumptions about returns and covariances that break down during regime shifts and volatility spikes. This project frames asset allocation as a sequential decision-making problem and trains an RL agent that adapts its allocation policy to changing market conditions, optimizing for risk-adjusted returns rather than raw return.

## Methodology

### Data

- **Source:** 34 Year Daily Stock Data (Kaggle) — daily prices, volumes, and macro indicators for major global indices, 1990–2024
- **Asset universe:** S&P 500, Dow Jones Industrial Average, Hang Seng Index
- **Splits (chronological):** train 1990–2020 · validation 2021–2022 · test 2023–2024
- **Features:** daily returns, normalized OHLCV, plus macro context — VIX, ADS index, US 3-month T-bill rate, joblessness rate, Economic Policy Uncertainty (EPU) index, trading volumes

### Custom Gym Environments

Two custom environments built on `gymnasium`:

- **`PortfolioEnv`** — continuous action space (Box) for A2C and PPO; outputs a 3-dimensional weight vector clipped to [0, 1] and normalized to sum to 1
- **`DQNPortfolioEnv`** — discrete action space with 27 fixed allocation patterns built from weights {0, 0.5, 1} across the three indices

**Reward function** (both environments):

reward = log(V_t / V_{t-1}) − transaction_cost · turnover − 0.1 · return²
Combines log-portfolio-return, a turnover penalty (transaction cost = 0.1% per unit of weight change), and a small quadratic risk penalty to discourage volatile allocations.

### Models & Hyperparameters

| Algorithm | Action Space | γ | Learning Rate | Other |
|-----------|-------------|----|---------------|-------|
| **DQN** | Discrete (27 patterns) | 0.99 | 1e-4 | batch=128, buffer=80k, train_freq=4 |
| **A2C** | Continuous (3-dim Box) | 0.99 | 5e-4 | n_steps=10 |
| **PPO** | Continuous (3-dim Box) | 0.995 | 1e-4 | n_steps=2048, batch=128 |

### Evaluation

- **Financial metrics:** final portfolio value, Sharpe ratio, maximum drawdown
- **Classification metrics** (vs equal-weight baseline on daily directional decisions): accuracy, precision, recall, F1
- All evaluation done on held-out 2023–2024 data — no peeking

## Project Structure

```
portfolio-optimization-RL/
├── data_prepared/      # train.csv, val.csv, test.csv — generated from stock_data.csv
├── models/             # Trained model checkpoints (DQN, A2C, PPO)
├── portfolio_env.py    # Custom Gymnasium environments
├── train_rl.py         # Trains DQN, A2C, PPO (uses GPU if available)
├── app.py              # Streamlit dashboard for EDA + RL evaluation
└── requirements.txt    # Python dependencies
```
## Tech Stack

Python · PyTorch · Stable-Baselines3 · Gymnasium · pandas · NumPy · Streamlit · Plotly

## Getting Started

```bash
git clone https://github.com/dhairya1310/portfolio-optimization-RL.git
cd portfolio-optimization-RL

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**For GPU training**, install CUDA-enabled PyTorch (example for CUDA 11.8):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU is detected:
```python
import torch
print(torch.cuda.is_available())
```

### Train the models

```bash
python train_rl.py
```

This produces `models/DQN_portfolio.zip`, `models/A2C_portfolio.zip`, and `models/PPO_portfolio.zip`.

### Launch the dashboard

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`). The dashboard includes:
- **EDA** — data preview, distributions, time-series plots, correlation heatmap
- **Single Model mode** — portfolio simulation, Sharpe/drawdown metrics, hyperparameter view for any of the three agents
- **Compare All mode** — side-by-side test-set portfolio curves and metrics across all three models

## Key Findings

1. **Continuous action spaces win.** A2C and PPO both beat DQN on every financial metric. The discrete 27-action grid limited DQN's ability to make fine-grained reallocation decisions.
2. **A2C achieved the best Sharpe (1.88).** It produced smooth reallocations that adapted well to regime shifts in the test period — particularly the Hang Seng's downtrend through 2023.
3. **PPO was a close second** with the highest recall (0.597), suggesting it caught more upward-trending days than the other agents.
4. **All agents kept drawdown under control** (~9–10%); risk-adjusted return is the more meaningful comparison than raw return.

## Limitations & Future Work

- **Long-only, no leverage** — extending to long/short would require modeling borrow costs and margin
- **Transaction costs simplified** — execution slippage and market impact not modeled
- **Three-asset universe** — scaling to broader cross-asset portfolios (bonds, commodities, FX) is a natural next step
- **Single rebalancing frequency** — multi-timescale policies (intraday + daily) are an open extension
- **Reward shaping** — alternative rewards (Sortino, CVaR-penalized returns) could yield more risk-aware policies
