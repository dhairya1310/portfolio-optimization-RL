# ============================================================
# Streamlit Dashboard — Deep RL Portfolio Optimization
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, DQN
import gymnasium as gym
import os

st.set_page_config(page_title="RL Portfolio Dashboard", layout="wide")

st.title("📊 Deep Reinforcement Learning for Dynamic Portfolio Optimization")
st.markdown("""
Visualize the real performance of **DQN**, **A2C**, and **PPO** models on 34-year stock data (1990–2024).  
Toggle between single-model and multi-model views to explore agent behavior.
""")

# --------------------------
# Sidebar Controls
# --------------------------
mode = st.sidebar.radio("Choose Mode", ["Single Model", "Compare All Models"])

@st.cache_resource
def load_models():
    return {
        "DQN": DQN.load("models/DQN_portfolio.zip"),
        "A2C": A2C.load("models/A2C_portfolio.zip"),
        "PPO": PPO.load("models/PPO_portfolio.zip"),
    }

models = load_models()

# Load test data
@st.cache_data
def load_test_data():
    df = pd.read_csv("data_prepared/test.csv")
    return df

df = load_test_data()

# Define a minimal environment structure
class PortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, data, tickers):
        super().__init__()
        self.data = data
        self.tickers = tickers
        self.current_step = 0
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(len(tickers),), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(tickers),), dtype=np.float32)
        self.initial_value = 100000
        self.portfolio_value = self.initial_value

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.portfolio_value = self.initial_value
        obs = self.data[self.tickers].iloc[self.current_step].values
        return obs, {}

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.data):
            terminated = True
        else:
            terminated = False
        
        obs = self.data[self.tickers].iloc[self.current_step].values
        reward = np.dot(action, obs)  # simplified reward (expected return)
        self.portfolio_value *= (1 + reward / 100)
        info = {"portfolio_value": self.portfolio_value}
        return obs, reward, terminated, False, info

tickers = ["sp500", "djia", "hsi"]
env = PortfolioEnv(df, tickers)

# --------------------------
# Simulation Function
# --------------------------
def simulate_model(model):
    obs, _ = env.reset()
    done = False
    portfolio_values = [env.portfolio_value]
    
    while not done:
        # ✅ FIX: reshape observation for compatibility
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
    
    return pd.Series(portfolio_values)


# --------------------------
# Single Model Mode
# --------------------------
if mode == "Single Model":
    model_choice = st.sidebar.selectbox("Select Model", ["DQN", "A2C", "PPO"])
    st.subheader(f"📈 {model_choice} Portfolio Simulation")
    with st.spinner(f"Running {model_choice}..."):
        values = simulate_model(models[model_choice])
    
    st.line_chart(values)
    st.caption(f"Final Portfolio Value: ${values.iloc[-1]:,.2f}")

# --------------------------
# Compare All Models Mode
# --------------------------
else:
    st.subheader("🔍 Model Comparison — Portfolio Values")
    results = {}
    with st.spinner("Simulating all models..."):
        for name, model in models.items():
            results[name] = simulate_model(model)
    
    df_results = pd.DataFrame(results)
    st.line_chart(df_results)
    final_vals = {name: vals.iloc[-1] for name, vals in results.items()}
    st.table(pd.DataFrame(final_vals, index=["Final Portfolio Value ($)"]))

st.markdown("🧠 *Developed by Dhairya Shah & Nidhi Gurukumar (CS583)*")
