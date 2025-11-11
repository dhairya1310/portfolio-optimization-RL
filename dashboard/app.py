# ============================================================
# Streamlit Dashboard — Deep RL Portfolio Optimization

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, DQN

st.set_page_config(page_title="RL Portfolio Dashboard", layout="wide")

st.title("📈 Deep Reinforcement Learning for Dynamic Portfolio Optimization")
st.markdown("""
Visualize model performance of **DQN**, **A2C**, and **PPO** on 34-year stock data (1990–2024).
""")

# Sidebar options
view_mode = st.sidebar.radio("Choose Mode", ["Single Model", "Compare All Models"])

@st.cache_resource
def load_models():
    return {
        "DQN": DQN.load("models/DQN_portfolio.zip"),
        "A2C": A2C.load("models/A2C_portfolio.zip"),
        "PPO": PPO.load("models/PPO_portfolio.zip"),
    }

models = load_models()

# Dummy simulation (replace with real one later)
x = np.arange(100)
data = {
    "DQN": np.cumprod(1 + 0.001 * np.random.randn(100)),
    "A2C": np.cumprod(1 + 0.0012 * np.random.randn(100)),
    "PPO": np.cumprod(1 + 0.0009 * np.random.randn(100)),
}

df = pd.DataFrame(data, index=x)

if view_mode == "Single Model":
    model_choice = st.sidebar.selectbox("Select Model", ["DQN", "A2C", "PPO"])
    st.line_chart(df[[model_choice]])
else:
    st.line_chart(df)

st.markdown("🧠 *Developed by Dhairya Shah & Nidhi Gurukumar (CS583)*")
