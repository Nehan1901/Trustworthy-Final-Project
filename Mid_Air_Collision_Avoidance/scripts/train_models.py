import os
import torch
import pandas as pd
from stable_baselines3 import PPO, DQN, A2C, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from env.collision_env import CollisionAvoidanceEnv
from utils.data_utils import load_mock_data

#  Load cleaned mock flight data
df = load_mock_data()

#  Drop non-numeric column (e.g., flight_id)
if "flight_id" in df.columns:
    df = df.drop(columns=["flight_id"])

#  Create Gym environment
env = DummyVecEnv([lambda: CollisionAvoidanceEnv(df.copy(), max_steps=100)])

#  Define training device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

#  Create directory for saving models
os.makedirs("models", exist_ok=True)

# ---------------- PPO MODEL ----------------
ppo_model_path = "models/ppo_model.zip"
ppo_model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    batch_size=64,
    n_steps=1024,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    device=device
)
print("\n🚀 Training the PPO Model...")
ppo_model.learn(total_timesteps=20000)
ppo_model.save(ppo_model_path.replace(".zip", ""))
print(f"\n✅ PPO Model Training Completed. Model saved at {ppo_model_path}")

# ---------------- DQN MODEL ----------------
dqn_model_path = "models/dqn_model.zip"
dqn_model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0005,
    batch_size=64,
    buffer_size=50000,
    learning_starts=1000,
    gamma=0.99,
    train_freq=4,
    device=device
)
print("\n🚀 Training the DQN Model...")
dqn_model.learn(total_timesteps=20000)
dqn_model.save(dqn_model_path.replace(".zip", ""))
print(f"\n✅ DQN Model Training Completed. Model saved at {dqn_model_path}")

# ---------------- A2C MODEL ----------------
a2c_model_path = "models/a2c_model.zip"
a2c_model = A2C(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    gamma=0.99,
    device=device
)
print("\n🚀 Training the A2C Model...")
a2c_model.learn(total_timesteps=20000)
a2c_model.save(a2c_model_path.replace(".zip", ""))
print(f"\n✅ A2C Model Training Completed. Model saved at {a2c_model_path}")

