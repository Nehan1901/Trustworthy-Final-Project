# import os
# import gym
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from stable_baselines3 import PPO, DQN
# from env.collision_env import CollisionAvoidanceEnv
# from utils.data_utils import process_flight_data

# # Set seed for reproducibility
# SEED = 42
# np.random.seed(SEED)
# torch.manual_seed(SEED)

# # Check for GPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"🚀 Using device: {device}")

# # Load data
# data = process_flight_data()

# # Create environment
# env = CollisionAvoidanceEnv(data)

# def train_and_save(model_class, model_name, timesteps=10000):
#     model = model_class("MlpPolicy", env, verbose=1, device=device)
#     model.learn(total_timesteps=timesteps)
#     model_path = f"models/{model_name}.zip"
#     model.save(model_path)
#     print(f"✅ Model saved to {model_path}")
#     return model

# # Train PPO
# ppo_model = train_and_save(PPO, "ppo_model")

# # Train DQN
# dqn_model = train_and_save(DQN, "dqn_model")

# # Evaluate models
# def evaluate_model(model, name):
#     obs = env.reset()
#     rewards = []
#     for _ in range(100):
#         action, _states = model.predict(obs)
#         obs, reward, done, _ = env.step(action)
#         rewards.append(reward)
#         if done:
#             break
#     print(f"{name} Mean Reward: {np.mean(rewards):.2f}")
#     return rewards

# ppo_rewards = evaluate_model(ppo_model, "PPO")
# dqn_rewards = evaluate_model(dqn_model, "DQN")

# # Plot comparison
# plt.plot(ppo_rewards, label="PPO")
# plt.plot(dqn_rewards, label="DQN")
# plt.xlabel("Timestep")
# plt.ylabel("Reward")
# plt.title("Model Comparison")
# plt.legend()
# plt.grid(True)
# plt.show()
