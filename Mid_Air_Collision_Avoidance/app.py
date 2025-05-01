from flask import Flask, render_template, request, jsonify
from utils.data_utils import process_flight_data
from env.collision_env import CollisionAvoidanceEnv
from utils.llm_explainer import explain_action_with_llm
from stable_baselines3 import PPO, DQN
import numpy as np
import os

app = Flask(__name__)

# Load and prepare environment
data = process_flight_data()
env = CollisionAvoidanceEnv(data)

# Load models
ppo_model = PPO.load("models/ppo_model.zip")
dqn_model = DQN.load("models/dqn_model.zip")

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    model_type = request.form.get("model")

    obs = env.reset()
    if model_type == "ppo":
        action, _ = ppo_model.predict(obs)
    else:
        action, _ = dqn_model.predict(obs)

    _, reward, _, _ = env.step(action)
    explanation = explain_action_with_llm(int(action), obs)

    result = {
        "observation": obs.tolist(),
        "action": int(action),
        "reward": float(reward),
        "done": bool(False),
        "explanation": explanation
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
