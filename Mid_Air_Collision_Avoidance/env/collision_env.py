import gym
from gym import spaces
import numpy as np
from scipy.spatial.distance import euclidean

class CollisionAvoidanceEnv(gym.Env):
    def __init__(self, aircraft_data, max_steps=50):
        super(CollisionAvoidanceEnv, self).__init__()
        self.aircraft_data = aircraft_data.reset_index(drop=True)
        self.num_aircraft = len(self.aircraft_data)
        self.max_steps = max_steps

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0 = Hold, 1 = Climb, 2 = Turn

        self.current_step = 0
        self.done = False
        self.reset()

    def reset(self):
        self.current_step = 0
        self.done = False

        # Randomly select 2 aircraft
        self.idx1, self.idx2 = np.random.choice(self.num_aircraft, 2, replace=False)
        self.ac1 = self.aircraft_data.iloc[self.idx1]
        self.ac2 = self.aircraft_data.iloc[self.idx2]

        return self._get_observation()

    def _get_observation(self):
        obs = np.array([
            euclidean((self.ac1.longitude, self.ac1.latitude), (self.ac2.longitude, self.ac2.latitude)),
            self.ac1.altitude - self.ac2.altitude,
            self.ac1.velocity - self.ac2.velocity,
            self.ac1.heading - self.ac2.heading,
            self.ac1.vertical_rate - self.ac2.vertical_rate
        ], dtype=np.float32)
        return obs

    def step(self, action):
        self.current_step += 1

        distance = euclidean((self.ac1.longitude, self.ac1.latitude), (self.ac2.longitude, self.ac2.latitude))
        alt_diff = abs(self.ac1.altitude - self.ac2.altitude)

        reward = 0
        if action == 0:  # Hold
            reward = -10 if distance < 5 and alt_diff < 300 else 1
        elif action == 1:  # Climb
            reward = 5 if alt_diff < 300 else -1
            self.ac1.altitude += 1000
        elif action == 2:  # Turn
            reward = 2 if distance < 10 else -1
            self.ac1.heading += 15

        obs = self._get_observation()
        self.done = self.current_step >= self.max_steps

        return obs, reward, self.done, {}
