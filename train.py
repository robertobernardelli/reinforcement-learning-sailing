from stable_baselines3 import PPO
import os
from sailing import SailingEnv
import time


def calculate_average_reward(model, env, num_episodes=1):
    total_reward = 0
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, _info = env.step(action)
            total_reward += reward
    return total_reward / num_episodes


models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = SailingEnv()
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
iters = 0
best_average_reward = -float("inf")
while True:
    iters += 1
    model.learn(
        total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO"
    )

    # Calculate average reward over some number of episodes
    avg_reward = calculate_average_reward(model, env, num_episodes=1)
    model.save(f"{models_dir}/{TIMESTEPS*iters}")

    if avg_reward > best_average_reward:
        best_average_reward = avg_reward
        model.save(f"{models_dir}/best")
        print(f"SAVED BEST MODEL {avg_reward}")
