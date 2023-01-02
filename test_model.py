import gym
from stable_baselines3 import PPO
from sailing import SailingEnv

def main():

    env = SailingEnv()
    env.reset()

    model_path = "best_model.zip"
    model = PPO.load(model_path, env=env)

    episodes = 5

    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()
            print(rewards)
    
    env.close()
        
if __name__ == '__main__':
    main()