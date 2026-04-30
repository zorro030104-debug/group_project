import numpy as np
from student_env import MathTutorEnv

NUM_ACTIONS = 2

def run_random_baseline(episodes=1000, max_steps=30):
    rewards = []

    for _ in range(episodes):
        env = MathTutorEnv(max_steps=max_steps)
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = np.random.choice(NUM_ACTIONS)
            next_state, reward, done, info = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    return rewards

def run_weakest_skill_baseline(episodes=1000, max_steps=30):
    rewards = []

    for _ in range(episodes):
        env = MathTutorEnv(max_steps=max_steps)
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            if env.add_skill <= env.sub_skill:
                action = 0
            else:
                action = 1

            next_state, reward, done, info = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    return rewards