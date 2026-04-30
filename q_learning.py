import numpy as np
from student_env import MathTutorEnv, state_to_index

NUM_STATES = 9
NUM_ACTIONS = 2

def choose_action(state_idx, q_table, epsilon):
    """
    Choose an action using epsilon-greedy policy.
    """
    if np.random.rand() < epsilon:
        return np.random.choice(NUM_ACTIONS)
    else:
        return np.argmax(q_table[state_idx])

def train_q_learning(
    episodes=1000,
    alpha=0.1,
    gamma=0.95,
    epsilon=0.2,
    max_steps=30
):
    """
    Train a tabular Q-learning agent.
    """
    env = MathTutorEnv(max_steps=max_steps)
    q_table = np.zeros((NUM_STATES, NUM_ACTIONS))

    episode_rewards = []

    for episode in range(episodes):
        state = env.reset()
        state_idx = state_to_index(state)

        total_reward = 0
        done = False

        while not done:
            action = choose_action(state_idx, q_table, epsilon)

            next_state, reward, done, info = env.step(action)
            next_state_idx = state_to_index(next_state)

            q_table[state_idx, action] = q_table[state_idx, action] + alpha * (
                reward + gamma * np.max(q_table[next_state_idx]) - q_table[state_idx, action]
            )

            state_idx = next_state_idx
            total_reward += reward

        episode_rewards.append(total_reward)

    return q_table, episode_rewards