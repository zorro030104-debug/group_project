import numpy as np
from student_env import MathTutorEnv, state_to_index


NUM_STATES = 9
NUM_ACTIONS = 2


def choose_action(state_idx, q_table, epsilon):
    """
    Epsilon-greedy action selection.
    """
    if np.random.rand() < epsilon:
        return np.random.choice(NUM_ACTIONS)
    else:
        return np.argmax(q_table[state_idx])


def train_sarsa(
    episodes=3000,
    alpha=0.1,
    gamma=0.95,
    epsilon=0.2,
    max_steps=30
):
    """
    Train a tabular SARSA agent.

    SARSA is an on-policy RL algorithm.
    It updates Q-values using the action actually selected
    in the next state.
    """
    env = MathTutorEnv(max_steps=max_steps)
    q_table = np.zeros((NUM_STATES, NUM_ACTIONS))

    episode_rewards = []
    episode_accuracy = []
    final_add_skills = []
    final_sub_skills = []

    for episode in range(episodes):
        state = env.reset()
        state_idx = state_to_index(state)

        action = choose_action(state_idx, q_table, epsilon)

        total_reward = 0
        correct_count = 0
        done = False

        while not done:
            next_state, reward, done, info = env.step(action)
            next_state_idx = state_to_index(next_state)

            next_action = choose_action(next_state_idx, q_table, epsilon)

            # SARSA update:
            # Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
            q_table[state_idx, action] = q_table[state_idx, action] + alpha * (
                reward + gamma * q_table[next_state_idx, next_action]
                - q_table[state_idx, action]
            )

            state_idx = next_state_idx
            action = next_action

            total_reward += reward

            if info["correct"]:
                correct_count += 1

        episode_rewards.append(total_reward)
        episode_accuracy.append(correct_count / max_steps)
        final_add_skills.append(env.add_skill)
        final_sub_skills.append(env.sub_skill)

    results = {
        "rewards": episode_rewards,
        "accuracy": episode_accuracy,
        "final_add_skills": final_add_skills,
        "final_sub_skills": final_sub_skills
    }

    return q_table, results