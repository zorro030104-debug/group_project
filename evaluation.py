import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, window=100):
    """
    Smooth the curve using moving average.
    """
    return np.convolve(data, np.ones(window) / window, mode="valid")

def evaluate_baselines(random_rewards, weakest_rewards):
    """
    Return summary statistics for baseline methods.
    """
    results = {
        "Random": np.mean(random_rewards),
        "Weakest": np.mean(weakest_rewards)
    }
    return results

def plot_training_rewards(q_results, sarsa_results, window=100):
    """
    Plot reward curves for Q-learning and SARSA.
    """
    q_rewards_smooth = moving_average(q_results["rewards"], window)
    sarsa_rewards_smooth = moving_average(sarsa_results["rewards"], window)

    plt.figure(figsize=(8, 5))
    plt.plot(q_rewards_smooth, label="Q-learning")
    plt.plot(sarsa_rewards_smooth, label="SARSA")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_training_accuracy(q_results, sarsa_results, window=100):
    """
    Plot accuracy curves for Q-learning and SARSA.
    """
    q_acc_smooth = moving_average(q_results["accuracy"], window)
    sarsa_acc_smooth = moving_average(sarsa_results["accuracy"], window)

    plt.figure(figsize=(8, 5))
    plt.plot(q_acc_smooth, label="Q-learning")
    plt.plot(sarsa_acc_smooth, label="SARSA")
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_policy_comparison(q_results, sarsa_results, random_rewards, weakest_rewards):
    """
    Compare RL models with baseline methods.
    """
    q_avg = np.mean(q_results["rewards"][-100:])
    sarsa_avg = np.mean(sarsa_results["rewards"][-100:])
    random_avg = np.mean(random_rewards)
    weakest_avg = np.mean(weakest_rewards)

    labels = ["Q-learning", "SARSA", "Random", "Weakest"]
    values = [q_avg, sarsa_avg, random_avg, weakest_avg]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom"
        )

    plt.ylabel("Average Total Reward")
    plt.title("RL vs Baseline Performance Comparison")

    bars[0].set_color("steelblue")
    bars[1].set_color("orange")

    plt.grid(axis="y")
    plt.show()

def plot_accuracy_comparison(q_results, sarsa_results):
    """
    Compare final accuracy of Q-learning and SARSA.
    """
    labels = ["Q-learning", "SARSA"]
    values = [
        np.mean(q_results["accuracy"][-100:]),
        np.mean(sarsa_results["accuracy"][-100:])
    ]

    plt.figure(figsize=(6, 5))
    plt.bar(labels, values)
    plt.ylabel("Average Accuracy")
    plt.title("Final Accuracy Comparison")
    plt.ylim(0, 1)
    plt.grid(axis="y")
    plt.show()

def print_learned_policy(q_table, model_name="Model"):
    """
    Print the learned policy for each state.
    State:
        0 = low, 1 = medium, 2 = high
    Action:
        0 = Addition, 1 = Subtraction
    """
    level_names = ["Low", "Medium", "High"]

    print(f"\nLearned Policy: {model_name}")
    print("-" * 40)

    for add_level in range(3):
        for sub_level in range(3):
            state_idx = add_level * 3 + sub_level
            best_action = np.argmax(q_table[state_idx])
            action_name = "Addition" if best_action == 0 else "Subtraction"

            print(
                f"Addition: {level_names[add_level]}, "
                f"Subtraction: {level_names[sub_level]} "
                f"-> Choose {action_name}"
            )