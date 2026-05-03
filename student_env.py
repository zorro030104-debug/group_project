import numpy as np

def discretize_skill(skill):
    """
    Convert continuous skill level into discrete bins.
    0 = low, 1 = medium, 2 = high
    """
    if skill < 0.4:
        return 0
    elif skill < 0.7:
        return 1
    else:
        return 2

def state_to_index(state):
    """
    Convert state tuple into index for Q-table.
    state = (addition_level, subtraction_level)
    """
    add_level, sub_level = state
    return add_level * 3 + sub_level

class MathTutorEnv:
    """
    A simulated math tutoring environment.

    State:
        (addition_skill_level, subtraction_skill_level)

    Actions:
        0 = give an addition question
        1 = give a subtraction question

    Reward:
        improvement in total student skill after one interaction
    """

    def __init__(self, max_steps=30):
        self.max_steps = max_steps
        self.step_count = 0
        self.reset()

    def reset(self):
        self.add_skill = np.random.uniform(0.2, 0.6)
        self.sub_skill = np.random.uniform(0.2, 0.6)
        self.step_count = 0
        return self.get_state()

    def get_state(self):
        return (
            discretize_skill(self.add_skill),
            discretize_skill(self.sub_skill)
        )

    def step(self, action):
        self.step_count += 1

        old_total_skill = self.add_skill + self.sub_skill
        old_gap = abs(self.add_skill - self.sub_skill)

        if action == 0:
            skill = self.add_skill
        elif action == 1:
            skill = self.sub_skill
        else:
            raise ValueError("Invalid action. Use 0 for addition or 1 for subtraction.")

        difficulty = 0.05
        prob_correct = skill - difficulty
        prob_correct = np.clip(prob_correct, 0.05, 0.95)

        correct = np.random.rand() < prob_correct

        correct_gain = 0.04
        incorrect_gain = 0.01

        if action == 0:
            if correct:
                self.add_skill += correct_gain
            else:
                self.add_skill += incorrect_gain
        else:
            if correct:
                self.sub_skill += correct_gain
            else:
                self.sub_skill += incorrect_gain

        self.add_skill = np.clip(self.add_skill, 0.0, 1.0)
        self.sub_skill = np.clip(self.sub_skill, 0.0, 1.0)

        new_total_skill = self.add_skill + self.sub_skill
        new_gap = abs(self.add_skill - self.sub_skill)

        performance_reward = 1 if correct else -0.2
        learning_gain = new_total_skill - old_total_skill
        balance_gain = old_gap - new_gap

        # OPTIMIZED: reward now considers correctness, learning progress, and balanced skill development.
        reward = performance_reward + 5 * learning_gain + 2 * balance_gain

        next_state = self.get_state()
        done = self.step_count >= self.max_steps

        info = {
            "correct": correct,
            "add_skill": self.add_skill,
            "sub_skill": self.sub_skill,
            "prob_correct": prob_correct,
            "learning_gain": learning_gain,
            "balance_gain": balance_gain
        }

        return next_state, reward, done, info