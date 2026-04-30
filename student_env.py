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

        if action == 0:
            skill = self.add_skill
        elif action == 1:
            skill = self.sub_skill
        else:
            raise ValueError("Invalid action. Use 0 for addition or 1 for subtraction.")

        # OPTIMIZED: Added question difficulty so correctness depends on both student skill and task challenge.
        difficulty = 0.05  # fixed difficulty for all questions, can be randomized or scaled with skill if desired

        # OPTIMIZED: Probability of answering correctly is now skill-adjusted instead of using raw skill directly.
        prob_correct = skill - difficulty

        # OPTIMIZED: Clip probability to avoid impossible values such as below 0 or above 1.
        prob_correct = np.clip(prob_correct, 0.05, 0.95)

        correct = np.random.rand() < prob_correct

        # OPTIMIZED: Reduced skill increase after a correct answer to make learning more gradual and realistic.
        correct_gain = 0.04

        # OPTIMIZED: Added smaller learning gain after an incorrect answer to reflect learning from mistakes.
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

        # OPTIMIZED: Changed reward from skill gain to performance-based reward.
        # This better matches the tutoring goal: encourage questions the student can answer correctly.
        reward = 1 if correct else -0.2

        next_state = self.get_state()
        done = self.step_count >= self.max_steps

        info = {
            "correct": correct,
            "add_skill": self.add_skill,
            "sub_skill": self.sub_skill,

            # OPTIMIZED: Added prob_correct to help evaluate and debug the simulation.
            "prob_correct": prob_correct
        }

        return next_state, reward, done, info