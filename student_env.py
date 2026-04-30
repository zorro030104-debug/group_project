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

        if action == 0:
            current_skill = self.add_skill
        elif action == 1:
            current_skill = self.sub_skill
        else:
            raise ValueError("Invalid action. Use 0 for addition or 1 for subtraction.")

        correct = np.random.rand() < current_skill

        if action == 0:
            if correct:
                self.add_skill += 0.05
            else:
                self.add_skill += 0.02
        else:
            if correct:
                self.sub_skill += 0.05
            else:
                self.sub_skill += 0.02

        self.add_skill = min(self.add_skill, 1.0)
        self.sub_skill = min(self.sub_skill, 1.0)

        new_total_skill = self.add_skill + self.sub_skill
        reward = new_total_skill - old_total_skill

        next_state = self.get_state()
        done = self.step_count >= self.max_steps

        info = {
            "correct": correct,
            "add_skill": self.add_skill,
            "sub_skill": self.sub_skill
        }

        return next_state, reward, done, info