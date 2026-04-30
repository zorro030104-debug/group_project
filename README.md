# Adaptive Math Tutoring using Reinforcement Learning

## Project Overview

This project explores how reinforcement learning (RL) can be used to improve personalized tutoring strategies. We model the tutoring process as a sequential decision-making problem and apply tabular RL methods to adaptively select question types for students.

The goal is to maximize student learning performance by dynamically choosing between different types of math questions based on the student’s current skill level.

---

## Problem Formulation (MDP)

We formulate the tutoring problem as a Markov Decision Process (MDP):

- **State**:  
  Discretized student skill levels in:
  - Addition (low / medium / high)
  - Subtraction (low / medium / high)

- **Action**:  
  - 0 = Assign addition question  
  - 1 = Assign subtraction question  

- **Transition**:  
  Student skill updates based on:
  - Whether the student answers correctly
  - Learning gains from practice

- **Reward**:  
  - +1 for correct answer  
  - -0.2 for incorrect answer  

This formulation captures the sequential and uncertain nature of student learning.

---

## Methods

### Reinforcement Learning Models
- Q-learning (off-policy)
- SARSA (on-policy)

### Baselines
- Random policy
- Weakest-skill heuristic

---

## Environment Design

- Initial skill levels are randomly initialized  
- Question difficulty affects correctness probability  
- Skills improve gradually with practice  
- Incorrect answers still provide small learning gains  

---

## Results

### Performance Comparison

| Method      | Avg Reward |
|------------|-----------|
| SARSA      | ~18.14    |
| Q-learning | ~17.21    |
| Random     | ~12.67    |
| Weakest    | ~11.76    |

See the plots in `main.ipynb` for training curves and detailed comparisons.

---

## Key Insights

- Adaptive learning is more effective than static heuristics  
- SARSA performs better due to its on-policy nature  
- RL balances:
  - Reinforcing strengths  
  - Addressing weaknesses  

---

## Project Structure

```text
student_env.py      - Environment (MDP)
q_learning.py       - Q-learning
sarsa.py            - SARSA
baselines.py        - Baselines
evaluation.py       - Visualization
main.ipynb          - Experiments
```

---

## How to Run

Install dependencies:

```bash
pip install numpy matplotlib
```

Run:

- Open `main.ipynb`
- Execute all cells to reproduce results and plots

---

## Version Notes

### v0.1 Initial Model
- Built initial environment and Q-learning model  
- RL did not outperform baseline  

### v0.2 Improved Model
- Refined reward function and environment  
- Added SARSA  
- RL significantly outperforms baselines  

---

## Future Work

- Extend to more subjects  
- Use continuous state representations  
- Explore deep reinforcement learning  