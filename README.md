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

Q-learning is first implemented to learn an effective tutoring policy and already outperforms baseline strategies. 
We then introduce SARSA to compare on-policy and off-policy learning approaches and evaluate how different learning strategies perform in this tutoring environment.

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
| Q-learning | ~20.7     |
| SARSA      | ~19.7     |
| Random     | ~16.2     |
| Weakest    | ~15.7     |

See the plots in `main.ipynb` for training curves and detailed comparisons.

### Key Findings

- Both reinforcement learning methods significantly outperform baseline strategies
- Q-learning achieves slightly better performance than SARSA after environment refinement
- Adaptive learning strategies outperform fixed heuristics such as Random and Weakest-skill

### Additional Analysis

The change in performance between Q-learning and SARSA highlights the impact of environment design.
In earlier versions, SARSA performed better due to its conservative nature in uncertain environments.
After refining the reward function and learning dynamics, the environment became more stable, allowing Q-learning to better approximate the optimal policy.

---

## Key Insights

- Adaptive learning is more effective than static heuristics
- Both Q-learning and SARSA successfully learn effective tutoring strategies
- After refining the environment and reward function, Q-learning achieves slightly better performance
- This suggests that a more stable and structured environment benefits off-policy learning
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