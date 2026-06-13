<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&pause=1000&width=435&lines=%F0%9F%A6%85+Automated+FlappyBird+using+DeepRL" alt="Typing SVG" /></a>

A sleek and efficient implementation of a Reinforcement Learning agent trained to play the classic Flappy Bird game. The agent transitions from random, unstable actions to highly precise maneuvering by calculating optimum trajectories across successive episodes.

---

## 📺 Training Highlights

<p align="center">
  <img src="img/flappy.gif" alt="Flappy Bird RL Terminal Training Progress" width="600px" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);"/>
</p>

---

## 🛠️ Tech Stack & Dependencies

Built entirely using Python and specialized libraries for building simulated environments, managing state spaces, and tracking training telemetry:

* **Game Framework:** `pygame` — Custom, lightweight environment loop rendering bird positions, pipelines, and continuous collisions.
* **Numerical & Array Calculations:** `numpy` — Controls state discretization, Q-table indexing, and tracking reward arrays.
* **Reinforcement Learning Framework:** `torch` & `gymnasium` — Implements the core RL algorithms, neural networks, and standardized agent-environment interfaces.
* **Configuration Management:** `yaml` — Handles hyperparameters, training environment rules, and model settings from an external configuration file.
* **System Controls:** `os` — Manages training directories, log checkpoints, and model file paths.
* **Loop Utilities:** `itertools` — Provides efficient, infinite looping mechanisms for continuous training episodes.
* **Stochastic Processing:** `random` — Generates random numbers for exploration scheduling ($\epsilon$-greedy policies) and randomized environment seeding.
* 
---

## 🧠 How It Works

1. **State Spaces:** The bird senses its current environment by calculating structural boundaries (e.g., horizontal/vertical distances to the upcoming pipeline, current falling velocity).
2. **Action Mechanics:** The agent optimizes a binary choice space: `Action 0` (Do Nothing / Fall) or `Action 1` (Flap).
3. **Reward Alignment:** The framework continuously penalizes crashes into obstructions while tracking progressive rewards for maintaining active frames or cleaning gaps.

---

## 🚀 Usage Instructions

Clone the repository, verify that your modules are installed, and execute your model using the two specialized run commands below:

### 1. Training the Agent
To start or resume the training routine across multiple simulation episodes, execute the environment with the `--train` flag:
```bash
python agent.py flappybirdv0 --train
```

### 2. Testing the Trained Agent
Once weights or Q-values are generated, watch the agent exploit its learned policies and test its capabilities using the execution command below:
```bash
python agent.py flappybirdv0
```
