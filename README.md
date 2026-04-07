# Fridge Elephant Gym
A lightweight **Gymnasium/Gym-compatible** reinforcement learning benchmark for **dependency-constrained sequential tasks**.
The task follows a strict workflow:
**Open door → Move/Align into fridge region → Close door**
This repository is designed for reproducible AGI-oriented benchmarking and educational RL experiments.
---
## Highlights
- **Environment**: `FridgeGameEnv` with 5D observations and 6D one-hot actions
- **Baselines**:
  - `RuleBasedAgent` (interpretable deterministic policy)
  - `DQNAgent` (replay buffer + target network + epsilon-greedy)
- **Evaluation**:
  - separates behavior-policy success and greedy-policy success
  - supports best-checkpoint restoration based on greedy performance
- **Visualization**:
  - pygame-based human mode for debugging/demo
  - headless mode for efficient training
---
## Observation and Action
### Observation (5D)
\[
[door\_open,\ elephant\_x(m),\ elephant\_y(m),\ fridge\_x(m),\ fridge\_y(m)]
\]
- `door_open`: 0/1
- `elephant_x`, `elephant_y`: elephant coordinates (meters)
- `fridge_x`, `fridge_y`: fridge coordinates (meters)
- `inside` is inferred from coordinate proximity
### Action (6D one-hot)
- `[1,0,0,0,0,0]`: open
- `[0,1,0,0,0,0]`: close
- `[0,0,1,0,0,0]`: up
- `[0,0,0,1,0,0]`: down
- `[0,0,0,0,1,0]`: left
- `[0,0,0,0,0,1]`: right
---
## Install
```bash
pip install -r requirements.txt
Main dependencies: gymnasium (or gym), pygame, numpy, torch (for DQN).

Run
python examples/demo.py
Demo controls
1: manual mode
2: rule-based auto mode
3: DQN training
4: DQN greedy execution
H: save current manual position as training/eval start
K: clear DQN progress
R: reset episode
Manual test keys:

Arrow keys: move elephant (manual debug only, not RL action space)
O: open
C: close
W/A/S/D: up/left/down/right (RL action keys)
Reproducibility Notes
Keep training env and visualization env parameters consistent.
Report behavior-policy and greedy-policy metrics separately.
Use multiple seeds for stable comparisons.
Project Structure
fridge_gym/
  envs/         # FridgeGameEnv
  agents/       # Rule-based and DQN agents
  elements/     # Entity definitions
  utils/        # Rendering helpers
examples/
  demo.py       # Interactive demo and training entry
docs/
  THESIS_CORE_OUTLINE.md
  DOUBAO_THESIS_PROMPT.md
Paper Submission Context
This project is used for preparing an AGI conference paper draft under LNCS format.

CFP reference: AGI-26 Call for Papers

