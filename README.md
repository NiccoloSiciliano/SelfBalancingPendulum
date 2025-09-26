# Self-Balancing AI: Modified Cartpole with Q-Learning

A reinforcement learning implementation that solves an enhanced version of the classic Cartpole (Inverted Pendulum) problem using Q-Learning algorithms.

## Authors
- Siciliano Niccolò (1958541)
- Paossi Davide (1950062)

## Overview

This project implements a self-balancing inverted pendulum system that goes beyond the traditional Cartpole problem. Our AI agent can not only maintain balance when the pole is upright but also swing the pendulum from any starting position back to the vertical equilibrium.

### Demonstration Video
Watch our AI in action: [Self-Balancing AI Demonstration](https://www.youtube.com/watch?v=4U_HstrUaH0)

### Research Paper
For comprehensive technical details, methodology, and experimental analysis, see the full research paper: Ai_Lab_self_balancing_AI.pdf

### Key Features
- **Enhanced Problem Scope**: Handles pendulum recovery from any angle (0-360°)
- **Realistic Physics**: Incorporates gravity and air friction
- **Environmental Challenges**: Includes random wind forces to test robustness
- **Advanced Reward System**: Custom reward function optimized for both balance and swing-up
- **Comprehensive State Space**: 3D state representation (angle, direction, velocity)

## Problem Formulation

The system is modeled as a Markov Decision Process (MDP) with:
- **State Space**: 3 variables (pole angle, direction, velocity)
- **Action Space**: 20 discretized speed actions (10 left, 10 right)
- **Objective**: Maximize cumulative reward while maintaining balance with minimal velocity

## Technical Implementation

### Dependencies
- Python
- PyGame (graphics visualization)
- PyMunk (physics simulation)

### Core Algorithm
Implements Q-Learning with the update rule:
```
Q_new(s_t, a_t) ← (1-α) * Q(s_t, a_t) + α * (r_t + γ * max Q(s_t+1, a))
```

### State Discretization
- **Angle**: 40 equal-sized bins
- **Velocity**: 20 variable-sized bins (larger bins for higher speeds)
- **Total State Space**: ~32,000 states

### Reward Function
Our optimized reward function balances position and velocity:
```
α * sin(angle) + β * (20 - speed)/20
```
Where α = β = 0.5, providing maximum reward at 90° with zero velocity.

## Experimental Results

### Performance Metrics
- Achieves stable balancing for extended periods
- Peak average reward: 0.66 at ~10k episodes
- Sustained performance: ~0.5 average reward after convergence
- Optimal learning rate: 0.1

### Training Innovations
1. **Random Initialization**: Episodes start from random angles for better exploration
2. **Grace Period**: Initial actions before failure conditions apply
3. **Linear Decay**: Better performance than exponential decay strategies

## Known Limitations

- **CPU-Only Implementation**: No GPU acceleration support
- **Training Time**: Computationally intensive (hours for 100k episodes)
- **Catastrophic Forgetting**: Performance may degrade after peak learning
- **State Space Constraints**: Limited by computational feasibility

## Getting Started

### Installation
```bash
pip install pygame pymunk
```

### Running the Simulation
```bash
python main.py
```

### Training Parameters
- Learning Rate (α): 0.1 (recommended)
- Discount Factor (γ): < 1.0
- Epsilon Decay: Linear strategy
- Episodes: 10,000+ for optimal performance

## Project Structure
```
├── main.py                 # Main simulation loop
├── q_learning.py          # Q-Learning implementation
├── environment.py         # Physics environment setup
├── reward_functions.py    # Reward calculation methods
├── visualization.py       # PyGame graphics
└── README.md             # This file
```

## Future Improvements

- GPU acceleration for faster training
- Larger state space exploration
- Advanced neural network integration (Deep Q-Learning)
- Multi-agent scenarios
- Real-world hardware implementation

## Research Context

This implementation serves as a benchmark for reinforcement learning algorithms in continuous control problems. The enhanced Cartpole environment provides a more challenging and realistic testing ground compared to traditional implementations.

## License

This project was developed for academic research purposes at AI Lab.

## References

Based on research conducted on September 26, 2025, exploring advanced Q-Learning applications in control systems.

---

*For technical questions or collaboration opportunities, please refer to the original research paper or contact the authors.*
