# 10-Armed Bandit

This repository contains a Python implementation of a 10-armed bandit problem, which demonstrates the exploration-exploitation trade-off in reinforcement learning.

## Overview

The 10-armed bandit problem is a classic problem in reinforcement learning where an agent (or gambler) faces a series of 10 slot machines (bandits) with different payout probabilities. The agent's objective is to maximize the cumulative reward over a series of plays by balancing exploration (trying out different bandits) and exploitation (choosing the bandit with the highest expected payout).

## Features

- Implementation of various strategies:
  - Epsilon-Greedy
  - Optimistic initial value
  - Upper Confidence Bound (UCB)

- Visualization of results: The repository includes visualizations to showcase the performance of different strategies in terms of cumulative rewards.

## Prerequisites

- Python 3.9 or later

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/10-armed-bandit.git
   cd 10-armed-bandit

2. Install dependencies
   ```bash
   pip install numpy matplotlib

3. To run the code:
   ```bash
   python main.py

## Results
The results of running the different strategies can be found in the `results` directory. Visualizations demonstrate the cumulative rewards obtained by each strategy over time.
   
