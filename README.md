**Multi-Armed Bandit Recommendation System**
**Overview**

This project implements and compares different Multi-Armed Bandit (MAB) strategies for personalized recommendation across three users.

The goal is to maximize cumulative reward while balancing exploration and exploitation.

**Implemented Strategies**

Upper Confidence Bound (UCB)

SoftMax (Boltzmann Exploration)

**Project Structure**

├── data/
├── ucb_strategy.py
├── softmax_strategy.py
├── plotting.ipynb
├── results/
└── README.md

**Methodology**
1. **Data Preprocessing**

Encoded categorical variables using Label Encoding.

Handled unseen labels in test data safely.

2. **Reward Simulation**

Rewards generated per user interaction.

Stored for cumulative analysis.

3. **Strategy Implementation**
**UCB**

Selects arms using:

UCB = mean_reward + c * sqrt(log(t) / n)

**SoftMax**

Uses probabilistic selection:

P(a) = exp(Q(a)/τ) / Σ exp(Q/τ)

**Experiments Conducted**

* Individual user reward plots

* Strategy comparison plots

* Hyperparameter sensitivity analysis

* Combined cumulative reward graphs

**Results Summary**

* UCB converged faster for deterministic rewards.

* SoftMax handled stochastic rewards better.

Moderate hyperparameters gave optimal performance.

**How to Run**

* Install dependencies:
pip install numpy pandas matplotlib scikit-learn

* Run strategies:
python ucb_strategy.py
python softmax_strategy.py

Generate plots:
Open plotting.ipynb

**Future Improvements**

* Thompson Sampling implementation

* Contextual Bandits

Deep Reinforcement Learning integration

**Author**

Manan Gadesha

**Approach**

We built a contextual news recommendation system using user classification as context and news categories as arms.

**Algorithms Implemented**

* Epsilon-Greedy
* Upper Confidence Bound (UCB)
* SoftMax

**Simulation**

Each algorithm was simulated for 10,000 steps using the provided sampler.

**Key Findings**

* UCB achieved the highest average reward.
* Epsilon required hyperparameter tuning.
* SoftMax performance depended on temperature.