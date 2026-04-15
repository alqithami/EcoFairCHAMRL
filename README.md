# EcoFair-CH-MARL: Scalable Constrained Hierarchical Multi-Agent RL for Maritime Logistics
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2603.14625-b31b1b.svg)](https://arxiv.org/abs/2603.14625 )

This project implements **EcoFair-CH-MARL**, a multi-agent reinforcement learning (MARL) framework designed to simulate and optimize maritime logistics operations. It focuses on achieving simultaneous efficiency, environmental sustainability through real-time emission budgets, and equitable resource distribution via fairness guarantees. The framework incorporates a novel hierarchical control structure within a realistic digital twin environment, addressing complexities and stochasticities inherent in real-world maritime scenarios.

## Table of Contents

1.  [Features](#features)
2.  [Installation](#installation)
3.  [Usage](#usage)
    *   [Basic Execution](#basic-execution)
    *   [Command-Line Arguments](#command-line-arguments)
    *   [Understanding the Advanced Travel Logic](#understanding-the-advanced-travel-logic)
    *   [Debugging Mode](#debugging-mode)
4.  [Ablation Studies](#ablation-studies)
5.  [Fairness Metrics](#fairness-metrics)
6.  [Convergence Approximation Experiment](#convergence-approximation-experiment)
7.  [Output Files](#output-files)
8.  [Acknowledgements](#acknowledgements)

## 1. Features

*   **Hierarchical Control Architecture (Implicit):** The environment is structured to facilitate hierarchical learning. A single RL agent learns to make both strategic (high-level) and tactical (low-level) decisions. High-level directives (e.g., destination ports) are updated at a coarser timescale (`--hl_update_interval`), while low-level actions (e.g., vessel speed) are applied at every step, allowing the agent to learn complex, multi-timescale policies.
*   **Realistic Vessel Movement & Dynamics:**
    *   **Multi-Step Journeys:** Vessels now undertake realistic multi-step journeys, consuming fuel over time as they travel between ports.
    *   **Dynamic Queuing System:** Vessels intelligently handle port congestion by entering a `QUEUING` state if their chosen destination is at capacity, either at their current location or upon arrival at a full destination port.
    *   **Stochastic Weather Effects:** Random storms are simulated, dynamically reducing vessel speed and increasing fuel consumption, adding crucial real-world stochasticity.
    *   **Idle/Queue Fuel Consumption:** Fuel consumption for idle and queuing states is now dependent on each vessel's size/specifications, preventing Gini coefficients from being artificially forced to zero.
*   **Emission Cap Mechanism:** Implements a real-time emission budget, penalizing agents if cumulative emissions exceed a predefined cap, thereby encouraging environmentally conscious behavior.
*   **Fairness Shaping:** Integrates explicit fairness objectives into the reward function to promote equitable fuel usage among vessels.
    *   **Gini Coefficient:** Measures the inequality of fuel consumption distribution among vessels.
    *   **Max-Min Ratio:** Indicates how close the least-efficient vessel's fuel usage is to the most-efficient, providing insight into the minimum performance guarantee.
*   **Algorithm-Agnostic Runner:** The framework supports and demonstrates compatibility with various reinforcement learning algorithms:
    *   **PPO (Proximal Policy Optimization):** The robust default algorithm.
    *   **SOTO (Socially Optimized Trade-Off):** Applies fairness through a dedicated reward wrapper.
    *   **FEN (Fairness-Enhanced Navigation):** Applies fairness through a dedicated reward wrapper.
    *   **QMIX & MAPPO:** The framework is designed to integrate with these advanced multi-agent algorithms from `sb3_contrib`, demonstrating its flexibility for future research and broader applicability.
*   **Scalability Demonstration:** The environment is designed to scale to larger numbers of ports and vessels, with experiments conducted on configurations up to 16 ports and 50 vessels, showcasing the framework's ability to handle increased complexity.
*   **Comprehensive Logging & Analysis:** Explicit fairness metrics and episode returns are logged and saved to CSV files for detailed analysis. This includes granular logging of fairness metrics throughout the entire training process, providing insights into the evolution of fairness over time.
*   **Convergence Approximation Experiment:** A simplified toy model to visualize the convergence behavior of hierarchical multi-agent systems with fairness considerations, aiding in theoretical understanding.

## 2. Installation

1.  **Prerequisites:** Ensure you have Python 3.8+ installed.
2.  **Save the Code:** Save the provided Python script as `EcoFairCHMARL.py`.
3.  **Install Dependencies:** Open your terminal or command prompt and run:
    ```bash
    pip install gymnasium numpy pandas matplotlib stable-baselines3
    ```
4.  **Optional: `sb3_contrib` for QMIX/MAPPO:** If you intend to experiment with QMIX or MAPPO, you'll need to install `sb3_contrib`.
    ```bash
    pip install sb3_contrib
    ```

## 3. Usage

The simulation can be run from the command line with various options to configure the environment and training process.

### Basic Execution

To run the simulation with default settings (PPO algorithm, no emission cap, no fairness penalty applied directly in the environment, 1000 episodes, 8 ports, 20 vessels, and a high-level update interval of 5 steps):

```bash
python EcoFairCHAMRL.py
```

### Command-Line Arguments

You can customize the simulation using the following arguments:

*   `--algo {PPO,QMIX,MAPPO,SOTO,FEN}`:
    *   Specifies the learning algorithm or fairness baseline.
    *   Default: `PPO`.
    *   `SOTO` and `FEN` apply fairness through reward wrappers.
    *   `QMIX` and `MAPPO` will fall back to `PPO` if `sb3_contrib` is not installed or the algorithms are not found.
*   `--episodes <int>`:
    *   Number of training episodes.
    *   Default: `1000`.
*   `--emission_cap`:
    *   Flag to enable the emission cap mechanism.
    *   If set, agents are penalized for exceeding `emission_cap_value`.
*   `--fairness`:
    *   Flag to enable the fairness penalty directly in the environment's reward function (when `algo` is `PPO`).
    *   If `algo` is `SOTO` or `FEN`, fairness is handled by their respective reward wrappers, and this flag is internally set to `False` for the base environment.
*   `--lambda_fair <float>`:
    *   The weighting coefficient for the fairness penalty in the reward function.
    *   Default: `10.0`.
*   `--num_ports <int>`:
    *   Number of ports in the simulation environment.
    *   Default: `8`.
*   `--num_vessels <int>`:
    *   Number of vessels in the simulation environment.
    *   Default: `20`.
*   `--outdir <path>`:
    *   Directory to save results (CSV files and plots).
    *   Default: `results/`.
*   `--convergence`:
    *   Flag to run the separate convergence approximation demo instead of the main MARL simulation.
*   `--debug`:
    *   Flag to enable verbose debugging output during simulation steps. Highly recommended for understanding agent behavior and environment dynamics.
*   `--hl_update_interval <int>`:
    *   **NEW!** The interval (in simulation steps) at which the high-level directive (destination port) for vessels is updated. This simulates the coarser timescale of strategic decisions.
    *   Default: `5`.

### Understanding the Advanced Travel Logic

With the updated environment, vessel movement is significantly more realistic and complex:

*   **Multi-Step Journeys:** When a vessel chooses a destination, it enters a `TRAVELING` state. It will consume fuel and cover distance over multiple simulation steps until its `remaining_distance` reaches zero.
*   **Dynamic Queuing:**
    *   If an `IDLE` vessel attempts to travel to a port that is currently at its capacity, it will enter a `QUEUING` state at its *current* port. It will wait there, consuming queue fuel, until capacity becomes available.
    *   If a `TRAVELING` vessel arrives at its `destination_port` and that port is full, it will enter a `QUEUING` state *at the destination port*. It will then wait there until capacity opens up.
*   **Stochasticity:** Weather events (storms) can occur randomly during travel, affecting the vessel's speed and fuel efficiency for that specific step.

This advanced logic means agents must learn to plan multi-step routes, consider dynamic port congestion, and adapt to environmental changes, making the learning problem more complex and realistic.

### Debugging Mode

The `--debug` flag provides extensive print statements during the simulation, which are invaluable for understanding the new travel logic and agent behavior:

*   **Vessel State Transitions:** Observe when vessels change status (IDLE, QUEUING, TRAVELING).
*   **Fuel Consumption:** Detailed breakdown of fuel used per vessel per step.
*   **Movement Details:** Distance covered, remaining distance, and effective speed during travel.
*   **Reward Components:** Observe how fuel cost, emission penalties, and fairness penalties contribute to the total reward at each step.
*   **Fairness Metrics:** Gini and Max-Min ratio values are printed at each step.

To enable debugging:
```bash
python EcoFairCHMARL.py --fairness --debug --hl_update_interval 1
```

## 4. Ablation Studies

Ablation studies allow you to isolate the impact of different components of the reward function or environment.

*   **Baseline (No Emission Cap, No Fairness):**
    ```bash
    python EcoFairCHMARL.py --episodes 2000
    ```
*   **Emission Cap Only:**
    ```bash
    python EcoFairCHMARL.py --emission_cap --episodes 2000
    ```
*   **Fairness Only (using environment's direct penalty):**
    ```bash
    python EcoFairCHMARL.py --fairness --episodes 2000
    ```
*   **Fairness Only (using SOTO reward wrapper):**
    ```bash
    python EcoFairCHMARL.py --algo SOTO --episodes 2000
    ```
*   **Fairness Only (using FEN reward wrapper):**
    ```bash
    python EcoFairCHMARL.py --algo FEN --episodes 2000
    ```
*   **Both Emission Cap and Fairness:**
    ```bash
    python EcoFairCHMARL.py --emission_cap --fairness --episodes 2000
    ```
*   **Varying Fairness Weight (`lambda_fair`):**
    ```bash
    python EcoFairCHMARL.py --fairness --lambda_fair 5.0 --episodes 2000
    python EcoFairCHMARL.py --fairness --lambda_fair 20.0 --episodes 2000
    ```

Remember to adjust `--episodes` as needed for sufficient training time.

## 5. Fairness Metrics

The project explicitly tracks and logs two key fairness metrics:

*   **Gini Coefficient:** A measure of statistical dispersion, here applied to the distribution of fuel consumption among vessels. A Gini coefficient of 0 expresses perfect equality (all vessels use the same amount of fuel), while a coefficient of 1 expresses maximal inequality (one vessel uses all the fuel).
*   **Max-Min Ratio:** Calculated as `min(fuel_usage) / max(fuel_usage)`. This metric indicates how close the fuel consumption of the least-consuming vessel is to the most-consuming vessel. A ratio of 1 indicates perfect equality, while a ratio closer to 0 indicates greater disparity.

These metrics are saved in `fairness_metrics_*.csv` (for evaluation episodes) and `training_fairness_metrics_*.csv` (for all training episodes).

## 6. Convergence Approximation Experiment

This is a separate, simplified toy model that demonstrates the concept of convergence in a hierarchical multi-agent system with fairness. It's useful for understanding the theoretical underpinnings of fairness shaping without the complexity of the full maritime environment.

To run this experiment:

```bash
python EcoFairCHMARL.py --convergence
```

This will generate a plot (`CHMARL_Refined_Fairness.png`) in your output directory visualizing the approximation.

## 7. Output Files

All generated data and plots are saved in the `results/` directory by default (or the path specified by `--outdir`).

*   `results_<algo_name>.csv`: Contains the average returns for the 10-episode evaluation phase after training.
*   `fairness_metrics_<algo_name>.csv`: Contains the Gini coefficient and Max-Min ratio for the 10-episode evaluation phase.
*   `training_fairness_metrics_<algo_name>.csv`: Logs the Gini coefficient and Max-Min ratio for *every episode* during the training process. This is crucial for observing the evolution of fairness over time.
*   `ppo_only_test_returns.csv`: Generated by the legacy `run_experiment` function (if called directly).
*   `CHMARL_Refined_Fairness.png`: The plot generated by the `--convergence` experiment.

## 8. Acknowledgements

This project builds upon foundational concepts in multi-agent reinforcement learning and environmental simulation. Specific inspirations and components are noted within the code comments.
