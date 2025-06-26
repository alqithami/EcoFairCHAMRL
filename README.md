# Hierarchical Multi-Agent Maritime Logistics Simulation with Fairness and Emission Control

This project implements a hierarchical multi-agent reinforcement learning (MARL) environment designed to simulate maritime logistics operations. It focuses on optimizing vessel movements while considering critical real-world constraints such as emission caps and promoting fairness in resource (fuel) distribution among vessels. A key feature of this simulation is its realistic modeling of vessel travel, including multi-step journeys, queuing at busy ports, and dynamic environmental factors like stochastic weather.

## Table of Contents

1.  [Features](#features)
2.  [Installation](#installation)
3.  [Usage](#usage)
    *   [Basic Execution](#basic-execution)
    *   [Command-Line Arguments](#command-line-arguments)
    *   [Understanding the New Travel Logic](#understanding-the-new-travel-logic)
    *   [Debugging Mode](#debugging-mode)
4.  [Ablation Studies](#ablation-studies)
5.  [Fairness Metrics](#fairness-metrics)
6.  [Convergence Approximation Experiment](#convergence-approximation-experiment)
7.  [Output Files](#output-files)
8.  [Acknowledgements](#acknowledgements)

## 1. Features

*   **Multi-Agent Reinforcement Learning Environment:** Built using Gymnasium/Gym, providing a flexible platform for MARL research.
*   **Realistic Vessel Movement:**
    *   Vessels now undertake multi-step journeys, consuming fuel over time as they travel.
    *   Dynamic queuing system: Vessels can enter a queuing state at their current port if their chosen destination is full, or at the destination port if it becomes full upon arrival.
    *   Stochastic Weather Effects: Random storms can reduce vessel speed and increase fuel consumption during travel, adding real-world complexity.
*   **Emission Cap Mechanism:** Penalizes agents if total emissions exceed a predefined cap, encouraging environmentally conscious behavior.
*   **Fairness Shaping:** Integrates fairness objectives into the reward function to promote equitable fuel usage among vessels.
    *   **Gini Coefficient:** Measures the inequality of fuel consumption.
    *   **Max-Min Ratio:** Indicates how close the least-efficient vessel's fuel usage is to the most-efficient.
*   **Algorithm-Agnostic Runner:** Supports various reinforcement learning algorithms:
    *   **PPO (Proximal Policy Optimization):** Default and robust baseline.
    *   **SOTO (Socially Optimized Trade-Off):** Fairness applied via a reward wrapper.
    *   **FEN (Fairness-Enhanced Navigation):** Fairness applied via a reward wrapper.
    *   **QMIX & MAPPO:** Placeholders for future integration with `sb3_contrib` (requires separate installation).
*   **Comprehensive Logging:** Explicit fairness metrics and episode returns are logged and saved to CSV files for detailed analysis, including metrics tracked throughout the training process.
*   **Convergence Approximation Experiment:** A simplified toy model to visualize the convergence behavior of hierarchical multi-agent systems with fairness considerations.

## 2. Installation

1.  **Prerequisites:** Ensure you have Python 3.8+ installed.
2.  **Save the Code:** Save the provided Python script as `multi_agent_maritime_full.py`.
3.  **Install Dependencies:** Open your terminal or command prompt and run:
    ```bash
    pip install gymnasium numpy pandas matplotlib stable-baselines3
    ```
4.  **Optional: `sb3_contrib` for QMIX/MAPPO:** If you intend to experiment with QMIX or MAPPO, you'll need to install `sb3_contrib`. Note that these algorithms are currently placeholders and might require specific versions or additional setup.
    ```bash
    pip install sb3_contrib
    ```

## 3. Usage

The simulation can be run from the command line with various options to configure the environment and training process.

### Basic Execution

To run the simulation with default settings (PPO algorithm, no emission cap, no fairness penalty applied directly in the environment, 1000 episodes, 8 ports, 20 vessels):

```bash
python multi_agent_maritime_full.py
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
    *   **NEW!** Flag to enable verbose debugging output during simulation steps. Highly recommended for understanding agent behavior and environment dynamics.

### Understanding the New Travel Logic

With the updated environment, vessels no longer instantly teleport to a destination if the travel time is short. Instead:

*   **Multi-Step Journeys:** When a vessel chooses a destination, it enters a `TRAVELING` state. It will consume fuel and cover distance over multiple simulation steps until `remaining_distance` reaches zero.
*   **Queuing:**
    *   If an `IDLE` vessel attempts to travel to a port that is currently at its capacity, it will enter a `QUEUING` state at its *current* port. It will wait there, consuming queue fuel, until capacity becomes available.
    *   If a `TRAVELING` vessel arrives at its `destination_port` and that port is full, it will enter a `QUEUING` state *at the destination port*. It will then wait there until capacity opens up.
*   **Stochasticity:** Weather events (storms) can occur randomly during travel, affecting the vessel's speed and fuel efficiency for that specific step.

This new logic means agents must learn to plan multi-step routes, consider port congestion, and adapt to environmental changes, making the learning problem more complex and realistic.

### Debugging Mode

The `--debug` flag provides extensive print statements during the simulation, which are invaluable for understanding the new travel logic and agent behavior:

*   **Vessel State Transitions:** See when vessels change status (IDLE, QUEUING, TRAVELING).
*   **Fuel Consumption:** Detailed breakdown of fuel used per vessel per step.
*   **Movement Details:** Distance covered, remaining distance, and effective speed during travel.
*   **Reward Components:** Observe how fuel cost, emission penalties, and fairness penalties contribute to the total reward at each step.
*   **Fairness Metrics:** Gini and Max-Min ratio values are printed at each step.

To enable debugging:
```bash
python multi_agent_maritime_full.py --fairness --debug
```

## 4. Ablation Studies

Ablation studies allow you to isolate the impact of different components of the reward function or environment.

*   **Baseline (No Emission Cap, No Fairness):**
    ```bash
    python multi_agent_maritime_full.py --episodes 2000
    ```
*   **Emission Cap Only:**
    ```bash
    python multi_agent_maritime_full.py --emission_cap --episodes 2000
    ```
*   **Fairness Only (using environment's direct penalty):**
    ```bash
    python multi_agent_maritime_full.py --fairness --episodes 2000
    ```
*   **Fairness Only (using SOTO reward wrapper):**
    ```bash
    python multi_agent_maritime_full.py --algo SOTO --episodes 2000
    ```
*   **Fairness Only (using FEN reward wrapper):**
    ```bash
    python multi_agent_maritime_full.py --algo FEN --episodes 2000
    ```
*   **Both Emission Cap and Fairness:**
    ```bash
    python multi_agent_maritime_full.py --emission_cap --fairness --episodes 2000
    ```
*   **Varying Fairness Weight (`lambda_fair`):**
    ```bash
    python multi_agent_maritime_full.py --fairness --lambda_fair 5.0 --episodes 2000
    python multi_agent_maritime_full.py --fairness --lambda_fair 20.0 --episodes 2000
    ```

Remember to adjust `--episodes` as needed for sufficient training time.

## 5. Fairness Metrics

The project explicitly tracks and logs two key fairness metrics:

*   **Gini Coefficient:** A measure of statistical dispersion intended to represent the income or wealth distribution of a nation's residents, but here applied to the distribution of fuel consumption among vessels. A Gini coefficient of 0 expresses perfect equality (all vessels use the same amount of fuel), while a coefficient of 1 expresses maximal inequality (one vessel uses all the fuel).
*   **Max-Min Ratio:** Calculated as `min(fuel_usage) / max(fuel_usage)`. This metric indicates how close the fuel consumption of the least-consuming vessel is to the most-consuming vessel. A ratio of 1 indicates perfect equality, while a ratio closer to 0 indicates greater disparity.

These metrics are saved in `fairness_metrics_*.csv` (for evaluation episodes) and `training_fairness_metrics_*.csv` (for all training episodes).

## 6. Convergence Approximation Experiment

This is a separate, simplified toy model that demonstrates the concept of convergence in a hierarchical multi-agent system with fairness. It's useful for understanding the theoretical underpinnings of fairness shaping without the complexity of the full maritime environment.

To run this experiment:

```bash
python multi_agent_maritime_full.py --convergence
```

This will generate a plot (`CHMARL_Refined_Fairness.png`) in your output directory visualizing the approximation.

## 7. Output Files

All generated data and plots are saved in the `results/` directory by default (or the path specified by `--outdir`).

*   `results_<algo_name>.csv`: Contains the average returns for the 10-episode evaluation phase after training.
*   `fairness_metrics_<algo_name>.csv`: Contains the Gini coefficient and Max-Min ratio for the 10-episode evaluation phase.
*   `training_fairness_metrics_<algo_name>.csv`: **NEW!** Logs the Gini coefficient and Max-Min ratio for *every episode* during the training process. This is crucial for observing the evolution of fairness over time.
*   `ppo_only_test_returns.csv`: Generated by the legacy `run_experiment` function (if called directly).
*   `CHMARL_Refined_Fairness.png`: The plot generated by the `--convergence` experiment.

## 8. Acknowledgements

This project builds upon foundational concepts in multi-agent reinforcement learning and environmental simulation. Specific inspirations and components are noted within the code comments.

---
