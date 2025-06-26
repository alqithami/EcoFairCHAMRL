#!/usr/bin/env python3
"""
EcoFairCHMARL.py  —  26 June 2025
──────────────────────────────────────────────
Maritime hierarchical multi‑agent RL environment with:
      • emission cap
      • fairness shaping (Gini / max‑min)
      • algorithm‑agnostic runner (PPO default; SOTO / FEN wrappers;
        QMIX & MAPPO placeholders until MARL‑SB3 appears on PyPI)

The file is a strict superset of the original plus.
"""

# --------------------------------------------------------------------------- #
# 0  Imports                                                                   #
# --------------------------------------------------------------------------- #
import argparse, os, math, random, sys, pkgutil, importlib
from typing import List, Dict, Tuple

# Gym / Gymnasium compatibility ---------------------------------------------
try:
    import gymnasium as gym
    GYM_NEW_API = hasattr(gym.Env, "step") and gym.__version__ >= "0.26"
except ImportError:                            # classic gym fallback
    import gym                                 # type: ignore
    GYM_NEW_API = False

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

# -------- optional QMIX / MAPPO (robust finder) -----------------------------
QMIX = MAPPO = None
try:
    import sb3_contrib
    for modinfo in pkgutil.walk_packages(sb3_contrib.__path__,
                                         sb3_contrib.__name__ + "."):
        try:
            m = importlib.import_module(modinfo.name)
        except Exception:
            continue
        if QMIX is None and hasattr(m, "QMIX"):
            QMIX = getattr(m, "QMIX")
        if MAPPO is None and hasattr(m, "MAPPO"):
            MAPPO = getattr(m, "MAPPO")
except ImportError:
    pass  # sb3_contrib not installed – placeholders will be used

# --------------------------------------------------------------------------- #
# 0‑A  Fairness helpers                                                       #
# --------------------------------------------------------------------------- #
def compute_gini(vals: List[float]) -> float:
    x = np.asarray(vals, dtype=np.float64)
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    # Avoid division by zero if sum is zero
    if x.sum() == 0:
        return 0.0
    return (2 * (np.arange(1, n + 1) * x).sum()) / (n * x.sum()) - (n + 1) / n


def compute_minmax_ratio(vals: List[float]) -> float:
    x = np.asarray(vals, dtype=np.float64)
    if np.allclose(x, 0):
        return 1.0
    # Avoid division by zero if max is zero
    if x.max() == 0:
        return 1.0
    return float(x.min() / x.max())

# --------------------------------------------------------------------------- #
# 1  Synthetic‑data generator                                                 #
# --------------------------------------------------------------------------- #
def generate_synthetic_data(num_ports=16, num_vessels=50):
    ports = [{"name": f"Port_{i}", "capacity": np.random.randint(2, 6)}
             for i in range(num_ports)]

    dists = np.zeros((num_ports, num_ports))
    for i in range(num_ports):
        for j in range(i + 1, num_ports):
            d = np.random.randint(100, 601)
            dists[i, j] = dists[j, i] = d

    specs = [{"id": f"Vessel_{v}",
              "type": "Cargo",
              "max_speed": np.random.randint(14, 21),              # kt
              "fuel_curve_factor": np.random.uniform(5e-4, 1e-3)}  # kg/(kt³·nm)
             for v in range(num_vessels)]
    return ports, dists, specs

# --------------------------------------------------------------------------- #
# 2  Environment (idle / queue fuel now spec‑dependent)                       #
# --------------------------------------------------------------------------- #
class MultiAgentMaritimeEnv(gym.Env):
    metadata = {"render.modes": []}

    # idle/queue load factors (fraction of full thrust) ‑‑ can be tuned
    IDLE_LOAD  = 0.25
    QUEUE_LOAD = 0.10

    def __init__(self,
                 ports, distances, vessel_specs,
                 emission_cap_enabled=False, fairness_enabled=False,
                 partial_obs_enabled=False, weather_enabled=True,
                 max_steps=100, emission_cap_value=1000.0,
                 gamma_emis=10.0, gamma_fair=5.0, gamma_queue=0.2,
                 storm_prob=0.1, storm_speed_penalty=0.8,
                 storm_fuel_factor=1.2,
                 debug_mode=False,
                 hl_update_interval=5): # == NEW: High-level update interval ==
        super().__init__()
        self.ports, self.distances, self.vessel_specs = ports, distances, vessel_specs
        self.num_ports, self.num_vessels = len(ports), len(vessel_specs)
        self.max_steps = max_steps

        self.emission_cap_enabled = emission_cap_enabled
        self.fairness_enabled = fairness_enabled
        self.partial_obs_enabled = partial_obs_enabled
        self.weather_enabled = weather_enabled
        self.debug_mode = debug_mode
        self.hl_update_interval = hl_update_interval # == NEW ==

        self.emission_cap_value = emission_cap_value
        self.gamma_emis = gamma_emis
        self.gamma_fair = gamma_fair
        self.gamma_queue = gamma_queue

        self.storm_prob = storm_prob
        self.storm_speed_penalty = storm_speed_penalty
        self.storm_fuel_factor = storm_fuel_factor

        self.speed_levels = [0.5, 0.75, 1.0]
        # Action space: Each vessel chooses an action from 0 to action_size_per_vessel - 1
        # This single action encodes both a high-level (destination) and low-level (speed) decision.
        #   potential_hl_directive = a // len(self.speed_levels)
        #   speed_idx_low_level = a % len(self.speed_levels)
        self.action_size_per_vessel = self.num_ports * len(self.speed_levels)
        self.action_space = gym.spaces.Box(0,
                                           self.action_size_per_vessel - 1,
                                           (self.num_vessels,),
                                           dtype=np.float32)
        
        # Observation space:
        # For each vessel:
        # [status_code (0=IDLE, 1=QUEUING, 2=TRAVELING),
        #  current_port (actual port if IDLE/QUEUING, -1 if TRAVELING),
        #  high_level_directive (assigned destination port from high-level agent),
        #  chosen_speed_factor (from low-level agent's action),
        #  total_fuel_used_episode,
        #  queue_time,
        #  remaining_distance (if TRAVELING, else 0),
        #  current_speed (if TRAVELING, else 0)]
        # Plus global state variables: [global_emissions, steps_left]
        obs_dim = (8 * self.num_vessels) + 2 # Expanded for hierarchical observation
        self.observation_space = gym.spaces.Box(-1.0, 1e5, (obs_dim,), dtype=np.float32)

        self.current_episode = 0
        self.initial_gamma_fair = 1.0
        self.max_gamma_fair = 5.0
        self.total_episodes = 1000

        self.reset()
    
    def update_fairness_penalty(self):
        if self.total_episodes == 0:
            self.gamma_fair = self.initial_gamma_fair
        else:
            self.gamma_fair = self.initial_gamma_fair + \
                          (self.max_gamma_fair - self.initial_gamma_fair) * \
                          (self.current_episode / self.total_episodes)
        if self.debug_mode:
            print(f"[DEBUG] Episode {self.current_episode}: Fairness penalty (gamma_fair) updated to: {self.gamma_fair:.3f}")

    # ------------------------------------------------------------------ #
    # reset                                                               #
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        
        self.current_episode += 1
        self.update_fairness_penalty()
        
        self.current_step = 0
        self.port_occupancy = np.zeros(self.num_ports, dtype=int)
        self.vessel_states = []
        self.global_emissions = 0.0

        for spec in self.vessel_specs:
            init_port = np.random.randint(0, self.num_ports)
            self.port_occupancy[init_port] += 1
            self.vessel_states.append({
                "status": "IDLE", # "IDLE", "QUEUING", "TRAVELING"
                "current_port": init_port, # -1 if traveling
                "high_level_directive": init_port, # The destination assigned by high-level agent
                "chosen_speed_factor": self.speed_levels[-1], # Default to max speed for LL action
                "destination_port": -1, # Actual destination if TRAVELING (might differ from directive if queuing)
                "remaining_distance": 0.0, # Only relevant if TRAVELING
                "current_speed": 0.0, # Only relevant if TRAVELING
                "fuel_used_this_trip": 0.0, # For current travel/queue/idle segment
                "total_fuel_used_episode": 0.0, # Cumulative for fairness
                "queue_time": 0.0 # Cumulative for current queue segment
            })
        
        self.episode_fuel_usage = [vs["total_fuel_used_episode"] for vs in self.vessel_states]
        obs = self._get_obs()
        return (obs, {}) if GYM_NEW_API else obs
        
    # ------------------------------------------------------------------ #
    # observation helper                                                 #
    # ------------------------------------------------------------------ #
    def _get_obs(self):
        out = []
        for vs in self.vessel_states:
            # Map status to a numerical code
            status_code = 0 # IDLE
            if vs["status"] == "QUEUING":
                status_code = 1
            elif vs["status"] == "TRAVELING":
                status_code = 2

            # Port observation: actual port if IDLE/QUEUING, -1 if TRAVELING
            port_obs = float(vs["current_port"]) if vs["status"] != "TRAVELING" else -1.0
            
            # Remaining distance and speed are only relevant for TRAVELING
            remaining_dist_obs = float(vs["remaining_distance"]) if vs["status"] == "TRAVELING" else 0.0
            current_speed_obs = float(vs["current_speed"]) if vs["status"] == "TRAVELING" else 0.0

            out.extend([
                float(status_code),
                port_obs,
                float(vs["high_level_directive"]),
                float(vs["chosen_speed_factor"]),
                float(vs["total_fuel_used_episode"]),
                float(vs["queue_time"]),
                remaining_dist_obs,
                current_speed_obs
            ])
        
        # Add global state variables
        out.extend([float(self.global_emissions), float(self.max_steps - self.current_step)])
        
        return np.asarray(out, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # step                                                               #
    # ------------------------------------------------------------------ #
    def step(self, action):
        if not isinstance(action, np.ndarray):
            action = np.asarray(action)
        if self.debug_mode:
            print(f"\n[DEBUG] --- Step {self.current_step} ---")
            print(f"[DEBUG] Actions received by environment: {action}")
        
        step_fuels = [0.0] * self.num_vessels # Fuel consumed by each vessel in this step

        # Process each vessel's state and action
        for i in range(self.num_vessels):
            vessel_spec = self.vessel_specs[i]
            vessel_state = self.vessel_states[i]
            full_thrust_fuel = vessel_spec["fuel_curve_factor"] * (vessel_spec["max_speed"] ** 3)

            # Interpret the single action as both high-level (destination) and low-level (speed)
            a = int(round(action[i]))
            a = np.clip(a, 0, self.action_size_per_vessel - 1)
            
            speed_idx_low_level = a % len(self.speed_levels)
            potential_hl_directive = a // len(self.speed_levels)

            # == NEW: High-level directive update logic ==
            # High-level directive only updates at specific intervals
            if self.current_step % self.hl_update_interval == 0:
                vessel_state["high_level_directive"] = potential_hl_directive
            
            # Low-level speed factor updates at every step
            vessel_state["chosen_speed_factor"] = self.speed_levels[speed_idx_low_level]
            chosen_speed = vessel_spec["max_speed"] * vessel_state["chosen_speed_factor"]

            if self.debug_mode:
                print(f"[DEBUG] Vessel {i} (Initial State): Status={vessel_state['status']}, Current Port={vessel_state['current_port']}, Directive={vessel_state['high_level_directive']}, Chosen Speed Factor={vessel_state['chosen_speed_factor']:.1f}, Rem Dist={vessel_state['remaining_distance']:.1f}, Queue Time={vessel_state['queue_time']:.1f}")

            # Handle vessels based on their current status
            if vessel_state["status"] == "IDLE":
                # If high-level directive is to stay at current port
                if vessel_state["high_level_directive"] == vessel_state["current_port"]:
                    fuel_used_this_step = full_thrust_fuel * self.IDLE_LOAD
                    vessel_state["fuel_used_this_trip"] = fuel_used_this_step # Reset trip fuel
                    if self.debug_mode:
                        print(f"[DEBUG] Vessel {i}: High-level directive is to stay IDLE at port {vessel_state['current_port']}. Fuel used: {fuel_used_this_step:.3f}")
                else:
                    # High-level directive is to move, attempt to travel
                    target_destination = vessel_state["high_level_directive"]
                    if self.port_occupancy[target_destination] >= self.ports[target_destination]["capacity"]:
                        # Destination port is full, enter queuing state at current port
                        vessel_state["status"] = "QUEUING"
                        vessel_state["queue_time"] += 1.0
                        fuel_used_this_step = full_thrust_fuel * self.QUEUE_LOAD
                        vessel_state["fuel_used_this_trip"] = fuel_used_this_step # Reset trip fuel
                        if self.debug_mode:
                            print(f"[DEBUG] Vessel {i}: Attempted to move to port {target_destination}, but it's full. Now QUEUING at port {vessel_state['current_port']}. Fuel used: {fuel_used_this_step:.3f}")
                    else:
                        # Start traveling
                        dist_to_travel = self.distances[vessel_state["current_port"], target_destination]
                        
                        # Decrease occupancy at current port as vessel leaves
                        self.port_occupancy[vessel_state["current_port"]] -= 1

                        vessel_state["status"] = "TRAVELING"
                        vessel_state["destination_port"] = target_destination
                        vessel_state["remaining_distance"] = dist_to_travel
                        vessel_state["current_speed"] = chosen_speed
                        vessel_state["fuel_used_this_trip"] = 0.0 # Reset trip fuel
                        vessel_state["queue_time"] = 0.0 # Reset queue time

                        # Calculate fuel for this step of travel
                        effective_speed = chosen_speed
                        fuel_factor_modifier = 1.0
                        if self.weather_enabled and np.random.rand() < self.storm_prob:
                            effective_speed *= self.storm_speed_penalty
                            fuel_factor_modifier *= self.storm_fuel_factor
                            if self.debug_mode:
                                print(f"[DEBUG] Vessel {i}: Storm encountered! Effective speed reduced to {effective_speed:.1f}, fuel factor increased to {fuel_factor_modifier:.1f}")

                        distance_covered_this_step = effective_speed * 1.0 # Assuming 1 time unit per step
                        fuel_used_this_step = full_thrust_fuel * (vessel_state["chosen_speed_factor"] ** 3) * fuel_factor_modifier
                        
                        vessel_state["remaining_distance"] -= distance_covered_this_step
                        vessel_state["fuel_used_this_trip"] += fuel_used_this_step
                        if self.debug_mode:
                            print(f"[DEBUG] Vessel {i}: Started TRAVELING to port {target_destination} at speed {chosen_speed:.1f}. Covered {distance_covered_this_step:.1f} dist. Rem dist: {vessel_state['remaining_distance']:.1f}. Fuel used: {fuel_used_this_step:.3f}")

            elif vessel_state["status"] == "QUEUING":
                # Check if the port is no longer full
                if self.port_occupancy[vessel_state["current_port"]] < self.ports[vessel_state["current_port"]]["capacity"]:
                    # Port has capacity, vessel can now become IDLE
                    vessel_state["status"] = "IDLE"
                    vessel_state["queue_time"] = 0.0 # Reset queue time
                    fuel_used_this_step = full_thrust_fuel * self.IDLE_LOAD # Consume idle fuel for this step
                    if self.debug_mode:
                        print(f"[DEBUG] Vessel {i}: Port {vessel_state['current_port']} now has capacity. Vessel is now IDLE. Fuel used: {fuel_used_this_step:.3f}")
                else:
                    # Continue queuing
                    vessel_state["queue_time"] += 1.0
                    fuel_used_this_step = full_thrust_fuel * self.QUEUE_LOAD
                    if self.debug_mode:
                        print(f"[DEBUG] Vessel {i}: Still QUEUING at port {vessel_state['current_port']}. Queue time: {vessel_state['queue_time']:.1f}. Fuel used: {fuel_used_this_step:.3f}")

            elif vessel_state["status"] == "TRAVELING":
                # Continue traveling
                effective_speed = chosen_speed # Use the speed chosen by the low-level agent for this step
                fuel_factor_modifier = 1.0
                if self.weather_enabled and np.random.rand() < self.storm_prob:
                    effective_speed *= self.storm_speed_penalty
                    fuel_factor_modifier *= self.storm_fuel_factor
                    if self.debug_mode:
                        print(f"[DEBUG] Vessel {i}: Storm encountered! Effective speed reduced to {effective_speed:.1f}, fuel factor increased to {fuel_factor_modifier:.1f}")

                distance_covered_this_step = effective_speed * 1.0
                fuel_used_this_step = full_thrust_fuel * (vessel_state["chosen_speed_factor"] ** 3) * fuel_factor_modifier
                
                vessel_state["remaining_distance"] -= distance_covered_this_step
                vessel_state["fuel_used_this_trip"] += fuel_used_this_step

                if vessel_state["remaining_distance"] <= 0:
                    # Vessel has arrived at destination
                    if self.port_occupancy[vessel_state["destination_port"]] >= self.ports[vessel_state["destination_port"]]["capacity"]:
                        # Destination port is full upon arrival, enter queuing state at destination
                        vessel_state["status"] = "QUEUING"
                        vessel_state["current_port"] = vessel_state["destination_port"] # Now at destination, but queuing
                        vessel_state["queue_time"] += 1.0
                        # Fuel for this step already accounted for, but now consuming queue fuel for future steps
                        if self.debug_mode:
                            print(f"[DEBUG] Vessel {i}: Arrived at port {vessel_state['destination_port']}, but it's full. Now QUEUING there. Fuel used this step: {fuel_used_this_step:.3f}")
                    else:
                        # Arrived and port has capacity, enter idle state
                        vessel_state["status"] = "IDLE"
                        self.port_occupancy[vessel_state["destination_port"]] += 1 # Increase occupancy at destination
                        vessel_state["current_port"] = vessel_state["destination_port"]
                        vessel_state["queue_time"] = 0.0
                        if self.debug_mode:
                            print(f"[DEBUG] Vessel {i}: Arrived at port {vessel_state['destination_port']} and docked. Now IDLE. Fuel used this step: {fuel_used_this_step:.3f}")
                    
                    # Reset travel-specific states
                    vessel_state["destination_port"] = -1
                    vessel_state["remaining_distance"] = 0.0
                    vessel_state["current_speed"] = 0.0
                    vessel_state["fuel_used_this_trip"] = 0.0 # Reset for next trip
                else:
                    # Still traveling
                    if self.debug_mode:
                        print(f"[DEBUG] Vessel {i}: Still TRAVELING. Covered {distance_covered_this_step:.1f} dist. Rem dist: {vessel_state['remaining_distance']:.1f}. Fuel used this step: {fuel_used_this_step:.3f}")
            
            # Add fuel used this step to total episode fuel for the vessel
            vessel_state["total_fuel_used_episode"] += fuel_used_this_step
            step_fuels[i] = fuel_used_this_step

        # ----- accounting -------------------------------------------------
        self.global_emissions += sum(step_fuels)
        self.episode_fuel_usage = [vs["total_fuel_used_episode"] for vs in self.vessel_states]

        reward = -sum(step_fuels)
        emission_penalty = 0.0
        fairness_penalty = 0.0

        if self.emission_cap_enabled and self.global_emissions > self.emission_cap_value:
            emission_penalty = self.gamma_emis * (self.global_emissions - self.emission_cap_value)
            reward -= emission_penalty
        
        current_gini = compute_gini(self.episode_fuel_usage)
        current_minmax_ratio = compute_minmax_ratio(self.episode_fuel_usage)

        if self.fairness_enabled:
            fairness_penalty = self.gamma_fair * current_gini
            reward -= fairness_penalty

        if self.debug_mode:
            print(f"[DEBUG] Step {self.current_step} Global Metrics:")
            print(f"  Port Occupancy: {self.port_occupancy}")
            print(f"  Total fuel used (episode): {[f'{f:.3f}' for f in self.episode_fuel_usage]}")
            print(f"  Gini: {current_gini:.3f}, Max-Min Ratio: {current_minmax_ratio:.3f}")
            print(f"  Reward components: Fuel Cost={-sum(step_fuels):.3f}, Emission Penalty={-emission_penalty:.3f}, Fairness Penalty={-fairness_penalty:.3f}")
            print(f"  Total Reward: {reward:.3f}")

        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated  = False
        obs = self._get_obs()
        info = {
            "gini": current_gini,
            "max_min_ratio": current_minmax_ratio,
            "episode_fuel_usage": self.episode_fuel_usage
        }

        if GYM_NEW_API:
            return obs, reward, terminated, truncated, info
        else:
            return obs, reward, terminated, info

    def render(self): pass

# ---------------------------------------------------------------- #
# 3  Reward‑wrappers for SOTO / FEN                                #
# ---------------------------------------------------------------- #
class _FairWrapper(gym.Wrapper):
    def __init__(self, env, lam=10.0):
        super().__init__(env)
        self.lam = lam

    def _penalty(self) -> float:
        raise NotImplementedError

    def step(self, act):
        res = self.env.step(act)
        if GYM_NEW_API:
            obs, rew, term, trunc, info = res
            penalty_val = self._penalty()
            rew -= self.lam * penalty_val
            if self.env.debug_mode:
                print(f"[DEBUG] Wrapper Penalty: {penalty_val:.3f}, Applied Reward Change: {-self.lam * penalty_val:.3f}")
            return obs, rew, term, trunc, info
        else:
            obs, rew, done, info = res
            penalty_val = self._penalty()
            rew -= self.lam * penalty_val
            if self.env.debug_mode:
                print(f"[DEBUG] Wrapper Penalty: {penalty_val:.3f}, Applied Reward Change: {-self.lam * penalty_val:.3f}")
            return obs, rew, done, info

class SOTORewardWrapper(_FairWrapper):
    def _penalty(self) -> float:
        return compute_gini(self.env.episode_fuel_usage)

class FENRewardWrapper(_FairWrapper):
    def _penalty(self) -> float:
        return 1.0 - compute_minmax_ratio(self.env.episode_fuel_usage)

# --------------------------------------------------------------------------- #
# 4  MetricVecEnv (wrapper‑safe)                                              #
# --------------------------------------------------------------------------- #
class MetricVecEnv(DummyVecEnv):
    def reset(self, **kw):
        obs = super().reset(**kw)
        self.last_fuel_usage = self.envs[0].unwrapped.episode_fuel_usage
        return obs

    def step(self, act):
        obs, rews, dones, infos = super().step(act)
        if dones[0]:
            self.last_fuel_usage = self.envs[0].unwrapped.episode_fuel_usage
        return obs, rews, dones, infos

# --------------------------------------------------------------------------- #
# 4-A  FairnessMetricsCallback for training logging                           #
# --------------------------------------------------------------------------- #
class FairnessMetricsCallback(BaseCallback):
    """
    A custom callback to log fairness metrics (Gini, Max-Min Ratio) during training.
    """
    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.training_ginis = []
        self.training_ratios = []
        self.episodes_trained = 0

    def _on_training_start(self) -> None:
        # Log hyperparameters to the logger
        self.logger.record("hparams/gamma_fair_initial", self.training_env.envs[0].unwrapped.initial_gamma_fair)
        self.logger.record("hparams/gamma_fair_max", self.training_env.envs[0].unwrapped.max_gamma_fair)
        self.logger.record("hparams/total_episodes_env", self.training_env.envs[0].unwrapped.total_episodes)
        self.logger.record("hparams/fairness_enabled", self.training_env.envs[0].unwrapped.fairness_enabled)
        self.logger.record("hparams/emission_cap_enabled", self.training_env.envs[0].unwrapped.emission_cap_enabled)
        
    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            self.episodes_trained += 1
            info = self.locals["infos"][0]
            
            current_gini = info.get("gini", compute_gini(info.get("episode_fuel_usage", [])))
            current_ratio = info.get("max_min_ratio", compute_minmax_ratio(info.get("episode_fuel_usage", [])))

            self.training_ginis.append(current_gini)
            self.training_ratios.append(current_ratio)

            self.logger.record("rollout/gini", current_gini)
            self.logger.record("rollout/max_min_ratio", current_ratio)
            self.logger.record("rollout/episode_num", self.episodes_trained)
            self.logger.dump(self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        df = pd.DataFrame({
            "episode": range(1, len(self.training_ginis) + 1),
            "gini": self.training_ginis,
            "max_min_ratio": self.training_ratios
        })
        df.to_csv(self.save_path, index=False)
        if self.verbose > 0:
            print(f"Training fairness metrics saved to {self.save_path}")

# --------------------------------------------------------------------------- #
# 5  Legacy PPO ablation (kept intact for back‑compat)                        #
# --------------------------------------------------------------------------- #
def run_experiment(emission_cap=False, fairness=False, partial_obs=False, weather=True,
                   num_ports=16, num_vessels=50, max_steps=50, total_episodes=1000,
                   outdir="results/", debug_mode=False,
                   hl_update_interval=5): # == NEW ==
    os.makedirs(outdir, exist_ok=True)
    ports, dists, specs = generate_synthetic_data(num_ports, num_vessels)

    env = DummyVecEnv([lambda: MultiAgentMaritimeEnv(
        ports, dists, specs,
        emission_cap_enabled=emission_cap,
        fairness_enabled=fairness,
        partial_obs_enabled=partial_obs,
        weather_enabled=weather,
        max_steps=max_steps,
        debug_mode=debug_mode,
        hl_update_interval=hl_update_interval)]) # == NEW ==

    model = PPO("MlpPolicy", env, verbose=1,
                n_steps=256, batch_size=64, learning_rate=3e-4)
    model.learn(total_timesteps=total_episodes * max_steps)

    # quick 10‑episode test
    rets = []
    for _ in range(10):
        obs = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, _ = env.step(act)
            ep_r += rew[0]
        rets.append(ep_r)

    pd.DataFrame({"episode": range(1, 11), "return": rets}) \
      .to_csv(os.path.join(outdir, "ppo_only_test_returns.csv"), index=False)
    return model

# --------------------------------------------------------------------------- #
# 6  Unified training / evaluation                                            #
# --------------------------------------------------------------------------- #
def _env_factory(algo, emission_cap, fairness, lam, n_ports, n_vessels, total_episodes, debug_mode, hl_update_interval): # == NEW ==
    def _make():
        env = MultiAgentMaritimeEnv(*generate_synthetic_data(n_ports, n_vessels),
                                    emission_cap_enabled=emission_cap,
                                    fairness_enabled=fairness if algo not in {"SOTO", "FEN"} else False,
                                    partial_obs_enabled=False,
                                    weather_enabled=True,
                                    max_steps=50,
                                    debug_mode=debug_mode,
                                    hl_update_interval=hl_update_interval) # == NEW ==
        env.total_episodes = total_episodes
        if algo == "SOTO":
            env = SOTORewardWrapper(env, lam)
        elif algo == "FEN":
            env = FENRewardWrapper(env, lam)
        return env
    return _make

def train_agent(algo="PPO", episodes=1000, emission_cap=False, fairness=False,
                num_ports=16, num_vessels=50, outdir="results/", lam=10.0,
                debug_mode=False, hl_update_interval=5): # == NEW ==
    os.makedirs(outdir, exist_ok=True)
    
    if algo == "QMIX" and QMIX is None:
        print(f"[WARN] QMIX is not installed or available. Falling back to PPO.")
        algo = "PPO"
    elif algo == "MAPPO" and MAPPO is None:
        print(f"[WARN] MAPPO is not installed or available. Falling back to PPO.")
        algo = "PPO"

    vec_env = MetricVecEnv([_env_factory(algo, emission_cap, fairness, lam,
                                         num_ports, num_vessels, episodes, debug_mode, hl_update_interval)]) # == NEW ==

    if algo in {"PPO", "SOTO", "FEN"}:
        Model = PPO
        mkw = dict(policy="MlpPolicy", verbose=0,
                   n_steps=256, batch_size=64, learning_rate=3e-4)
    elif algo == "QMIX" and QMIX is not None:
        Model, mkw = QMIX, dict(policy="MlpPolicy", verbose=0)
    elif algo == "MAPPO" and MAPPO is not None:
        Model, mkw = MAPPO, dict(policy="MlpPolicy", verbose=0)
    else:
        Model = PPO
        mkw = dict(policy="MlpPolicy", verbose=0,
                   n_steps=256, batch_size=64, learning_rate=3e-4)

    model = Model(env=vec_env, **mkw)
    
    training_metrics_save_path = os.path.join(outdir, f"training_fairness_metrics_{algo.lower()}.csv")
    callback = FairnessMetricsCallback(save_path=training_metrics_save_path, verbose=1 if debug_mode else 0)

    model.learn(total_timesteps=episodes * 50, callback=callback)

    returns, ginis, ratios = [], [], []
    for _ in range(10):
        obs = vec_env.reset()
        done = False
        ep_r = 0.0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, info = vec_env.step(act)
            ep_r += rew[0]
        returns.append(ep_r)
        # Ensure info is correctly extracted from VecEnv output
        # info is a list of dicts, one for each env in the VecEnv. We only have one.
        env_info = info[0]
        ginis.append(env_info.get("gini", compute_gini(env_info.get("episode_fuel_usage", []))))
        ratios.append(env_info.get("max_min_ratio", compute_minmax_ratio(env_info.get("episode_fuel_usage", []))))

    pd.DataFrame({"episode": range(1, 11), "return": returns}) \
      .to_csv(os.path.join(outdir, f"results_{algo.lower()}.csv"), index=False)
    pd.DataFrame({"episode": range(1, 11),
                  "gini": ginis, "max_min_ratio": ratios}) \
      .to_csv(os.path.join(outdir, f"fairness_metrics_{algo.lower()}.csv"), index=False)

    return returns, ginis, ratios

# --------------------------------------------------------------------------- #
# 7  Convergence‑approximation playground (from convergence_appx.py)          #
# --------------------------------------------------------------------------- #
def run_convergence_approx(episodes=3000, outdir="results/"):
    print("[INFO] Running convergence approximation toy model …")
    resource_cap = 100
    num_agents   = 5
    baseline_reward = 20
    fairness_threshold = 0.8
    high_lr, low_lr = 5e-4, 1e-2
    init_w, max_w   = 1.0, 5.0
    smooth_alpha    = 0.9
    smooth_window   = 50

    hi_policy = np.random.rand(10) * 10
    lo_policies = [np.random.rand(10) for _ in range(num_agents)]
    smooth_alloc = 0

    hi_allocs, tot_use, fairness_vals, lo_rews, std_devs = [], [], [], [], []

    for ep in range(episodes):
        w = init_w + (max_w - init_w) * ep / episodes
        raw_alloc = hi_policy[np.random.randint(0, 10)] * 10
        total_alloc = smooth_alpha * smooth_alloc + (1 - smooth_alpha) * raw_alloc
        total_alloc = min(total_alloc, resource_cap)
        smooth_alloc = total_alloc
        hi_allocs.append(total_alloc)

        allocations = [pol[np.random.randint(0,10)] * (total_alloc/num_agents)
                       for pol in lo_policies]
        use = sum(allocations)
        penalty = 0.5 * max(0, use - resource_cap)
        agt_rewards = [baseline_reward - a - penalty for a in allocations]
        std = np.std(agt_rewards)
        fairness_vals.append(max(0, min(agt_rewards) / (max(agt_rewards) or 1)))
        std_devs.append(std)
        agt_rewards = [r - w*std  for r in agt_rewards]

        lo_rews.append(agt_rewards)
        hi_policy += high_lr * (np.random.rand(10) - 0.5)
        for i in range(num_agents):
            lo_policies[i] += low_lr * (np.random.rand(10) - 0.5)

    # plotting --------------------------------------------------------------
    lo_rews = np.array(lo_rews)
    fair_smoothed = np.convolve(fairness_vals,
                                np.ones(smooth_window)/smooth_window, mode='valid')
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(12,12))
    plt.subplot(5,1,1); plt.plot(hi_allocs); plt.axhline(resource_cap,c='r',ls='--')
    plt.title("High‑Level Allocations"); plt.subplot(5,1,2)
    plt.plot(tot_use := np.sum(lo_rews, axis=1)); plt.axhline(resource_cap,c='r',ls='--')
    plt.title("Total Resource Usage"); plt.subplot(5,1,3)
    plt.plot(np.mean(lo_rews, axis=1)); plt.title("Mean Low‑Level Reward")
    plt.subplot(5,1,4); plt.plot(fair_smoothed); plt.axhline(fairness_threshold,c='r',ls='--')
    plt.title("Smoothed Fairness Metric"); plt.subplot(5,1,5)
    plt.plot(std_devs); plt.title("Reward Std Dev (fairness proxy)")
    plt.tight_layout()
    fig_path = os.path.join(outdir, "CHMARL_Refined_Fairness.png")
    plt.savefig(fig_path, dpi=300)
    print(f"[INFO] Convergence figure saved to {fig_path}")

# --------------------------------------------------------------------------- #
# 8  CLI                                                                      #
# --------------------------------------------------------------------------- #
def build_cli():
    cli = argparse.ArgumentParser("Maritime MARL + fairness")
    cli.add_argument("--algo", default="PPO",
                     choices=["PPO","QMIX","MAPPO","SOTO","FEN"],
                     help="Learning algorithm / baseline")
    cli.add_argument("--episodes", type=int, default=1000)
    cli.add_argument("--emission_cap", action="store_true")
    cli.add_argument("--fairness", action="store_true")
    cli.add_argument("--lambda_fair", type=float, default=10.0)
    cli.add_argument("--num_ports", type=int, default=8)
    cli.add_argument("--num_vessels", type=int, default=20)
    cli.add_argument("--outdir", default="results/")
    cli.add_argument("--convergence", action="store_true",
                     help="Run convergence approximation demo instead")
    cli.add_argument("--debug", action="store_true",
                     help="Enable verbose debugging output")
    cli.add_argument("--hl_update_interval", type=int, default=5,
                     help="High-level directive update interval (steps)") # == NEW ==
    return cli.parse_args()

# --------------------------------------------------------------------------- #
# 9  Main                                                                     #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    args = build_cli()

    if args.convergence:
        run_convergence_approx(outdir=args.outdir)
        sys.exit(0)

    returns, ginis, ratios = train_agent(
        algo=args.algo.upper(),
        episodes=args.episodes,
        emission_cap=args.emission_cap,
        fairness=args.fairness,
        num_ports=args.num_ports,
        num_vessels=args.num_vessels,
        outdir=args.outdir,
        lam=args.lambda_fair,
        debug_mode=args.debug,
        hl_update_interval=args.hl_update_interval) # == NEW ==

    print(f"\n=== {args.algo.upper()}  (10‑episode eval) ===")
    print(f"Average return        : {np.mean(returns): .3f}")
    print(f"Average Gini          : {np.mean(ginis): .3f}")
    print(f"Average max‑min ratio : {np.mean(ratios): .3f}")
    print("CSV files written to  :", os.path.abspath(args.outdir))

