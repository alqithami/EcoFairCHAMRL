#!/usr/bin/env python3
"""Run EcoFair-CH-MARL and publish real evaluation output to the CH-MARL DataV portal.

This script does not create demo data. It runs the existing training/evaluation code and
POSTs the resulting episode summary to the portal backend endpoint:

    POST /api/chmarl/ingest

Example:
    python run_portal_training.py --portal_url http://localhost:8787 --episodes 200 --fairness --emission_cap
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, List
from urllib import request

import numpy as np

from EcoFairCHMARL import train_agent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def scenario_id(args: argparse.Namespace) -> str:
    if args.emission_cap and args.fairness:
        return "emissions-fairness"
    if args.emission_cap:
        return "emissions-aware"
    if args.fairness:
        return "fairness-aware"
    return "baseline"


def build_episode_payload(args: argparse.Namespace, returns: List[float], ginis: List[float], ratios: List[float]) -> Dict[str, Any]:
    avg_return = float(np.mean(returns)) if returns else 0.0
    avg_gini = float(np.mean(ginis)) if ginis else 0.0
    avg_ratio = float(np.mean(ratios)) if ratios else 0.0
    fairness_score = max(0.0, min(1.0, avg_ratio))
    reward_index = avg_return / max(1.0, abs(avg_return)) if avg_return != 0 else 0.0
    timestamp = utc_now()
    scenario = scenario_id(args)
    experiment_id = args.experiment_id or f"ecofair-{scenario}-{timestamp}"

    return {
        "experimentId": experiment_id,
        "scenarioId": scenario,
        "steps": [
            {
                "experimentId": experiment_id,
                "scenarioId": scenario,
                "episode": int(args.episodes),
                "step": 0,
                "timestamp": timestamp,
                "state": {
                    "num_ports": int(args.num_ports),
                    "num_vessels": int(args.num_vessels),
                    "algorithm": args.algo.upper(),
                    "evaluation_episodes": len(returns),
                },
                "actions": [
                    {
                        "agentId": "coordinator",
                        "agentType": "fleet",
                        "actionType": "policy_mode",
                        "actionValue": scenario,
                    }
                ],
                "rewards": [
                    {"agentId": "coordinator", "component": "global", "value": reward_index},
                    {"agentId": "coordinator", "component": "fairness", "value": fairness_score},
                ],
                "constraints": [
                    {
                        "constraintId": "fairness-gini",
                        "name": "Fairness Gini",
                        "value": avg_gini,
                        "limit": args.gini_limit,
                        "satisfied": avg_gini <= args.gini_limit,
                        "severity": "high" if avg_gini > args.gini_limit else "low",
                    },
                    {
                        "constraintId": "minmax-ratio",
                        "name": "Max-min ratio",
                        "value": avg_ratio,
                        "limit": args.minmax_limit,
                        "satisfied": avg_ratio >= args.minmax_limit,
                        "severity": "high" if avg_ratio < args.minmax_limit else "low",
                    },
                ],
                "fairness": [
                    {"metricId": "gini", "name": "Gini coefficient", "value": avg_gini, "groupBy": "vessel"},
                    {"metricId": "max-min-ratio", "name": "Max-min ratio", "value": avg_ratio, "groupBy": "vessel"},
                ],
                "hierarchyDecisions": [
                    {
                        "level": "coordinator",
                        "decisionId": f"{experiment_id}-summary",
                        "decisionLabel": f"{args.algo.upper()} evaluation completed",
                        "rationale": "Published directly from EcoFair-CH-MARL training/evaluation output.",
                    }
                ],
            }
        ],
    }


def post_to_portal(portal_url: str, payload: Dict[str, Any], token: str | None) -> None:
    endpoint = portal_url.rstrip("/") + "/api/chmarl/ingest"
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = request.Request(endpoint, data=body, headers=headers, method="POST")
    with request.urlopen(req, timeout=20) as response:
        print(response.read().decode("utf-8"))


def build_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run EcoFair-CH-MARL and publish results to CH-MARL DataV")
    parser.add_argument("--portal_url", default="http://localhost:8787")
    parser.add_argument("--portal_token", default=None)
    parser.add_argument("--experiment_id", default=None)
    parser.add_argument("--algo", default="PPO", choices=["PPO", "QMIX", "MAPPO", "SOTO", "FEN"])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--emission_cap", action="store_true")
    parser.add_argument("--fairness", action="store_true")
    parser.add_argument("--lambda_fair", type=float, default=10.0)
    parser.add_argument("--num_ports", type=int, default=8)
    parser.add_argument("--num_vessels", type=int, default=20)
    parser.add_argument("--outdir", default="results/")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--hl_update_interval", type=int, default=5)
    parser.add_argument("--gini_limit", type=float, default=0.25)
    parser.add_argument("--minmax_limit", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = build_cli()
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
        hl_update_interval=args.hl_update_interval,
    )
    payload = build_episode_payload(args, returns, ginis, ratios)
    post_to_portal(args.portal_url, payload, args.portal_token)


if __name__ == "__main__":
    main()
