from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
PPO_ROOT = ROOT / "ppo_rule_pool"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ptcg_rl import config as base_config
from ptcg_rl.env import BattleRunner, DecisionSample, active_player, terminal_reward
from ptcg_rl.features import HistoryTracker, action_to_multihot
from ptcg_rl.model import PTCGPolicyNet, action_logprob_for_action, batch_encoded, select_action_from_logits
from ptcg_rl.train_utils import chunks, load_checkpoint, save_checkpoint, teacher_bc_loss
from ptcg_rl.utils import action_is_valid, set_seed


CONFIG_PATH = Path(os.environ.get("PTCG_RULE_POOL_PPO_CONFIG", str(PPO_ROOT / "configs" / "rule_pool_ppo.json")))
RUN_ROOT: Path | None = None
CHECKPOINT_DIR: Path | None = None
LOG_DIR: Path | None = None
EVAL_DIR: Path | None = None
LATEST_PATH: Path | None = None
FINAL_PATH: Path | None = None
METRICS_PATH: Path | None = None
MANIFEST_PATH: Path | None = None


BEST_METRICS = [
    "score",
    "total_wr",
    "worst_wr",
    "original_archaludon_wr",
    "official_iono_wr",
]
OPPONENT_KEYS = [
    "original_archaludon",
    "original_manual_edit",
    "archaludon_v4",
    "official_dragapult_ex",
    "official_iono",
    "official_mega_abomasnow_ex",
    "official_mega_lucario_ex",
    "community_lucario",
    "community_alakazam",
]


@dataclass
class RunnerSpec:
    key: str
    name: str
    runner: BattleRunner
    train_games: int
    eval_games: int


def load_run_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8-sig") as f:
        cfg = json.load(f)
    return cfg


def choose_device() -> torch.device:
    if base_config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_run_paths(run_name: str) -> None:
    global RUN_ROOT, CHECKPOINT_DIR, LOG_DIR, EVAL_DIR, LATEST_PATH, FINAL_PATH, METRICS_PATH, MANIFEST_PATH
    RUN_ROOT = PPO_ROOT / "runs" / run_name
    CHECKPOINT_DIR = RUN_ROOT / "checkpoints"
    LOG_DIR = RUN_ROOT / "logs"
    EVAL_DIR = RUN_ROOT / "evals"
    LATEST_PATH = CHECKPOINT_DIR / "ppo_rules_latest.pt"
    FINAL_PATH = CHECKPOINT_DIR / "ppo_rules_final.pt"
    METRICS_PATH = LOG_DIR / "ppo_rules_metrics.csv"
    MANIFEST_PATH = RUN_ROOT / "manifest.json"


def _has_submission(path: Path) -> bool:
    return (path / "main.py").exists() and (path / "deck.csv").exists()


def project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def build_runners(cfg: dict[str, Any]) -> list[RunnerSpec]:
    out: list[RunnerSpec] = []
    eval_games_each = int(cfg.get("eval_games_each", 100))
    for row in cfg["opponents"]:
        key = row["key"]
        submission_dir = project_path(str(row["path"]))
        if not _has_submission(submission_dir):
            raise RuntimeError(f"opponent {key} is missing main.py/deck.csv: {submission_dir}")
        runner = BattleRunner(opponent_submission=submission_dir)
        runner.display_name = key
        out.append(
            RunnerSpec(
                key=key,
                name=str(row.get("name", key)),
                runner=runner,
                train_games=0,
                eval_games=int(row.get("eval_games", eval_games_each)),
            )
        )
    return out


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _damage(card: dict | None) -> float:
    if not isinstance(card, dict):
        return 0.0
    try:
        hp = float(card.get("hp") or 0.0)
        remaining = float(card.get("remainingHp") or 0.0)
    except Exception:
        hp = remaining = 0.0
    if hp > 0 and remaining > 0:
        return max(0.0, hp - remaining)
    try:
        return max(0.0, float(card.get("damage") or 0.0))
    except Exception:
        return 0.0


def _cards(value: Any) -> list[dict]:
    if isinstance(value, list):
        return [card for card in value if isinstance(card, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def _player(obs: dict, idx: int) -> dict:
    players = ((obs.get("current") or {}).get("players") or [])
    if 0 <= idx < len(players) and isinstance(players[idx], dict):
        return players[idx]
    return {}


def _prizes_remaining(obs: dict, idx: int) -> int:
    return len(_player(obs, idx).get("prize") or [])


def _board_damage(obs: dict, idx: int) -> float:
    player = _player(obs, idx)
    total = 0.0
    for area in ("active", "bench"):
        for card in _cards(player.get(area)):
            total += _damage(card)
    return total


def transition_reward(before: dict, after: dict, learner_seat: int, reward_cfg: dict[str, Any]) -> float:
    opp = 1 - learner_seat
    before_our_prizes = _prizes_remaining(before, learner_seat)
    after_our_prizes = _prizes_remaining(after, learner_seat)
    before_opp_prizes = _prizes_remaining(before, opp)
    after_opp_prizes = _prizes_remaining(after, opp)

    our_taken = max(0, before_our_prizes - after_our_prizes)
    opp_taken = max(0, before_opp_prizes - after_opp_prizes)
    reward = our_taken * float(reward_cfg["prize_taken"])
    reward += opp_taken * float(reward_cfg["prize_lost"])

    dealt = max(0.0, _board_damage(after, opp) - _board_damage(before, opp))
    taken = max(0.0, _board_damage(after, learner_seat) - _board_damage(before, learner_seat))
    cap = float(reward_cfg["damage_step_cap"])
    reward += min(cap, dealt * float(reward_cfg["damage_dealt_scale"]))
    reward += max(-cap, taken * float(reward_cfg["damage_taken_scale"]))

    step_clip = float(reward_cfg["step_clip"])
    return max(-step_clip, min(step_clip, reward))


def terminal_to_reward(obs: dict, learner_seat: int, reward_cfg: dict[str, Any]) -> float:
    raw = terminal_reward(obs, learner_seat)
    if raw > 0:
        return float(reward_cfg["win"])
    if raw < 0:
        return float(reward_cfg["loss"])
    return float(reward_cfg["draw"])


def value_output(values: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    return torch.tanh(values) if bool(cfg.get("value_tanh", True)) else values


def disable_stochastic_layers_for_ppo(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.MultiheadAttention)):
            module.eval()


def logprob_for_action(
    logits: torch.Tensor,
    mask: torch.Tensor,
    action: list[int],
    temperature: float,
    min_count: int,
    max_count: int,
) -> torch.Tensor:
    scaled = logits / max(1e-6, temperature)
    return action_logprob_for_action(scaled, mask, action, min_count, max_count)


@torch.no_grad()
def select_model_action(
    model: PTCGPolicyNet,
    encoded,
    device: torch.device,
    temperature: float,
    greedy: bool,
) -> tuple[list[int], float, float]:
    batch = batch_encoded([encoded], device)
    logits, value = model(batch)
    scaled_logits = logits[0] / max(1e-6, temperature)
    action, logp, _ = select_action_from_logits(
        scaled_logits,
        batch["option_mask"][0],
        encoded.min_count,
        encoded.max_count,
        greedy=greedy,
    )
    return action, float(logp.item()), float(value[0].item())


@torch.no_grad()
def collect_ppo_game(
    runner: BattleRunner,
    model: PTCGPolicyNet,
    device: torch.device,
    cfg: dict[str, Any],
    learner_seat: int,
    greedy: bool = False,
) -> tuple[list[DecisionSample], dict[str, Any]]:
    runner.reset_rule_modules()
    deck_a = runner.teacher.deck()
    deck_b = runner.opponent_deck()
    decks = [None, None]
    decks[learner_seat] = deck_a
    decks[1 - learner_seat] = deck_b
    obs, start = runner.cg_game.battle_start(decks[0], decks[1])
    if obs is None:
        return [], {
            "error": f"battle_start {start.errorPlayer}/{start.errorType}",
            "reward": 0.0,
            "decisions": 0,
            "episode_len": 0,
        }

    history = HistoryTracker()
    samples: list[DecisionSample] = []
    last_learner_sample: DecisionSample | None = None
    invalid_model_actions = 0
    reward_cfg = cfg["reward"]
    temperature = float(cfg["temperature"])
    try:
        for _ in range(int(base_config.MAX_STEPS_PER_GAME)):
            cur = obs.get("current") or {}
            if cur.get("result", -1) != -1:
                break
            pi = active_player(obs)
            if pi is None:
                break

            before = obs
            if pi == learner_seat:
                teacher_action = runner.teacher.act(obs)
                encoded = runner.encoder.encode(obs, history, runner.teacher.module)
                action, logp, value = select_model_action(model, encoded, device, temperature, greedy=greedy)
                if bool(cfg.get("value_tanh", True)):
                    value = math.tanh(value)
                if not action_is_valid(obs, action):
                    invalid_model_actions += 1
                    action = teacher_action
                    batch = batch_encoded([encoded], device)
                    logits, _ = model(batch)
                    logp = float(
                        logprob_for_action(
                            logits[0],
                            batch["option_mask"][0],
                            action,
                            temperature,
                            encoded.min_count,
                            encoded.max_count,
                        ).item()
                    )
                sample = DecisionSample(
                    encoded=encoded,
                    teacher_action=teacher_action,
                    action=action,
                    logprob=logp,
                    value=value,
                    reward=0.0,
                    done=False,
                    player_index=learner_seat,
                )
                samples.append(sample)
                last_learner_sample = sample
            else:
                action = runner.opponent_act(obs)

            if not action_is_valid(obs, action):
                return samples, {
                    "error": f"invalid action by {pi}",
                    "reward": -1.0,
                    "invalid_model_actions": invalid_model_actions,
                    "decisions": len(samples),
                    "episode_len": len(samples),
                }

            history.update(obs, pi, action)
            obs = runner.cg_game.battle_select(action)

            if last_learner_sample is not None:
                last_learner_sample.reward += transition_reward(before, obs, learner_seat, reward_cfg)

            result = (obs.get("current") or {}).get("result", -1)
            if result != -1:
                if last_learner_sample is not None:
                    last_learner_sample.reward += terminal_to_reward(obs, learner_seat, reward_cfg)
                    last_learner_sample.done = True
                break

        if samples and not samples[-1].done:
            samples[-1].done = True
        if invalid_model_actions and samples:
            samples[-1].reward += invalid_model_actions * float(reward_cfg["invalid_action"])
        final_reward = terminal_reward(obs, learner_seat)
        return samples, {
            "reward": final_reward,
            "result": (obs.get("current") or {}).get("result", -1),
            "invalid_model_actions": invalid_model_actions,
            "decisions": len(samples),
            "episode_len": len(samples),
        }
    finally:
        runner.cg_game.battle_finish()


def compute_returns_advantages(samples: list[DecisionSample], gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    rewards = torch.tensor([s.reward for s in samples], dtype=torch.float32)
    values = torch.tensor([s.value for s in samples] + [0.0], dtype=torch.float32)
    dones = torch.tensor([s.done for s in samples], dtype=torch.float32)
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(samples))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
    return returns, advantages


def masked_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(mask <= 0, -1e9)
    probs = torch.softmax(masked, dim=-1)
    log_probs = torch.log_softmax(masked, dim=-1)
    return -(probs * log_probs).masked_fill(mask <= 0, 0.0).sum(dim=-1).mean()


def masked_policy_kl(logits: torch.Tensor, base_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    current_logp = torch.log_softmax(logits.masked_fill(mask <= 0, -1e9), dim=-1)
    base_logp = torch.log_softmax(base_logits.masked_fill(mask <= 0, -1e9), dim=-1)
    current_p = torch.softmax(logits.masked_fill(mask <= 0, -1e9), dim=-1)
    return (current_p * (current_logp - base_logp)).masked_fill(mask <= 0, 0.0).sum(dim=-1).mean()


def grad_norm_by_group(model: PTCGPolicyNet) -> dict[str, float]:
    sums = {"backbone": 0.0, "policy": 0.0, "value": 0.0}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if name.startswith("value."):
            group = "value"
        elif name.startswith("policy."):
            group = "policy"
        else:
            group = "backbone"
        norm = float(param.grad.detach().float().norm(2).item())
        sums[group] += norm * norm
    return {key: math.sqrt(value) for key, value in sums.items()}


def ppo_update(
    model: PTCGPolicyNet,
    base_model: PTCGPolicyNet,
    optimizer: torch.optim.Optimizer,
    samples: list[DecisionSample],
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.train()
    disable_stochastic_layers_for_ppo(model)
    base_model.eval()
    returns, advantages = compute_returns_advantages(samples, float(cfg["gamma"]), float(cfg["gae_lambda"]))
    returns = returns.to(device)
    advantages = advantages.to(device)
    adv_mean = float(advantages.mean().item()) if advantages.numel() else 0.0
    adv_std = float(advantages.std(unbiased=False).item()) if advantages.numel() > 1 else 0.0
    adv_max = float(advantages.max().item()) if advantages.numel() else 0.0
    adv_min = float(advantages.min().item()) if advantages.numel() else 0.0
    indices = list(range(len(samples)))
    metrics: list[dict[str, float]] = []
    stopped_early = False

    for _ in range(int(cfg["ppo_epochs"])):
        random.shuffle(indices)
        for batch_indices in chunks(indices, int(cfg["batch_size"])):
            if not batch_indices:
                continue
            batch_samples = [samples[i] for i in batch_indices]
            batch_returns = returns[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch = batch_encoded([s.encoded for s in batch_samples], device)
            teacher_targets = torch.as_tensor(
                np.stack([action_to_multihot(s.teacher_action) for s in batch_samples], axis=0),
                dtype=torch.float32,
                device=device,
            )
            old_logp = torch.tensor([s.logprob for s in batch_samples], dtype=torch.float32, device=device)

            logits, values = model(batch)
            values_for_loss = value_output(values, cfg)
            train_logits = logits / max(1e-6, float(cfg["temperature"]))
            new_logp = torch.stack(
                [
                    action_logprob_for_action(
                        train_logits[row_idx],
                        batch["option_mask"][row_idx],
                        sample.action or [],
                        sample.encoded.min_count,
                        sample.encoded.max_count,
                    )
                    for row_idx, sample in enumerate(batch_samples)
                ]
            )
            ratio = torch.exp((new_logp - old_logp).clamp(-10.0, 10.0))
            unclipped = ratio * batch_advantages
            clipped = ratio.clamp(1.0 - float(cfg["clip_eps"]), 1.0 + float(cfg["clip_eps"])) * batch_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values_for_loss, batch_returns)
            entropy = masked_entropy(train_logits, batch["option_mask"])
            bc_loss = teacher_bc_loss(logits, batch["option_mask"], teacher_targets)
            with torch.no_grad():
                base_logits, _ = base_model(batch)
            base_kl = masked_policy_kl(logits, base_logits, batch["option_mask"])
            approx_kl = (old_logp - new_logp).mean()
            clip_fraction = ((ratio - 1.0).abs() > float(cfg["clip_eps"])).float().mean()
            ratio_std = ratio.std(unbiased=False) if ratio.numel() > 1 else ratio.sum() * 0.0
            returns_var = torch.var(batch_returns, unbiased=False)
            explained_variance = 1.0 - torch.var(
                batch_returns - values_for_loss.detach(),
                unbiased=False,
            ) / returns_var.clamp_min(1e-8)
            loss = (
                policy_loss
                + float(cfg["value_coef"]) * value_loss
                - float(cfg["entropy_coef"]) * entropy
                + float(cfg["bc_coef"]) * bc_loss
                + float(cfg["base_kl_coef"]) * base_kl
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norms = grad_norm_by_group(model)
            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["max_grad_norm"]))
            optimizer.step()

            row = {
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item()),
                "bc_loss": float(bc_loss.item()),
                "base_kl": float(base_kl.item()),
                "approx_kl": float(approx_kl.item()),
                "clip_fraction": float(clip_fraction.item()),
                "ratio_mean": float(ratio.mean().item()),
                "ratio_std": float(ratio_std.item()),
                "adv_mean": adv_mean,
                "adv_std": adv_std,
                "adv_max": adv_max,
                "adv_min": adv_min,
                "explained_variance": float(explained_variance.item()),
                "mean_return": float(batch_returns.mean().item()),
                "value_pred_mean": float(values_for_loss.mean().item()),
                "grad_norm_backbone": float(grad_norms["backbone"]),
                "grad_norm_policy": float(grad_norms["policy"]),
                "grad_norm_value": float(grad_norms["value"]),
                "grad_norm_total": float(total_grad_norm.item() if isinstance(total_grad_norm, torch.Tensor) else total_grad_norm),
            }
            metrics.append(row)
            kl_stop_metric = str(cfg.get("kl_stop_metric", "approx_kl"))
            kl_value = abs(row["approx_kl"]) if kl_stop_metric == "approx_kl" else row["base_kl"]
            if kl_value > float(cfg["target_kl"]):
                stopped_early = True
                break
        if stopped_early:
            break

    out: dict[str, float] = {}
    keys = metrics[0].keys() if metrics else []
    for key in keys:
        out[key] = sum(m[key] for m in metrics) / max(1, len(metrics))
    out["updates"] = float(len(metrics))
    out["early_stop"] = 1.0 if stopped_early else 0.0
    return out


def update_stats(stats: dict[str, dict[str, int]], key: str, reward: float, has_error: bool) -> None:
    row = stats.setdefault(key, {"wins": 0, "losses": 0, "draws": 0, "errors": 0, "games": 0})
    row["games"] += 1
    if has_error:
        row["errors"] += 1
    elif reward > 0:
        row["wins"] += 1
    elif reward < 0:
        row["losses"] += 1
    else:
        row["draws"] += 1


def winrate(row: dict[str, int]) -> float:
    valid = max(1, int(row.get("wins", 0)) + int(row.get("losses", 0)) + int(row.get("draws", 0)))
    return float(row.get("wins", 0)) / valid


def total_stats(stats: dict[str, dict[str, int]]) -> dict[str, int]:
    total = {"wins": 0, "losses": 0, "draws": 0, "errors": 0, "games": 0}
    for row in stats.values():
        for key in total:
            total[key] += int(row.get(key, 0))
    return total


def current_train_games(runners: list[RunnerSpec]) -> dict[str, int]:
    return {spec.key: int(spec.train_games) for spec in runners}


def apply_train_games(runners: list[RunnerSpec], counts: dict[str, int]) -> None:
    for spec in runners:
        spec.train_games = max(0, int(counts.get(spec.key, spec.train_games)))


def initial_train_weights(runners: list[RunnerSpec]) -> dict[str, float]:
    return {spec.key: 1.0 for spec in runners}


def train_weights_from_eval(runners: list[RunnerSpec], by_wr: dict[str, float]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for spec in runners:
        wr = float(by_wr.get(spec.key, 1.0))
        weight = 1.0
        if wr <= 0.5:
            weight += 1.0
        if wr <= 0.4:
            weight += 0.5
        weights[spec.key] = weight
    return weights


def counts_from_weights(runners: list[RunnerSpec], weights: dict[str, float], total_games: int) -> dict[str, int]:
    total_weight = sum(max(0.0, float(weights.get(spec.key, 0.0))) for spec in runners)
    if total_weight <= 0:
        weights = initial_train_weights(runners)
        total_weight = float(len(runners))
    exact = {spec.key: max(0.0, float(weights.get(spec.key, 0.0))) * total_games / total_weight for spec in runners}
    counts = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = int(total_games) - sum(counts.values())
    if remaining > 0:
        order = sorted(exact, key=lambda key: (exact[key] - counts[key], exact[key]), reverse=True)
        for key in order[:remaining]:
            counts[key] += 1
    return counts


def latest_eval_by_wr() -> dict[str, float] | None:
    if EVAL_DIR is None or not EVAL_DIR.exists():
        return None
    latest: tuple[int, Path] | None = None
    for path in EVAL_DIR.glob("eval_iter_*.json"):
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        try:
            iteration = int(parts[2])
        except ValueError:
            continue
        if latest is None or iteration > latest[0]:
            latest = (iteration, path)
    if latest is None:
        return None
    try:
        data = json.loads(latest[1].read_text(encoding="utf-8"))
    except Exception:
        return None
    by_wr = data.get("by_wr")
    if not isinstance(by_wr, dict):
        return None
    return {str(key): float(value) for key, value in by_wr.items()}


def build_train_schedule(runners: list[RunnerSpec]) -> list[RunnerSpec]:
    schedule: list[RunnerSpec] = []
    for spec in runners:
        schedule.extend([spec] * spec.train_games)
    random.shuffle(schedule)
    return schedule


def collect_training_batch(
    runners: list[RunnerSpec],
    model: PTCGPolicyNet,
    device: torch.device,
    cfg: dict[str, Any],
    iteration: int,
) -> tuple[list[DecisionSample], dict[str, Any]]:
    model.eval()
    samples: list[DecisionSample] = []
    stats: dict[str, dict[str, int]] = {}
    invalid_actions = 0
    rewards: list[float] = []
    episode_lengths: list[int] = []
    schedule = build_train_schedule(runners)
    for game_idx, spec in enumerate(schedule):
        game_samples, info = collect_ppo_game(
            spec.runner,
            model,
            device,
            cfg,
            learner_seat=(iteration + game_idx) % 2,
            greedy=False,
        )
        samples.extend(game_samples)
        invalid_actions += int(info.get("invalid_model_actions", 0))
        rewards.append(float(info.get("reward", 0.0)) if "error" not in info else 0.0)
        episode_lengths.append(int(info.get("episode_len", info.get("decisions", 0))))
        update_stats(stats, spec.key, float(info.get("reward", 0.0)), "error" in info)
    total = total_stats(stats)
    return samples, {
        "stats": stats,
        "opponent_stats": stats,
        "total": total,
        "invalid_actions": invalid_actions,
        "avg_episode_len": sum(episode_lengths) / max(1, len(episode_lengths)),
        "avg_reward": sum(rewards) / max(1, len(rewards)),
    }


@torch.no_grad()
def evaluate_policy(
    runners: list[RunnerSpec],
    model: PTCGPolicyNet,
    device: torch.device,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    stats: dict[str, dict[str, int]] = {}
    invalid_actions = 0
    for spec in runners:
        for game_idx in range(spec.eval_games):
            _, info = collect_ppo_game(
                spec.runner,
                model,
                device,
                cfg,
                learner_seat=game_idx % 2,
                greedy=True,
            )
            invalid_actions += int(info.get("invalid_model_actions", 0))
            update_stats(stats, spec.key, float(info.get("reward", 0.0)), "error" in info)
    total = total_stats(stats)
    by_wr = {key: winrate(row) for key, row in stats.items()}
    worst_key = min(by_wr, key=by_wr.get) if by_wr else ""
    total_wr = winrate(total)
    worst_wr = by_wr.get(worst_key, 0.0)
    mean_wr = sum(by_wr.values()) / max(1, len(by_wr))
    score = 0.55 * mean_wr + 0.35 * worst_wr + 0.10 * total_wr
    return {
        "stats": stats,
        "total": total,
        "by_wr": by_wr,
        "total_wr": total_wr,
        "mean_wr": mean_wr,
        "worst_key": worst_key,
        "worst_wr": worst_wr,
        "score": score,
        "invalid_actions": invalid_actions,
    }


def best_path(metric: str, rank: int) -> Path:
    return CHECKPOINT_DIR / f"ppo_rules_best_{metric}_rank{rank}.pt"


def latest_manifest(best_records: dict[str, list[dict[str, Any]]], latest_extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "latest": str(LATEST_PATH),
        "latest_extra": latest_extra,
        "best_records": best_records,
    }


def write_manifest(best_records: dict[str, list[dict[str, Any]]], latest_extra: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(latest_manifest(best_records, latest_extra), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def update_topk(
    metric: str,
    value: float,
    best_records: dict[str, list[dict[str, Any]]],
    model: PTCGPolicyNet,
    optimizer: torch.optim.Optimizer,
    extra: dict[str, Any],
    keep: int,
) -> None:
    old_records = list(best_records.get(metric, []))
    candidate_id = f"{metric}_it{int(extra['iteration'])}_{value:.8f}"
    candidate = {
        "id": candidate_id,
        "metric": metric,
        "value": float(value),
        "iteration": int(extra["iteration"]),
        "path": "",
        "summary": extra.get("summary", {}),
    }
    records = old_records + [candidate]
    records.sort(key=lambda row: float(row["value"]), reverse=True)
    top = records[:keep]
    if candidate_id not in {row["id"] for row in top}:
        best_records[metric] = top
        return

    tmp_dir = CHECKPOINT_DIR / "_topk_tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    old_files: dict[str, Path] = {}
    for rank, record in enumerate(old_records[:keep], 1):
        source = Path(record.get("path") or best_path(metric, rank))
        if source.exists():
            tmp = tmp_dir / f"{record['id']}.pt"
            shutil.copy2(source, tmp)
            old_files[record["id"]] = tmp

    for rank, record in enumerate(top, 1):
        target = best_path(metric, rank)
        record["path"] = str(target)
        if record["id"] == candidate_id:
            payload = dict(extra)
            payload["best_metric"] = metric
            payload["best_rank"] = rank
            payload["best_value"] = float(value)
            save_checkpoint(target, model, optimizer, payload)
            print(f"new PPO best {metric} rank{rank}={value:.4f} -> {target}", flush=True)
        elif record["id"] in old_files:
            shutil.copy2(old_files[record["id"]], target)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    best_records[metric] = top


def flatten_eval_for_row(eval_result: dict[str, Any]) -> dict[str, Any]:
    by_wr = eval_result.get("by_wr", {})
    row = {
        "eval_score": eval_result.get("score", ""),
        "eval_total_wr": eval_result.get("total_wr", ""),
        "eval_mean_wr": eval_result.get("mean_wr", ""),
        "eval_worst_key": eval_result.get("worst_key", ""),
        "eval_worst_wr": eval_result.get("worst_wr", ""),
        "eval_original_archaludon_wr": by_wr.get("original_archaludon", ""),
        "eval_official_iono_wr": by_wr.get("official_iono", ""),
        "eval_invalid_actions": eval_result.get("invalid_actions", ""),
    }
    for key in OPPONENT_KEYS:
        row[f"eval_{key}_wr"] = by_wr.get(key, "")
    return row


def summary_from_eval(eval_result: dict[str, Any]) -> dict[str, Any]:
    by_wr = eval_result["by_wr"]
    return {
        "score": eval_result["score"],
        "total_wr": eval_result["total_wr"],
        "mean_wr": eval_result["mean_wr"],
        "worst_key": eval_result["worst_key"],
        "worst_wr": eval_result["worst_wr"],
        "original_archaludon_wr": by_wr.get("original_archaludon", 0.0),
        "official_iono_wr": by_wr.get("official_iono", 0.0),
    }


def update_all_bests(
    summary: dict[str, Any],
    best_records: dict[str, list[dict[str, Any]]],
    model: PTCGPolicyNet,
    optimizer: torch.optim.Optimizer,
    extra: dict[str, Any],
    keep: int,
) -> None:
    best_extra = dict(extra)
    best_extra["summary"] = summary
    update_topk("score", summary["score"], best_records, model, optimizer, best_extra, keep)
    update_topk("total_wr", summary["total_wr"], best_records, model, optimizer, best_extra, keep)
    update_topk("worst_wr", summary["worst_wr"], best_records, model, optimizer, best_extra, keep)
    update_topk("original_archaludon_wr", summary["original_archaludon_wr"], best_records, model, optimizer, best_extra, keep)
    update_topk("official_iono_wr", summary["official_iono_wr"], best_records, model, optimizer, best_extra, keep)


def load_start_or_resume(
    model: PTCGPolicyNet,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    device: torch.device,
    runners: list[RunnerSpec],
):
    if LATEST_PATH.exists():
        extra = load_checkpoint(LATEST_PATH, model, optimizer, device=device)
        best_records = extra.get("best_records", {metric: [] for metric in BEST_METRICS})
        print(f"resume PPO from {LATEST_PATH} iteration={extra.get('iteration')}", flush=True)
        by_wr = latest_eval_by_wr()
        if by_wr is not None:
            weights = train_weights_from_eval(runners, by_wr)
            print("resume train weights recomputed from latest eval; unseen opponents use default weight=1", flush=True)
        else:
            saved_weights = extra.get("current_train_weights") or {}
            weights = {spec.key: float(saved_weights.get(spec.key, 1.0)) for spec in runners}
        plan = counts_from_weights(runners, weights, int(cfg["games_per_iteration"]))
        return int(extra.get("iteration", 0)), int(extra.get("seen_decisions", 0)), best_records, weights, plan, True

    start = project_path(str(cfg["start_checkpoint"]))
    if not start.exists():
        raise RuntimeError(f"start checkpoint not found: {start}")
    extra = load_checkpoint(start, model, device=device)
    print(f"start PPO from {start} extra={extra}", flush=True)
    weights = initial_train_weights(runners)
    return 0, 0, {metric: [] for metric in BEST_METRICS}, weights, counts_from_weights(runners, weights, int(cfg["games_per_iteration"])), False


def main() -> None:
    cfg = load_run_config()
    setup_run_paths(str(cfg["run_name"]))
    for path in (CHECKPOINT_DIR, LOG_DIR, EVAL_DIR):
        path.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg["seed"]))
    torch.set_num_threads(int(cfg.get("torch_num_threads", 4)))
    device = choose_device()

    runners = build_runners(cfg)
    print(f"PPO rules device={device} run={cfg['run_name']} params={sum(p.numel() for p in PTCGPolicyNet().parameters()):,}", flush=True)
    print(
        "opponents: "
        + ", ".join(
            f"{spec.key}[train={spec.train_games},eval={spec.eval_games}]"
            for spec in runners
        ),
        flush=True,
    )

    model = PTCGPolicyNet().to(device)
    base_model = PTCGPolicyNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    start_iteration, seen_decisions, best_records, train_weights, train_plan, resumed_from_latest = load_start_or_resume(
        model, optimizer, cfg, device, runners
    )
    apply_train_games(runners, train_plan)
    base_extra = load_checkpoint(project_path(str(cfg["start_checkpoint"])), base_model, device=device)
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad_(False)
    print(f"frozen base model loaded for KL extra={base_extra}", flush=True)

    eval_every = int(cfg.get("eval_every", 50))
    keep = int(cfg["best_keep"])

    if start_iteration == 0 and cfg.get("eval_before_training", False) and not resumed_from_latest:
        print("evaluating start checkpoint before PPO updates...", flush=True)
        eval_result = evaluate_policy(runners, model, device, cfg)
        eval_path = EVAL_DIR / "eval_iter_0000_start.json"
        eval_path.write_text(json.dumps(eval_result, ensure_ascii=False, indent=2), encoding="utf-8")
        train_weights = train_weights_from_eval(runners, eval_result["by_wr"])
        train_plan = counts_from_weights(runners, train_weights, int(cfg["games_per_iteration"]))
        summary = summary_from_eval(eval_result)
        latest_extra = {
            "iteration": 0,
            "seen_decisions": seen_decisions,
            "current_train_weights": train_weights,
            "current_train_games": train_plan,
            "config": cfg,
            "train_info": {},
            "train_metrics": {},
            "eval_result": eval_result,
            "best_records": best_records,
            "baseline": True,
        }
        update_all_bests(summary, best_records, model, optimizer, latest_extra, keep)
        latest_extra["best_records"] = best_records
        save_checkpoint(LATEST_PATH, model, optimizer, latest_extra)
        write_manifest(best_records, latest_extra)
        by_wr = eval_result["by_wr"]
        print(
            f"it=0 baseline eval total={eval_result['total_wr']:.3f} "
            f"orig={by_wr.get('original_archaludon', 0.0):.3f} v4={by_wr.get('archaludon_v4', 0.0):.3f} "
            f"iono={by_wr.get('official_iono', 0.0):.3f} ala={by_wr.get('community_alakazam', 0.0):.3f} "
            f"worst={eval_result['worst_key']}:{eval_result['worst_wr']:.3f} score={eval_result['score']:.3f} "
            f"next={' '.join(f'{k}:{v}' for k, v in train_plan.items())}",
            flush=True,
        )

    for iteration in range(start_iteration + 1, int(cfg["iterations"]) + 1):
        t0 = time.time()
        apply_train_games(runners, train_plan)
        train_games_this_iter = sum(train_plan.values())
        samples, train_info = collect_training_batch(runners, model, device, cfg, iteration)
        if not samples:
            raise RuntimeError(f"no PPO samples collected at iteration {iteration}: {train_info}")
        seen_decisions += len(samples)
        train_metrics = ppo_update(model, base_model, optimizer, samples, cfg, device)
        train_total = train_info["total"]
        train_wr = winrate(train_total)
        elapsed = time.time() - t0

        eval_result: dict[str, Any] = {}
        do_eval = eval_every > 0 and iteration % eval_every == 0
        rolled_back = False
        if do_eval:
            eval_result = evaluate_policy(runners, model, device, cfg)
            eval_path = EVAL_DIR / f"eval_iter_{iteration:04d}.json"
            eval_path.write_text(json.dumps(eval_result, ensure_ascii=False, indent=2), encoding="utf-8")
            train_weights = train_weights_from_eval(runners, eval_result["by_wr"])
            train_plan = counts_from_weights(runners, train_weights, int(cfg["games_per_iteration"]))

        latest_extra = {
            "iteration": iteration,
            "seen_decisions": seen_decisions,
            "current_train_weights": train_weights,
            "current_train_games": train_plan,
            "config": cfg,
            "train_info": train_info,
            "train_metrics": train_metrics,
            "eval_result": eval_result,
            "best_records": best_records,
        }
        if iteration % int(cfg["latest_every"]) == 0 and not do_eval:
            save_checkpoint(LATEST_PATH, model, optimizer, latest_extra)

        if do_eval:
            summary = summary_from_eval(eval_result)
            update_all_bests(summary, best_records, model, optimizer, latest_extra, keep)
            latest_extra["best_records"] = best_records
            save_checkpoint(LATEST_PATH, model, optimizer, latest_extra)

        write_manifest(best_records, latest_extra)

        row = {
            "iteration": iteration,
            "games": train_games_this_iter,
            "decisions": len(samples),
            "seen_decisions": seen_decisions,
            "seconds": round(elapsed, 3),
            "train_wr": train_wr,
            "train_wld": f"{train_total['wins']}/{train_total['losses']}/{train_total['draws']}/{train_total['errors']}",
            "train_invalid_actions": train_info["invalid_actions"],
            "avg_episode_len": train_info.get("avg_episode_len", ""),
            "avg_reward": train_info.get("avg_reward", ""),
            "train_weights": json.dumps(train_weights, ensure_ascii=False),
            "train_plan": json.dumps(train_plan, ensure_ascii=False),
            "train_stats": json.dumps(train_info["stats"], ensure_ascii=False),
            "loss": train_metrics.get("loss", ""),
            "policy_loss": train_metrics.get("policy_loss", ""),
            "value_loss": train_metrics.get("value_loss", ""),
            "entropy": train_metrics.get("entropy", ""),
            "bc_loss": train_metrics.get("bc_loss", ""),
            "base_kl": train_metrics.get("base_kl", ""),
            "approx_kl": train_metrics.get("approx_kl", ""),
            "clip_fraction": train_metrics.get("clip_fraction", ""),
            "ratio_mean": train_metrics.get("ratio_mean", ""),
            "ratio_std": train_metrics.get("ratio_std", ""),
            "adv_mean": train_metrics.get("adv_mean", ""),
            "adv_std": train_metrics.get("adv_std", ""),
            "adv_max": train_metrics.get("adv_max", ""),
            "adv_min": train_metrics.get("adv_min", ""),
            "explained_variance": train_metrics.get("explained_variance", ""),
            "value_pred_mean": train_metrics.get("value_pred_mean", ""),
            "mean_return": train_metrics.get("mean_return", ""),
            "grad_norm_backbone": train_metrics.get("grad_norm_backbone", ""),
            "grad_norm_policy": train_metrics.get("grad_norm_policy", ""),
            "grad_norm_value": train_metrics.get("grad_norm_value", ""),
            "grad_norm_total": train_metrics.get("grad_norm_total", ""),
            "updates": train_metrics.get("updates", ""),
            "early_stop": train_metrics.get("early_stop", ""),
            "rolled_back": int(rolled_back),
            **flatten_eval_for_row(eval_result),
        }
        _append_csv(METRICS_PATH, row)

        eval_text = ""
        if eval_result:
            by_wr = eval_result["by_wr"]
            eval_text = (
                f" eval total={eval_result['total_wr']:.3f}"
                f" orig={by_wr.get('original_archaludon', 0.0):.3f}"
                f" v4={by_wr.get('archaludon_v4', 0.0):.3f}"
                f" iono={by_wr.get('official_iono', 0.0):.3f}"
                f" ala={by_wr.get('community_alakazam', 0.0):.3f}"
                f" worst={eval_result['worst_key']}:{eval_result['worst_wr']:.3f}"
                f" score={eval_result['score']:.3f}"
            )
        opponent_text = ",".join(
            f"{name}:{row.get('wins', 0)}/{row.get('losses', 0)}/{row.get('draws', 0)}/{row.get('errors', 0)}"
            for name, row in train_info.get("opponent_stats", {}).items()
        )
        valid_games = max(
            1,
            int(train_total.get("wins", 0)) + int(train_total.get("losses", 0)) + int(train_total.get("draws", 0)),
        )
        print(
            f"it={iteration} sec={elapsed:.1f} updates={train_metrics.get('updates', 0):.0f}{eval_text}\n"
            f"  Rollout: avg_len={train_info.get('avg_episode_len', 0.0):.1f} "
            f"avg_reward={train_info.get('avg_reward', 0.0):.3f} "
            f"wr={train_total.get('wins', 0) / valid_games:.3f} "
            f"w/l/d/e={train_total.get('wins', 0)}/{train_total.get('losses', 0)}/{train_total.get('draws', 0)}/{train_total.get('errors', 0)} "
            f"decisions={len(samples)} pool={opponent_text}\n"
            f"  Policy: loss={train_metrics.get('policy_loss', 0.0):.4f} "
            f"value_loss={train_metrics.get('value_loss', 0.0):.4f} "
            f"entropy={train_metrics.get('entropy', 0.0):.4f} "
            f"approx_kl={train_metrics.get('approx_kl', 0.0):.5f} "
            f"clip_frac={train_metrics.get('clip_fraction', 0.0):.3f} "
            f"ratio={train_metrics.get('ratio_mean', 0.0):.3f}/{train_metrics.get('ratio_std', 0.0):.3f}\n"
            f"  Advantage: mean={train_metrics.get('adv_mean', 0.0):.3f} "
            f"std={train_metrics.get('adv_std', 0.0):.3f} "
            f"max={train_metrics.get('adv_max', 0.0):.3f} min={train_metrics.get('adv_min', 0.0):.3f} | "
            f"Value: ev={train_metrics.get('explained_variance', 0.0):.3f} "
            f"pred_mean={train_metrics.get('value_pred_mean', 0.0):.3f} "
            f"return_mean={train_metrics.get('mean_return', 0.0):.3f}\n"
            f"  Grad: backbone={train_metrics.get('grad_norm_backbone', 0.0):.3f} "
            f"policy={train_metrics.get('grad_norm_policy', 0.0):.3f} "
            f"value={train_metrics.get('grad_norm_value', 0.0):.3f} "
            f"total={train_metrics.get('grad_norm_total', 0.0):.3f} "
            f"rollback={int(rolled_back)}",
            flush=True,
        )

    save_checkpoint(
        FINAL_PATH,
        model,
        optimizer,
        {"iteration": int(cfg["iterations"]), "best_records": best_records, "current_train_games": train_plan},
    )
    print(f"wrote {FINAL_PATH}", flush=True)


if __name__ == "__main__":
    main()
