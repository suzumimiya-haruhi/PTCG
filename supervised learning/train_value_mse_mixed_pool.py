from __future__ import annotations

import csv
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
VALUE_ROOT = ROOT / "value_mse"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ptcg_rl import config as base_config
from ptcg_rl.env import DecisionSample, active_player
from ptcg_rl.features import FeatureEncoder, HistoryTracker
from ptcg_rl.model import PTCGPolicyNet, batch_encoded, select_action_from_logits
from ptcg_rl.rule_teacher import RuleTeacher
from ptcg_rl.train_utils import save_checkpoint
from ptcg_rl.utils import action_is_valid, call_agent, import_agent_module, reset_rule_agent, set_seed


CONFIG_PATH = Path(os.environ.get("PTCG_VALUE_MSE_CONFIG", str(VALUE_ROOT / "configs" / "value_mse_mixed_pool.json")))


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def choose_device() -> torch.device:
    if base_config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def run_paths(run_name: str) -> dict[str, Path]:
    run_root = VALUE_ROOT / "runs" / run_name
    return {
        "run_root": run_root,
        "checkpoints": run_root / "checkpoints",
        "logs": run_root / "logs",
        "eval": run_root / "fixed_eval",
        "latest": run_root / "checkpoints" / "value_mse_latest.pt",
        "final": run_root / "checkpoints" / "value_mse_final.pt",
        "metrics": run_root / "logs" / "value_mse_metrics.csv",
        "manifest": run_root / "manifest.json",
    }


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def reset_module(module: torch.nn.Module) -> None:
    for child in module.modules():
        if child is module:
            continue
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()


def freeze_except_value(model: PTCGPolicyNet) -> list[torch.nn.Parameter]:
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("value."))
    return [param for param in model.parameters() if param.requires_grad]


def set_value_only_mode(model: PTCGPolicyNet) -> None:
    # Keep dropout disabled everywhere; gradients still flow through value params.
    model.eval()


def load_start_or_resume(
    model: PTCGPolicyNet,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    paths: dict[str, Path],
    device: torch.device,
) -> tuple[int, int, list[dict[str, Any]]]:
    if bool(cfg.get("resume", True)) and paths["latest"].exists():
        payload = torch.load(paths["latest"], map_location=device)
        model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        extra = payload.get("extra", {})
        print(f"resume value MSE from {paths['latest']} iteration={extra.get('iteration')}", flush=True)
        return int(extra.get("iteration", 0)), int(extra.get("seen_decisions", 0)), list(extra.get("best_val_mse", []))

    start_path = project_path(str(cfg["start_checkpoint"]))
    payload = torch.load(start_path, map_location=device)
    model.load_state_dict(payload["model"])
    if bool(cfg.get("reset_value_on_fresh_start", True)):
        reset_module(model.value)
    print(
        f"start value MSE from checkpoint={start_path} "
        f"extra_iteration={payload.get('extra', {}).get('iteration')}",
        flush=True,
    )
    return 0, 0, []


def import_cg_game():
    submission = base_config.ORIG_SUBMISSION
    if str(submission) not in sys.path:
        sys.path.insert(0, str(submission))
    from cg import game as cg_game

    return cg_game


def target_for_player(result: int, player_index: int) -> float:
    if result == 2:
        return 0.0
    if result in (0, 1):
        return 1.0 if result == player_index else -1.0
    return 0.0


def random_valid_action(encoded) -> list[int]:
    valid_count = max(0, int(encoded.num_options))
    max_count = min(max(0, int(encoded.max_count)), valid_count)
    min_count = min(max(0, int(encoded.min_count)), max_count)
    if valid_count <= 0 or max_count <= 0:
        return []
    k = random.randint(min_count, max_count)
    if k <= 0:
        return []
    return random.sample(range(valid_count), k)


@torch.no_grad()
def select_policy_action(
    model: PTCGPolicyNet,
    encoded,
    device: torch.device,
    cfg: dict[str, Any],
    greedy: bool,
    epsilon: float,
) -> list[int]:
    if random.random() < float(epsilon):
        return random_valid_action(encoded)
    batch = batch_encoded([encoded], device)
    logits, _ = model(batch)
    temperature = max(0.05, float(cfg.get("temperature", 1.0)))
    scaled_logits = logits[0] / temperature
    action, _, _ = select_action_from_logits(
        scaled_logits,
        batch["option_mask"][0],
        int(encoded.min_count),
        int(encoded.max_count),
        greedy=greedy,
    )
    return action


@torch.no_grad()
def collect_selfplay_game(
    cg_game,
    model: PTCGPolicyNet,
    encoder: FeatureEncoder,
    teacher: RuleTeacher,
    deck: list[int],
    device: torch.device,
    cfg: dict[str, Any],
    seed: int,
    greedy: bool,
    epsilon: float,
) -> tuple[list[DecisionSample], dict[str, Any]]:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    reset_rule_agent(teacher.module)
    obs, start = cg_game.battle_start(list(deck), list(deck))
    if obs is None:
        return [], {"error": f"battle_start {start.errorPlayer}/{start.errorType}"}

    history = HistoryTracker()
    samples: list[DecisionSample] = []
    try:
        for step in range(int(cfg.get("max_steps_per_game", base_config.MAX_STEPS_PER_GAME))):
            cur = obs.get("current") or {}
            result = int(cur.get("result", -1))
            if result != -1:
                break
            pi = active_player(obs)
            if pi is None:
                break
            encoded = encoder.encode(obs, history, teacher.module)
            action = select_policy_action(model, encoded, device, cfg, greedy=greedy, epsilon=epsilon)
            if not action_is_valid(obs, action):
                fallback = select_policy_action(model, encoded, device, cfg, greedy=True, epsilon=0.0)
                action = fallback if action_is_valid(obs, fallback) else random_valid_action(encoded)
            if not action_is_valid(obs, action):
                return [], {"error": f"invalid action by {pi}", "action": action, "step": step}
            samples.append(
                DecisionSample(
                    encoded=encoded,
                    teacher_action=[],
                    action=action,
                    player_index=int(pi),
                )
            )
            history.update(obs, int(pi), action)
            obs = cg_game.battle_select(action)

        result = int((obs.get("current") or {}).get("result", -1))
        for sample in samples:
            sample.reward = target_for_player(result, int(sample.player_index))
        return samples, {
            "result": result,
            "decisions": len(samples),
            "winner": result if result in (0, 1) else "",
            "draw": result == 2 or result == -1,
        }
    finally:
        cg_game.battle_finish()


def collect_selfplay_batch(
    model: PTCGPolicyNet,
    device: torch.device,
    cfg: dict[str, Any],
    games: int,
    seed_base: int,
    greedy: bool,
    epsilon: float,
    label: str,
) -> tuple[list[DecisionSample], dict[str, Any]]:
    cg_game = import_cg_game()
    encoder = FeatureEncoder()
    teacher = RuleTeacher(base_config.ORIG_SUBMISSION, f"value_mse_teacher_{label}")
    teacher.reset()
    deck = teacher.deck()
    samples: list[DecisionSample] = []
    stats = {"games": 0, "p0_wins": 0, "p1_wins": 0, "draws": 0, "errors": 0, "decisions": 0}
    for game_idx in range(int(games)):
        seed = int(seed_base) + game_idx * 10007 + (game_idx % 17) * 97
        game_samples, info = collect_selfplay_game(
            cg_game,
            model,
            encoder,
            teacher,
            deck,
            device,
            cfg,
            seed=seed,
            greedy=greedy,
            epsilon=epsilon,
        )
        stats["games"] += 1
        if "error" in info:
            stats["errors"] += 1
        elif info.get("result") == 0:
            stats["p0_wins"] += 1
            samples.extend(game_samples)
        elif info.get("result") == 1:
            stats["p1_wins"] += 1
            samples.extend(game_samples)
        else:
            stats["draws"] += 1
            samples.extend(game_samples)
        stats["decisions"] += len(game_samples)
        if (game_idx + 1) % 100 == 0:
            print(f"  {label} collected {game_idx + 1}/{games} games decisions={len(samples)}", flush=True)
    return samples, stats


def load_mixed_opponents(cfg: dict[str, Any], frozen_model: PTCGPolicyNet) -> list[dict[str, Any]]:
    opponents: list[dict[str, Any]] = []
    for idx, item in enumerate(cfg["opponent_pool"]):
        name = str(item["name"])
        opponent_type = str(item.get("type", "rule"))
        weight = float(item.get("weight", 1.0))
        row: dict[str, Any] = {"name": name, "type": opponent_type, "weight": weight}
        if opponent_type == "rule":
            submission_dir = project_path(str(item["path"]))
            if not (submission_dir / "main.py").exists() or not (submission_dir / "deck.csv").exists():
                raise FileNotFoundError(f"missing rule opponent {name}: {submission_dir}")
            row["path"] = submission_dir
            row["module"] = import_agent_module(f"value_mse_opp_{idx}_{name}", submission_dir)
        elif opponent_type == "frozen_self":
            row["model"] = frozen_model
        else:
            raise ValueError(f"unsupported opponent type for {name}: {opponent_type}")
        opponents.append(row)
    if not opponents:
        raise RuntimeError("opponent_pool is empty")
    return opponents


def choose_mixed_opponent(opponents: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    total = sum(max(0.0, float(row.get("weight", 0.0))) for row in opponents)
    if total <= 0:
        return opponents[rng.randrange(len(opponents))]
    pick = rng.random() * total
    acc = 0.0
    for row in opponents:
        acc += max(0.0, float(row.get("weight", 0.0)))
        if pick <= acc:
            return row
    return opponents[-1]


@torch.no_grad()
def collect_model_vs_rule_game(
    cg_game,
    model: PTCGPolicyNet,
    encoder: FeatureEncoder,
    teacher: RuleTeacher,
    opponent: dict[str, Any],
    device: torch.device,
    cfg: dict[str, Any],
    seed: int,
    learner_seat: int,
    greedy: bool,
    epsilon: float,
) -> tuple[list[DecisionSample], dict[str, Any]]:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    teacher.reset()
    reset_rule_agent(opponent["module"])
    deck_a = teacher.deck()
    deck_b = call_agent(opponent["module"], opponent["path"], {"select": None, "logs": [], "current": None})
    decks = [None, None]
    decks[learner_seat] = deck_a
    decks[1 - learner_seat] = deck_b
    obs, start = cg_game.battle_start(decks[0], decks[1])
    if obs is None:
        return [], {"error": f"battle_start {start.errorPlayer}/{start.errorType}", "opponent": opponent["name"]}

    history = HistoryTracker()
    samples: list[DecisionSample] = []
    try:
        for step in range(int(cfg.get("max_steps_per_game", base_config.MAX_STEPS_PER_GAME))):
            cur = obs.get("current") or {}
            result = int(cur.get("result", -1))
            if result != -1:
                break
            pi = active_player(obs)
            if pi is None:
                break
            if int(pi) == learner_seat:
                encoded = encoder.encode(obs, history, teacher.module)
                action = select_policy_action(model, encoded, device, cfg, greedy=greedy, epsilon=epsilon)
                if not action_is_valid(obs, action):
                    fallback = select_policy_action(model, encoded, device, cfg, greedy=True, epsilon=0.0)
                    action = fallback if action_is_valid(obs, fallback) else random_valid_action(encoded)
                samples.append(
                    DecisionSample(
                        encoded=encoded,
                        teacher_action=[],
                        action=action,
                        player_index=learner_seat,
                    )
                )
            else:
                action = call_agent(opponent["module"], opponent["path"], obs)

            if not action_is_valid(obs, action):
                return [], {"error": f"invalid action by {pi}", "action": action, "step": step, "opponent": opponent["name"]}
            history.update(obs, int(pi), action)
            obs = cg_game.battle_select(action)

        result = int((obs.get("current") or {}).get("result", -1))
        target = target_for_player(result, learner_seat)
        for sample in samples:
            sample.reward = target
        return samples, {
            "result": result,
            "reward": target,
            "decisions": len(samples),
            "opponent": opponent["name"],
        }
    finally:
        cg_game.battle_finish()


@torch.no_grad()
def collect_model_vs_frozen_game(
    cg_game,
    model: PTCGPolicyNet,
    frozen_model: PTCGPolicyNet,
    encoder: FeatureEncoder,
    teacher: RuleTeacher,
    deck: list[int],
    device: torch.device,
    cfg: dict[str, Any],
    seed: int,
    learner_seat: int,
    greedy: bool,
    epsilon: float,
) -> tuple[list[DecisionSample], dict[str, Any]]:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    reset_rule_agent(teacher.module)
    decks = [list(deck), list(deck)]
    obs, start = cg_game.battle_start(decks[0], decks[1])
    if obs is None:
        return [], {"error": f"battle_start {start.errorPlayer}/{start.errorType}", "opponent": "frozen_self"}

    history = HistoryTracker()
    samples: list[DecisionSample] = []
    try:
        for step in range(int(cfg.get("max_steps_per_game", base_config.MAX_STEPS_PER_GAME))):
            cur = obs.get("current") or {}
            result = int(cur.get("result", -1))
            if result != -1:
                break
            pi = active_player(obs)
            if pi is None:
                break
            encoded = encoder.encode(obs, history, teacher.module)
            actor = model if int(pi) == learner_seat else frozen_model
            action = select_policy_action(actor, encoded, device, cfg, greedy=greedy, epsilon=epsilon)
            if not action_is_valid(obs, action):
                fallback = select_policy_action(actor, encoded, device, cfg, greedy=True, epsilon=0.0)
                action = fallback if action_is_valid(obs, fallback) else random_valid_action(encoded)
            if int(pi) == learner_seat:
                samples.append(
                    DecisionSample(
                        encoded=encoded,
                        teacher_action=[],
                        action=action,
                        player_index=learner_seat,
                    )
                )
            if not action_is_valid(obs, action):
                return [], {"error": f"invalid action by {pi}", "action": action, "step": step, "opponent": "frozen_self"}
            history.update(obs, int(pi), action)
            obs = cg_game.battle_select(action)

        result = int((obs.get("current") or {}).get("result", -1))
        target = target_for_player(result, learner_seat)
        for sample in samples:
            sample.reward = target
        return samples, {
            "result": result,
            "reward": target,
            "decisions": len(samples),
            "opponent": "frozen_self",
        }
    finally:
        cg_game.battle_finish()


def collect_mixed_pool_batch(
    model: PTCGPolicyNet,
    frozen_model: PTCGPolicyNet,
    opponents: list[dict[str, Any]],
    device: torch.device,
    cfg: dict[str, Any],
    games: int,
    seed_base: int,
    greedy: bool,
    epsilon: float,
    label: str,
) -> tuple[list[DecisionSample], dict[str, Any]]:
    cg_game = import_cg_game()
    encoder = FeatureEncoder()
    teacher = RuleTeacher(base_config.ORIG_SUBMISSION, f"value_mse_teacher_{label}")
    teacher.reset()
    deck = teacher.deck()
    samples: list[DecisionSample] = []
    stats: dict[str, Any] = {
        "games": 0,
        "wins": 0,
        "losses": 0,
        "p0_wins": 0,
        "p1_wins": 0,
        "draws": 0,
        "errors": 0,
        "decisions": 0,
        "opponents": {},
    }
    for game_idx in range(int(games)):
        seed = int(seed_base) + game_idx * 10007 + (game_idx % 17) * 97
        opponent = choose_mixed_opponent(opponents, seed + 7919)
        learner_seat = game_idx % 2
        if opponent["type"] == "frozen_self":
            game_samples, info = collect_model_vs_frozen_game(
                cg_game,
                model,
                frozen_model,
                encoder,
                teacher,
                deck,
                device,
                cfg,
                seed=seed,
                learner_seat=learner_seat,
                greedy=greedy,
                epsilon=epsilon,
            )
        else:
            game_samples, info = collect_model_vs_rule_game(
                cg_game,
                model,
                encoder,
                teacher,
                opponent,
                device,
                cfg,
                seed=seed,
                learner_seat=learner_seat,
                greedy=greedy,
                epsilon=epsilon,
            )
        name = str(info.get("opponent", opponent["name"]))
        row = stats["opponents"].setdefault(name, {"games": 0, "wins": 0, "losses": 0, "draws": 0, "errors": 0, "decisions": 0})
        stats["games"] += 1
        row["games"] += 1
        if "error" in info:
            stats["errors"] += 1
            row["errors"] += 1
        else:
            result = int(info.get("result", -1))
            if result == 0:
                stats["p0_wins"] += 1
            elif result == 1:
                stats["p1_wins"] += 1
            if result == learner_seat:
                stats["wins"] += 1
                row["wins"] += 1
            elif result in (0, 1):
                stats["losses"] += 1
                row["losses"] += 1
            else:
                stats["draws"] += 1
                row["draws"] += 1
            samples.extend(game_samples)
        stats["decisions"] += len(game_samples)
        row["decisions"] += len(game_samples)
        if (game_idx + 1) % 100 == 0:
            print(f"  {label} collected {game_idx + 1}/{games} games decisions={len(samples)}", flush=True)
    return samples, stats


@torch.no_grad()
def encode_states(
    model: PTCGPolicyNet,
    samples: list[DecisionSample],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    states: list[torch.Tensor] = []
    for start in range(0, len(samples), int(batch_size)):
        batch_samples = samples[start : start + int(batch_size)]
        if not batch_samples:
            continue
        batch = batch_encoded([sample.encoded for sample in batch_samples], device)
        _, state = model.encode_policy_inputs(batch)
        states.append(state.detach().cpu())
    if not states:
        return torch.empty((0, int(model.d_model)), dtype=torch.float32)
    return torch.cat(states, dim=0)


def samples_to_targets(samples: list[DecisionSample]) -> torch.Tensor:
    return torch.tensor([float(sample.reward) for sample in samples], dtype=torch.float32)


def value_mse(model: PTCGPolicyNet, states: torch.Tensor, targets: torch.Tensor, device: torch.device, batch_size: int) -> float:
    if states.numel() == 0:
        return 0.0
    model.value.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, states.shape[0], int(batch_size)):
            batch_states = states[start : start + int(batch_size)].to(device)
            batch_targets = targets[start : start + int(batch_size)].to(device)
            pred = torch.tanh(model.value(batch_states).squeeze(-1))
            losses.append(F.mse_loss(pred, batch_targets, reduction="sum").item())
    return float(sum(losses) / max(1, int(targets.numel())))


def baseline_mse(targets: torch.Tensor) -> float:
    if targets.numel() == 0:
        return 0.0
    mean_target = targets.mean()
    return float(torch.mean((targets - mean_target) ** 2).item())


def train_value_head(
    model: PTCGPolicyNet,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    targets: torch.Tensor,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    set_value_only_mode(model)
    indices = torch.arange(states.shape[0])
    batch_size = int(cfg["batch_size"])
    losses: list[float] = []
    updates = 0
    for _ in range(int(cfg["epochs"])):
        perm = indices[torch.randperm(indices.numel())]
        for start in range(0, perm.numel(), batch_size):
            batch_idx = perm[start : start + batch_size]
            if batch_idx.numel() == 0:
                continue
            batch_states = states[batch_idx].to(device)
            batch_targets = targets[batch_idx].to(device)
            pred = torch.tanh(model.value(batch_states).squeeze(-1))
            loss = F.mse_loss(pred, batch_targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.value.parameters(), float(cfg["max_grad_norm"]))
            optimizer.step()
            losses.append(float(loss.item()))
            updates += 1
    return {
        "train_mse": float(sum(losses) / max(1, len(losses))),
        "updates": float(updates),
    }


def validation_dataset_path(paths: dict[str, Path], cfg: dict[str, Any]) -> Path:
    return paths["eval"] / f"mixed_pool_val_seed{int(cfg['validation_seed'])}_games{int(cfg['validation_games'])}.pt"


def load_or_build_validation(
    model: PTCGPolicyNet,
    frozen_model: PTCGPolicyNet,
    opponents: list[dict[str, Any]],
    device: torch.device,
    cfg: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    path = validation_dataset_path(paths, cfg)
    if path.exists():
        print(f"load fixed mixed-pool validation states {path}", flush=True)
        return torch.load(path, map_location="cpu", weights_only=False)

    print(
        f"build fixed mixed-pool validation games={cfg['validation_games']} "
        f"seed={cfg['validation_seed']}",
        flush=True,
    )
    samples, stats = collect_mixed_pool_batch(
        model,
        frozen_model,
        opponents,
        device,
        cfg,
        games=int(cfg["validation_games"]),
        seed_base=int(cfg["validation_seed"]),
        greedy=bool(cfg.get("eval_greedy", False)),
        epsilon=float(cfg.get("eval_exploration_epsilon", 0.0)),
        label="validation",
    )
    states = encode_states(model, samples, device, int(cfg["state_batch_size"]))
    targets = samples_to_targets(samples)
    payload = {
        "states": states,
        "targets": targets,
        "stats": stats,
        "config": cfg,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    meta = dict(payload)
    meta["states"] = {"shape": list(states.shape)}
    meta["targets"] = {"shape": list(targets.shape), "mean": float(targets.mean().item()) if targets.numel() else 0.0}
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote fixed validation {path}", flush=True)
    return payload


def best_path(paths: dict[str, Path], rank: int) -> Path:
    return paths["checkpoints"] / f"value_mse_best_val_mse_rank{rank}.pt"


def update_best_by_mse(
    paths: dict[str, Path],
    best_records: list[dict[str, Any]],
    val_mse: float,
    model: PTCGPolicyNet,
    optimizer: torch.optim.Optimizer,
    extra: dict[str, Any],
    keep: int,
) -> list[dict[str, Any]]:
    candidate_id = f"it{int(extra['iteration'])}_{float(val_mse):.8f}"
    candidate = {
        "id": candidate_id,
        "value": float(val_mse),
        "iteration": int(extra["iteration"]),
        "path": "",
        "summary": extra.get("summary", {}),
    }
    records = list(best_records) + [candidate]
    records.sort(key=lambda row: float(row["value"]))
    top = records[:keep]
    if candidate_id not in {row["id"] for row in top}:
        return top

    tmp_dir = paths["checkpoints"] / "_value_mse_topk_tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    old_files: dict[str, Path] = {}
    for rank, record in enumerate(best_records[:keep], 1):
        source = Path(record.get("path") or best_path(paths, rank))
        if source.exists():
            tmp = tmp_dir / f"{record['id']}.pt"
            shutil.copy2(source, tmp)
            old_files[record["id"]] = tmp

    for rank, record in enumerate(top, 1):
        target = best_path(paths, rank)
        record["path"] = str(target)
        if record["id"] == candidate_id:
            payload = dict(extra)
            payload["best_metric"] = "val_mse"
            payload["best_rank"] = rank
            payload["best_value"] = float(val_mse)
            save_checkpoint(target, model, optimizer, payload)
            print(f"new best val_mse rank{rank}={val_mse:.6f} -> {target}", flush=True)
        elif record["id"] in old_files:
            shutil.copy2(old_files[record["id"]], target)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return top


def write_manifest(paths: dict[str, Path], latest_extra: dict[str, Any], best_records: list[dict[str, Any]]) -> None:
    paths["manifest"].write_text(
        json.dumps(
            {
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "latest": str(paths["latest"]),
                "latest_extra": latest_extra,
                "best_val_mse": best_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    cfg = load_config()
    paths = run_paths(str(cfg["run_name"]))
    for key in ("checkpoints", "logs", "eval"):
        paths[key].mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg["seed"]))
    torch.set_num_threads(int(cfg.get("torch_num_threads", 4)))
    device = choose_device()
    model = PTCGPolicyNet().to(device)
    frozen_model = PTCGPolicyNet().to(device)
    trainable_params = freeze_except_value(model)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    start_iteration, seen_decisions, best_records = load_start_or_resume(model, optimizer, cfg, paths, device)
    freeze_except_value(model)
    set_value_only_mode(model)
    frozen_payload = torch.load(project_path(str(cfg["start_checkpoint"])), map_location=device)
    frozen_model.load_state_dict(frozen_payload["model"])
    frozen_model.eval()
    for param in frozen_model.parameters():
        param.requires_grad_(False)
    opponents = load_mixed_opponents(cfg, frozen_model)
    opponent_text = ", ".join(f"{row['name']}@{float(row.get('weight', 0.0)):.3g}" for row in opponents)
    print(
        f"value MSE mixed-pool device={device} run={cfg['run_name']} "
        f"train_games={cfg['train_games_per_iter']} val_games={cfg['validation_games']} "
        f"params={sum(p.numel() for p in model.parameters()):,} "
        f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
        flush=True,
    )
    print(f"opponent pool: {opponent_text}", flush=True)

    validation = load_or_build_validation(model, frozen_model, opponents, device, cfg, paths)
    val_states = validation["states"]
    val_targets = validation["targets"]
    val_base_mse = baseline_mse(val_targets)
    print(
        f"fixed validation decisions={int(val_targets.numel())} "
        f"baseline_mse={val_base_mse:.6f} stats={validation['stats']}",
        flush=True,
    )

    keep = int(cfg["best_keep"])
    for iteration in range(start_iteration + 1, int(cfg["iterations"]) + 1):
        t0 = time.perf_counter()
        train_seed = int(cfg["seed"]) + iteration * 1000003
        samples, stats = collect_mixed_pool_batch(
            model,
            frozen_model,
            opponents,
            device,
            cfg,
            games=int(cfg["train_games_per_iter"]),
            seed_base=train_seed,
            greedy=bool(cfg.get("train_greedy", False)),
            epsilon=float(cfg.get("exploration_epsilon", 0.0)),
            label=f"train_it{iteration:04d}",
        )
        if not samples:
            raise RuntimeError(f"no self-play samples at iteration {iteration}: {stats}")
        train_states = encode_states(model, samples, device, int(cfg["state_batch_size"]))
        train_targets = samples_to_targets(samples)
        train_base_mse = baseline_mse(train_targets)
        update_metrics = train_value_head(model, optimizer, train_states, train_targets, cfg, device)
        seen_decisions += int(train_targets.numel())

        train_eval_mse = value_mse(model, train_states, train_targets, device, int(cfg["batch_size"]))
        val_mse = ""
        if iteration % int(cfg["eval_every"]) == 0:
            val_mse = value_mse(model, val_states, val_targets, device, int(cfg["batch_size"]))

        elapsed = time.perf_counter() - t0
        latest_extra = {
            "iteration": iteration,
            "seen_decisions": seen_decisions,
            "config": cfg,
            "train_stats": stats,
            "train_mse": train_eval_mse,
            "train_base_mse": train_base_mse,
            "val_mse": val_mse,
            "val_base_mse": val_base_mse,
            "best_val_mse": best_records,
            "summary": {
                "train_mse": train_eval_mse,
                "val_mse": val_mse,
                "val_base_mse": val_base_mse,
            },
        }
        if val_mse != "":
            best_records = update_best_by_mse(
                paths,
                best_records,
                float(val_mse),
                model,
                optimizer,
                latest_extra,
                keep,
            )
            latest_extra["best_val_mse"] = best_records

        if iteration % int(cfg["latest_every"]) == 0:
            save_checkpoint(paths["latest"], model, optimizer, latest_extra)
        write_manifest(paths, latest_extra, best_records)

        row = {
            "iteration": iteration,
            "games": int(cfg["train_games_per_iter"]),
            "decisions": int(train_targets.numel()),
            "seen_decisions": seen_decisions,
            "seconds": round(elapsed, 3),
            "wins": stats["wins"],
            "losses": stats["losses"],
            "p0_wins": stats["p0_wins"],
            "p1_wins": stats["p1_wins"],
            "draws": stats["draws"],
            "errors": stats["errors"],
            "opponent_stats": json.dumps(stats["opponents"], ensure_ascii=False),
            "train_update_mse": update_metrics["train_mse"],
            "train_mse": train_eval_mse,
            "train_base_mse": train_base_mse,
            "val_mse": val_mse,
            "val_base_mse": val_base_mse,
            "updates": update_metrics["updates"],
            "best_val_mse": best_records[0]["value"] if best_records else "",
        }
        append_csv_row(paths["metrics"], row)

        val_text = f" val_mse={float(val_mse):.6f}" if val_mse != "" else ""
        best_text = f" best={float(best_records[0]['value']):.6f}" if best_records else ""
        print(
            f"it={iteration} games={cfg['train_games_per_iter']} decisions={int(train_targets.numel())} "
            f"w/l/d/e={stats['wins']}/{stats['losses']}/{stats['draws']}/{stats['errors']} "
            f"train_mse={train_eval_mse:.6f} train_base={train_base_mse:.6f}"
            f"{val_text} val_base={val_base_mse:.6f}{best_text} "
            f"updates={int(update_metrics['updates'])} sec={elapsed:.1f}",
            flush=True,
        )
        if val_mse != "" and float(val_mse) <= float(cfg["target_val_mse"]):
            save_checkpoint(paths["final"], model, optimizer, latest_extra)
            print(f"target reached: val_mse={float(val_mse):.6f} <= {cfg['target_val_mse']}", flush=True)
            return

    save_checkpoint(
        paths["final"],
        model,
        optimizer,
        {
            "iteration": int(cfg["iterations"]),
            "seen_decisions": seen_decisions,
            "best_val_mse": best_records,
            "config": cfg,
        },
    )
    print(f"wrote {paths['final']}", flush=True)


if __name__ == "__main__":
    main()
