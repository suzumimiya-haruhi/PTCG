import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from ptcg_rl import config
from ptcg_rl.env import BattleRunner
from ptcg_rl.model import PTCGPolicyNet, batch_encoded
from ptcg_rl.opponents import OpponentPool, OpponentSpec
from ptcg_rl.train_utils import append_csv_row, batch_bc_step, choose_device, chunks, ensure_dirs, load_checkpoint, save_checkpoint
from ptcg_rl.utils import set_seed


START_CANDIDATES = [
    config.RUNS_DIR / "large_bc_best_score_rank1.pt",
    config.RUNS_DIR / "large_bc_best_teacher_exact_rank1.pt",
    config.RUNS_DIR / "large_bc_latest.pt",
]
LATEST_PATH = config.RUNS_DIR / "large_dagger_latest.pt"
FINAL_PATH = config.RUNS_DIR / "large_dagger_final.pt"
METRICS_PATH = config.RUNS_DIR / "large_dagger_metrics.csv"
BASELINE_PATH = config.RUNS_DIR / "large_dagger_orig_baseline.json"

BEST_METRICS = {
    "score": True,
    "orig_wr": True,
    "train_pool_wr": True,
    "teacher_exact": True,
    "assisted_orig_wr": True,
    "loss": False,
}


def best_path(metric: str, rank: int) -> Path:
    return config.RUNS_DIR / f"large_dagger_best_{metric}_rank{rank}.pt"


def empty_best_records() -> dict:
    return {metric: [] for metric in BEST_METRICS}


def _record_key(record: dict) -> tuple:
    return (
        int(record.get("iteration", -1)),
        round(float(record.get("value", 0.0)), 12),
        round(float(record.get("teacher_takeover", -1.0)), 4),
    )


def update_topk(metric: str, value: float, best_records: dict, model, optimizer, extra) -> None:
    old_records = list(best_records.get(metric, []))
    candidate = {
        "value": float(value),
        "iteration": int(extra.get("iteration", 0)),
        "teacher_takeover": float(extra.get("teacher_takeover", 0.0)),
        "student_ratio": float(extra.get("student_ratio", 1.0)),
        "seen": int(extra.get("seen", 0)),
    }
    records = old_records + [candidate]
    records.sort(key=lambda r: r["value"], reverse=BEST_METRICS[metric])
    top = records[: config.BEST_KEEP]
    if _record_key(candidate) not in {_record_key(r) for r in top}:
        best_records[metric] = top
        return

    tmp_dir = config.RUNS_DIR / "_large_dagger_best_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    old_paths = {}
    for rank, record in enumerate(old_records[: config.BEST_KEEP], 1):
        path = best_path(metric, rank)
        if path.exists():
            tmp = tmp_dir / f"{metric}_old_rank{rank}.pt"
            shutil.copy2(path, tmp)
            old_paths[_record_key(record)] = tmp

    for rank, record in enumerate(top, 1):
        target = best_path(metric, rank)
        if _record_key(record) == _record_key(candidate):
            payload = dict(extra)
            payload["best_metric"] = metric
            payload["best_rank"] = rank
            payload["best_value"] = float(value)
            payload["best_records"] = {**best_records, metric: top}
            save_checkpoint(target, model, optimizer, payload)
            print(f"new DAgger best {metric} rank{rank}={value:.4f} -> {target}", flush=True)
        elif _record_key(record) in old_paths:
            shutil.copy2(old_paths[_record_key(record)], target)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    best_records[metric] = top


def load_start_checkpoint(model, optimizer, device):
    if LATEST_PATH.exists():
        extra = load_checkpoint(LATEST_PATH, model, optimizer, device=device)
        print(f"resume {LATEST_PATH} extra={extra}", flush=True)
        return extra, "latest"
    start_path = next((p for p in START_CANDIDATES if p.exists()), None)
    if start_path is None:
        raise RuntimeError("No large BC checkpoint found. Run scripts/train_bc.py first.")
    extra = load_checkpoint(start_path, model, device=device)
    print(f"start DAgger from {start_path} extra={extra}", flush=True)
    return {}, "bc"


def build_train_specs() -> list[OpponentSpec]:
    pool = OpponentPool()
    specs = []

    official = sorted([s for s in pool.specs if s.name.startswith("official_")], key=lambda s: s.name)
    specs.extend(OpponentSpec(s.name, s.submission_dir, 1.0) for s in official)

    v4 = config.RULE_ROOT / "archaludon_75_wr_public_submission_v4" / "submission"
    if (v4 / "main.py").exists() and (v4 / "deck.csv").exists():
        specs.append(OpponentSpec("archaludon_75_wr_public_submission_v4", v4, 1.0))

    community_names = {
        "community_ptcg-mega-lucario-ex-v62",
        "community_rule-based-not-psychic-alakazam-best-5th",
    }
    for spec in sorted(pool.specs, key=lambda s: s.name):
        if spec.name in community_names:
            specs.append(OpponentSpec(spec.name, spec.submission_dir, 1.0))

    seen = set()
    unique = []
    for spec in specs:
        key = str(spec.submission_dir.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(spec)
    if len(unique) != 7:
        print(f"warning: expected 7 train opponents, got {len(unique)}", flush=True)
    return unique


def compute_original_baseline(games: int = config.DAGGER_RULE_BASELINE_GAMES) -> dict:
    if BASELINE_PATH.exists():
        data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        if data.get("games") == games:
            print(f"loaded original baseline {BASELINE_PATH}: {data}", flush=True)
            return data

    runner = BattleRunner(opponent_submission=config.ORIG_SUBMISSION)
    wins = losses = draws = errors = 0
    for g in range(games):
        _, info = runner.collect_bc_game(learner_seat=g % 2)
        reward = info.get("reward", 0.0)
        if "error" in info:
            errors += 1
        elif reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1
        if (g + 1) % 100 == 0:
            print(f"baseline original {g+1}/{games} w/l/d/e={wins}/{losses}/{draws}/{errors}", flush=True)
    valid = max(1, wins + losses + draws)
    data = {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "errors": errors,
        "wr": wins / valid,
        "threshold": max(0.0, wins / valid - config.DAGGER_RELEASE_THRESHOLD_MARGIN),
    }
    BASELINE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote original baseline {BASELINE_PATH}: {data}", flush=True)
    return data


def fixed_runner(spec: OpponentSpec) -> BattleRunner:
    runner = BattleRunner(opponent_submission=spec.submission_dir)
    runner.current_opponent = spec
    return runner


def update_stats(stats: dict, name: str, reward: float, has_error: bool) -> None:
    row = stats.setdefault(name, {"wins": 0, "losses": 0, "draws": 0, "errors": 0, "games": 0})
    row["games"] += 1
    if has_error:
        row["errors"] += 1
    elif reward > 0:
        row["wins"] += 1
    elif reward < 0:
        row["losses"] += 1
    else:
        row["draws"] += 1


def winrate_from_stats(row: dict) -> float:
    valid = max(1, row["wins"] + row["losses"] + row["draws"])
    return row["wins"] / valid


def total_stats(stats: dict) -> dict:
    total = {"wins": 0, "losses": 0, "draws": 0, "errors": 0, "games": 0}
    for row in stats.values():
        for key in total:
            total[key] += int(row.get(key, 0))
    return total


def collect_games(runners, model, device, games: int, teacher_takeover: float, train_mode_name: str):
    samples = []
    stats = {}
    student_used = teacher_used = student_invalid = student_match = student_total = 0
    student_ratio = 1.0 - teacher_takeover
    model.eval()
    for g in range(games):
        runner = runners[g % len(runners)]
        game_samples, info = runner.collect_dagger_game(
            model,
            device,
            learner_seat=g % 2,
            student_ratio=student_ratio,
            greedy=config.DAGGER_STUDENT_GREEDY,
        )
        samples.extend(game_samples)
        name = getattr(runner, "display_name", None) or info.get("opponent", train_mode_name)
        update_stats(stats, name, info.get("reward", 0.0), "error" in info)
        student_used += int(info.get("student_used", 0))
        teacher_used += int(info.get("teacher_used", 0))
        student_invalid += int(info.get("student_invalid", 0))
        student_match += int(info.get("student_match", 0))
        student_total += int(info.get("student_total", 0))
    return {
        "samples": samples,
        "stats": stats,
        "student_used": student_used,
        "teacher_used": teacher_used,
        "student_invalid": student_invalid,
        "student_match": student_match,
        "student_total": student_total,
    }


@torch.no_grad()
def eval_teacher_agreement(model, runner, device, games: int) -> dict:
    model.eval()
    total = single = exact = 0
    for g in range(games):
        samples, _ = runner.collect_bc_game(learner_seat=g % 2)
        for batch_samples in chunks(samples, config.DAGGER_BATCH_SIZE):
            if not batch_samples:
                continue
            batch = batch_encoded([s.encoded for s in batch_samples], device)
            logits, _ = model(batch)
            pred = logits.argmax(dim=-1).detach().cpu().tolist()
            for p, sample in zip(pred, batch_samples):
                total += 1
                if len(sample.teacher_action) == 1:
                    single += 1
                    if p == int(sample.teacher_action[0]):
                        exact += 1
    return {"eval_decisions": total, "eval_single_exact": exact / max(1, single)}


@torch.no_grad()
def eval_policy(runners, model, device, games: int) -> dict:
    model.eval()
    stats = {}
    for g in range(games):
        runner = runners[g % len(runners)]
        _, info = runner.collect_policy_game(model, device, learner_seat=g % 2, greedy=True)
        name = getattr(runner, "display_name", None) or info.get("opponent", "unknown")
        update_stats(stats, name, info.get("reward", 0.0), "error" in info)
    total = {"wins": 0, "losses": 0, "draws": 0, "errors": 0, "games": 0}
    for row in stats.values():
        for key in total:
            total[key] += row[key]
    return {"wr": winrate_from_stats(total), "total": total, "by_opp": stats}


def main():
    ensure_dirs()
    set_seed(config.SEED)
    device = choose_device()
    model = PTCGPolicyNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.DAGGER_LR, weight_decay=1e-4)
    print(f"Large DAgger device={device} params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    baseline = compute_original_baseline()
    fixed_threshold = getattr(config, "DAGGER_RELEASE_FIXED_THRESHOLD", None)
    threshold = float(fixed_threshold if fixed_threshold is not None else baseline["threshold"])
    print(
        f"release threshold: original baseline wr={baseline['wr']:.3f}, "
        f"baseline_threshold={baseline['threshold']:.3f}, using={threshold:.3f}",
        flush=True,
    )

    train_specs = build_train_specs()
    print(f"7-agent train pool ready: {len(train_specs)} agents", flush=True)

    original_runner = BattleRunner(opponent_submission=config.ORIG_SUBMISSION)
    original_runner.display_name = "original"
    original_runners = [original_runner]
    train_runners = []
    for spec in train_specs:
        runner = fixed_runner(spec)
        runner.display_name = spec.name
        train_runners.append(runner)
    agreement_runner = BattleRunner(opponent_submission=config.ORIG_SUBMISSION)
    agreement_runner.display_name = "original"

    extra, source = load_start_checkpoint(model, optimizer, device)
    start_it = int(extra.get("iteration", 0)) if source == "latest" else 0
    all_seen = int(extra.get("seen", 0)) if source == "latest" else 0
    teacher_takeover = (
        float(extra.get("teacher_takeover", config.DAGGER_TEACHER_TAKEOVER_START))
        if source == "latest"
        else config.DAGGER_TEACHER_TAKEOVER_START
    )
    best_records = empty_best_records()
    if source == "latest":
        best_records.update(extra.get("best_records", {}))

    for it in range(start_it + 1, config.DAGGER_ITERATIONS + 1):
        teacher_takeover = max(config.DAGGER_TEACHER_TAKEOVER_MIN, round(teacher_takeover, 4))
        student_ratio = 1.0 - teacher_takeover

        orig_batch = collect_games(
            original_runners,
            model,
            device,
            config.DAGGER_GAMES_PER_BATCH,
            teacher_takeover,
            "original",
        )
        train_batch = collect_games(
            train_runners,
            model,
            device,
            config.DAGGER_GAMES_PER_BATCH,
            teacher_takeover,
            "train_pool",
        )
        samples = orig_batch["samples"] + train_batch["samples"]
        random.shuffle(samples)

        metrics = []
        for _ in range(config.DAGGER_EPOCHS):
            for batch_samples in chunks(samples, config.DAGGER_BATCH_SIZE):
                if batch_samples:
                    metrics.append(batch_bc_step(model, optimizer, batch_samples, device))

        all_seen += len(samples)
        mean_loss = sum(m["loss"] for m in metrics) / max(1, len(metrics))
        mean_exact = sum(m["single_exact"] for m in metrics) / max(1, len(metrics))

        orig_total = next(iter(orig_batch["stats"].values()))
        assisted_orig_wr = winrate_from_stats(orig_total)
        train_pool_total = total_stats(train_batch["stats"])
        train_pool_wr = winrate_from_stats(train_pool_total)
        old_teacher_takeover = teacher_takeover
        reduced_teacher_takeover = False
        if assisted_orig_wr >= threshold and teacher_takeover > config.DAGGER_TEACHER_TAKEOVER_MIN:
            teacher_takeover = max(
                config.DAGGER_TEACHER_TAKEOVER_MIN,
                round(teacher_takeover - config.DAGGER_TEACHER_TAKEOVER_STEP, 4),
            )
            reduced_teacher_takeover = teacher_takeover < old_teacher_takeover

        total_student_used = orig_batch["student_used"] + train_batch["student_used"]
        total_teacher_used = orig_batch["teacher_used"] + train_batch["teacher_used"]
        total_student_match = orig_batch["student_match"] + train_batch["student_match"]
        total_student_total = orig_batch["student_total"] + train_batch["student_total"]
        takeover_actual = total_student_used / max(1, total_student_used + total_teacher_used)
        student_match = total_student_match / max(1, total_student_total)

        eval_orig = {}
        eval_pool = {}
        eval_agree = {}
        eval_orig_wr = eval_pool_wr = eval_exact = score = ""
        do_eval = it == 1 or it % config.DAGGER_EVAL_EVERY == 0
        if do_eval:
            eval_orig = eval_policy(original_runners, model, device, config.DAGGER_EVAL_ORIG_GAMES)
            eval_pool = eval_policy(train_runners, model, device, config.DAGGER_EVAL_POOL_GAMES)
            eval_agree = eval_teacher_agreement(model, agreement_runner, device, config.DAGGER_EVAL_AGREEMENT_GAMES)
            eval_orig_wr = eval_orig["wr"]
            eval_pool_wr = eval_pool["wr"]
            eval_exact = eval_agree["eval_single_exact"]
            score = eval_orig_wr * 0.60 + eval_pool_wr * 0.30 + eval_exact * 0.10 - mean_loss * 0.01

        row = {
            "iteration": it,
            "orig_games": config.DAGGER_GAMES_PER_BATCH,
            "train_pool_games": config.DAGGER_GAMES_PER_BATCH,
            "decisions": len(samples),
            "seen": all_seen,
            "teacher_takeover": old_teacher_takeover,
            "next_teacher_takeover": teacher_takeover,
            "student_ratio": student_ratio,
            "takeover_actual": takeover_actual,
            "student_match": student_match,
            "loss": mean_loss,
            "train_single_exact": mean_exact,
            "assisted_orig_wr": assisted_orig_wr,
            "assisted_orig_wld": f"{orig_total['wins']}/{orig_total['losses']}/{orig_total['draws']}/{orig_total['errors']}",
            "train_pool_wr": train_pool_wr,
            "train_pool_wld": (
                f"{train_pool_total['wins']}/{train_pool_total['losses']}/"
                f"{train_pool_total['draws']}/{train_pool_total['errors']}"
            ),
            "release_threshold": threshold,
            "reduced_teacher_takeover": reduced_teacher_takeover,
            "train_pool_stats": repr(train_batch["stats"]),
            "eval_orig_wr": eval_orig_wr,
            "eval_orig_wld": (
                f"{eval_orig.get('total', {}).get('wins', '')}/"
                f"{eval_orig.get('total', {}).get('losses', '')}/"
                f"{eval_orig.get('total', {}).get('draws', '')}/"
                f"{eval_orig.get('total', {}).get('errors', '')}"
            ),
            "eval_train_pool_wr": eval_pool_wr,
            "eval_train_pool_wld": (
                f"{eval_pool.get('total', {}).get('wins', '')}/"
                f"{eval_pool.get('total', {}).get('losses', '')}/"
                f"{eval_pool.get('total', {}).get('draws', '')}/"
                f"{eval_pool.get('total', {}).get('errors', '')}"
            ),
            "eval_single_exact": eval_exact,
            "score": score,
        }
        append_csv_row(METRICS_PATH, row)
        print(
            f"it={it} teacher_takeover={old_teacher_takeover:.2f}->{teacher_takeover:.2f} "
            f"orig_wr={assisted_orig_wr:.3f}/{threshold:.3f} "
            f"train_pool_wr={train_pool_wr:.3f} "
            f"loss={mean_loss:.4f} exact={mean_exact:.3f} "
            f"student_match={student_match:.3f} eval_orig_wr={eval_orig_wr} "
            f"eval_pool_wr={eval_pool_wr} eval_exact={eval_exact} score={score}",
            flush=True,
        )

        extra_payload = {
            "iteration": it,
            "seen": all_seen,
            "teacher_takeover": old_teacher_takeover,
            "student_ratio": student_ratio,
            "best_records": best_records,
            "metrics": row,
        }
        update_topk("loss", mean_loss, best_records, model, optimizer, extra_payload)
        update_topk("assisted_orig_wr", assisted_orig_wr, best_records, model, optimizer, extra_payload)
        if do_eval:
            update_topk("score", float(score), best_records, model, optimizer, extra_payload)
            update_topk("orig_wr", float(eval_orig_wr), best_records, model, optimizer, extra_payload)
            update_topk("train_pool_wr", float(eval_pool_wr), best_records, model, optimizer, extra_payload)
            update_topk("teacher_exact", float(eval_exact), best_records, model, optimizer, extra_payload)

        latest_payload = {
            "iteration": it,
            "seen": all_seen,
            "teacher_takeover": teacher_takeover,
            "student_ratio": 1.0 - teacher_takeover,
            "best_records": best_records,
            "metrics": row,
        }
        if it == 1 or it % config.DAGGER_LATEST_EVERY == 0:
            save_checkpoint(LATEST_PATH, model, optimizer, latest_payload)
            print(f"wrote {LATEST_PATH}", flush=True)

    save_checkpoint(
        FINAL_PATH,
        model,
        optimizer,
        {
            "iteration": config.DAGGER_ITERATIONS,
            "seen": all_seen,
            "teacher_takeover": teacher_takeover,
            "student_ratio": 1.0 - teacher_takeover,
            "best_records": best_records,
        },
    )
    print(f"wrote {FINAL_PATH}", flush=True)


if __name__ == "__main__":
    main()
