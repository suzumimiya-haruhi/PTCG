import random
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
from ptcg_rl.opponents import OpponentPool
from ptcg_rl.train_utils import append_csv_row, batch_bc_step, choose_device, chunks, ensure_dirs, load_checkpoint, save_checkpoint
from ptcg_rl.utils import set_seed


LATEST_PATH = config.RUNS_DIR / "large_bc_latest.pt"
FINAL_PATH = config.RUNS_DIR / "large_bc_final.pt"
METRICS_PATH = config.RUNS_DIR / "large_bc_metrics.csv"
BEST_METRICS = {
    "score": True,
    "teacher_exact": True,
    "loss": False,
}


def best_path(metric: str, rank: int) -> Path:
    return config.RUNS_DIR / f"large_bc_best_{metric}_rank{rank}.pt"


def empty_best_records() -> dict:
    return {metric: [] for metric in BEST_METRICS}


def update_top2(metric: str, value: float, best_records: dict, model, optimizer, extra) -> None:
    candidate = {
        "value": float(value),
        "iteration": int(extra.get("iteration", 0)),
        "seen": int(extra.get("seen", 0)),
    }
    records = list(best_records.get(metric, []))
    records.append(candidate)
    records.sort(key=lambda r: r["value"], reverse=BEST_METRICS[metric])
    top = records[: config.BEST_KEEP]
    qualifies = candidate in top
    best_records[metric] = top
    if not qualifies:
        return

    rank = top.index(candidate) + 1
    if rank == 1 and best_path(metric, 1).exists():
        import shutil

        shutil.copy2(best_path(metric, 1), best_path(metric, 2))
    payload = dict(extra)
    payload["best_metric"] = metric
    payload["best_rank"] = rank
    payload["best_value"] = float(value)
    payload["best_records"] = best_records
    save_checkpoint(best_path(metric, rank), model, optimizer, payload)
    print(f"new BC best {metric} rank{rank}={value:.4f} -> {best_path(metric, rank)}", flush=True)


@torch.no_grad()
def eval_teacher_agreement(model, runner, device, games: int) -> dict:
    model.eval()
    total = 0
    single = 0
    exact = 0
    rewards = []
    for g in range(games):
        samples, info = runner.collect_bc_game(learner_seat=g % 2)
        rewards.append(info.get("reward", 0.0))
        for batch_samples in chunks(samples, config.BC_BATCH_SIZE):
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
    return {
        "eval_decisions": total,
        "eval_single_exact": exact / max(1, single),
        "eval_reward": sum(rewards) / max(1, len(rewards)),
    }


def main():
    ensure_dirs()
    set_seed(config.SEED)
    device = choose_device()
    model = PTCGPolicyNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.BC_LR, weight_decay=1e-4)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Large BC device={device} params={param_count:,}", flush=True)

    start_it = 0
    all_seen = 0
    best_records = empty_best_records()
    if LATEST_PATH.exists():
        extra = load_checkpoint(LATEST_PATH, model, optimizer, device=device)
        start_it = int(extra.get("iteration", 0))
        all_seen = int(extra.get("seen", 0))
        best_records.update(extra.get("best_records", {}))
        print(f"resume {LATEST_PATH} extra={extra}", flush=True)

    opponent_pool = OpponentPool()
    print("opponent pool:")
    print(opponent_pool.summary(), flush=True)
    runner = BattleRunner(opponent_pool=opponent_pool)
    eval_runner = BattleRunner(opponent_pool=OpponentPool())

    for it in range(start_it + 1, config.BC_ITERATIONS + 1):
        samples = []
        rewards = []
        by_opp = {}
        for g in range(config.BC_GAMES_PER_BATCH):
            game_samples, info = runner.collect_bc_game(learner_seat=g % 2)
            samples.extend(game_samples)
            rewards.append(info.get("reward", 0.0))
            opp = info.get("opponent", "unknown")
            by_opp[opp] = by_opp.get(opp, 0) + 1

        random.shuffle(samples)
        metrics = []
        for _ in range(config.BC_EPOCHS):
            for batch_samples in chunks(samples, config.BC_BATCH_SIZE):
                if batch_samples:
                    metrics.append(batch_bc_step(model, optimizer, batch_samples, device))

        all_seen += len(samples)
        mean_loss = sum(m["loss"] for m in metrics) / max(1, len(metrics))
        mean_exact = sum(m["single_exact"] for m in metrics) / max(1, len(metrics))
        mean_reward = sum(rewards) / max(1, len(rewards))

        eval_metrics = {}
        score = -mean_loss
        if it == 1 or it % config.BC_EVAL_EVERY == 0:
            eval_metrics = eval_teacher_agreement(model, eval_runner, device, config.BC_EVAL_GAMES)
            score = eval_metrics["eval_single_exact"] - 0.01 * mean_loss

        row = {
            "iteration": it,
            "games": config.BC_GAMES_PER_BATCH,
            "decisions": len(samples),
            "seen": all_seen,
            "loss": mean_loss,
            "train_single_exact": mean_exact,
            "train_reward": mean_reward,
            "eval_decisions": eval_metrics.get("eval_decisions", ""),
            "eval_single_exact": eval_metrics.get("eval_single_exact", ""),
            "eval_reward": eval_metrics.get("eval_reward", ""),
            "score": score,
            "opponents": repr(by_opp),
        }
        append_csv_row(METRICS_PATH, row)
        print(
            f"it={it} games={config.BC_GAMES_PER_BATCH} decisions={len(samples)} "
            f"seen={all_seen} loss={mean_loss:.4f} exact={mean_exact:.3f} "
            f"reward={mean_reward:.3f} eval={eval_metrics} score={score}",
            flush=True,
        )

        extra = {"iteration": it, "seen": all_seen, "best_records": best_records, "metrics": row}
        update_top2("loss", mean_loss, best_records, model, optimizer, extra)
        if eval_metrics:
            update_top2("score", float(score), best_records, model, optimizer, extra)
            update_top2("teacher_exact", float(eval_metrics["eval_single_exact"]), best_records, model, optimizer, extra)
        if it == 1 or it % config.BC_LATEST_EVERY == 0:
            extra = {"iteration": it, "seen": all_seen, "best_records": best_records, "metrics": row}
            save_checkpoint(LATEST_PATH, model, optimizer, extra)
            print(f"wrote {LATEST_PATH}", flush=True)

    save_checkpoint(FINAL_PATH, model, optimizer, {"iteration": config.BC_ITERATIONS, "seen": all_seen, "best_records": best_records})
    print(f"wrote {FINAL_PATH}", flush=True)


if __name__ == "__main__":
    main()
