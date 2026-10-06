"""意味表現だけの学習（判断の KD/CE を使わない）: SEMFIX スクリーニング用。

  python -m tb250distill.student.sem_only --backend cl --device "GT 730" --data data/tok/pubA --lp 256 \\
      --init runs/common/init.npz --sem-emb data/emb/qwen3_emb_0p6b/pubA_notrunc --sem-cand-weight 1.0 \\
      --sem-batch 128 --sem-loss infonce --max-steps 3000 --run-dir runs/semfix/screen/infonce/gt730

Student は Common-S と同じ（embedding + GRU + head、init は --init）。各 step で train の unique 候補文字列から M 個を層別に取り
（student/semfix.py）、文脈なしの候補 pass -> projection head -> 損失だけで AdamW 更新する（prefix 側・判断 head は一切使わない。
head.* の重みは init のまま動かない）。決定的（sample は (seed, step) だけで決まる。cl は float atomic の非決定性あり）。
checkpoint: ckpt/p000..p100（総 step の 0/10/25/50/75/100%）、ckpt/last（--resume 用）、ckpt/best（= 最終 weight。diag_sem の既定入力）。
val/test の評価（diag_sem）は別プロセス。このスクリプトは評価専用 Teacher 埋め込み（data/emb_eval）を読まない（load_sem_dir のガードあり）。
SIGTERM / SIGINT では last を保存して終了する。
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time

import numpy as np

from . import model as M
from . import semfix as SF
from . import train as T

log = logging.getLogger("tb250.sem_only")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="意味表現だけの学習（SEMFIX スクリーニング）", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--backend", default="np", choices=["np", "cl"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--data", required=True, help="data/tok/<name>（train shard だけ読む）")
    ap.add_argument("--lp", type=int, default=None)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default="common_s")
    ap.add_argument("--init", default=None, help="初期 weight npz（Common-S の runs/common/init.npz）")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--sem-emb", required=True, metavar="DIR")
    ap.add_argument("--sem-dim", type=int, default=None)
    ap.add_argument("--sem-cand-weight", type=float, default=1.0, metavar="LAMBDA", help="損失全体にかける重み λ")
    SF.add_args(ap)
    ap.set_defaults(sem_batch=128)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smi-interval", type=float, default=20.0)
    ap.add_argument("--temp-pause", type=float, default=88.0)
    ap.add_argument("--temp-resume", type=float, default=78.0)
    ap.add_argument("--no-temp-guard", action="store_true")
    ap.add_argument("--np-dtype", default="float32", choices=["float32", "float64"])
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_dir = args.run_dir
    os.makedirs(os.path.join(run_dir, "ckpt"), exist_ok=True)
    last_path = os.path.join(run_dir, "ckpt", "last")
    resuming = args.resume and os.path.exists(last_path + ".npz")
    if (not args.resume) and os.path.exists(last_path + ".npz"):
        raise SystemExit(f"{run_dir} には既存 checkpoint がある。--resume するか別の --run-dir を使う")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.FileHandler(os.path.join(run_dir, "train.log"), mode="a"), logging.StreamHandler(sys.stdout)])
    if args.sem_batch <= 0:
        raise SystemExit("--sem-batch は 1 以上（sem-only は独立ミニバッチだけを使う）")
    lam = float(args.sem_cand_weight)
    if not (math.isfinite(lam) and lam > 0):
        raise SystemExit("--sem-cand-weight は正の有限値")

    tr_path = M.find_shard(args.data, "train", args.lp)
    if tr_path is None:
        raise SystemExit(f"train shard not found in {args.data}")
    tr = M.Shard(tr_path)
    try:
        sem, sem_info = M.load_sem_dir(args.sem_emb, tr.item_id, args.sem_dim)   # emb_eval は拒否される
        sem.check_against(tr)
        hook = SF.build_hook(args, tr, sem, lam, allow_items=True)
    except (ValueError, OSError) as e:
        raise SystemExit(f"--sem-emb/--sem-batch/--sem-loss: {e}")

    if resuming:
        cfg, params, opt_m, opt_v, rmeta = T.load_ckpt(last_path + ".npz")
        if cfg.sem_dim != sem.d:
            raise SystemExit(f"RESUME: checkpoint の sem_dim={cfg.sem_dim} と今回 {sem.d} が違う")
    else:
        if args.init:
            cfg, params, _ = M.load_params_npz(args.init)
        else:
            cfg = M.get_config(args.config)
            params = M.init_params(cfg, args.seed)
        if cfg.sem_dim not in (0, sem.d):
            raise SystemExit(f"--init の sem_dim={cfg.sem_dim} と今回の sem_dim={sem.d} が違う")
        if not cfg.sem_dim:
            cfg.sem_dim = sem.d
            params = {**params, **M.init_sem_params(cfg, args.seed)}   # train.py と同じ（seed 決定的な別 stream）
    if tr.max_token() >= cfg.vocab:
        raise SystemExit(f"shard の最大 token id {tr.max_token()} >= vocab {cfg.vocab}")
    cfg.lp, cfg.lc = tr.lp, tr.lc

    be = M.make_backend(args.backend, args.device, args.np_dtype)
    # 判断経路の buffer は使わないので最小（max_batch=1, max_k=1, max_lp=1）。候補側は max_sem=M 行を確保する
    model = M.Student(be, cfg, 1, 1, train=True, max_lp=1, max_lc=tr.lc, max_sem=hook.sampler.max_n)
    model.set_params(params)
    model.sem_lambda = lam
    model.sem_ext = hook
    total = max(1, int(args.max_steps))
    ck_steps = {}
    for fr in T.CKPT_FRACS:
        ck_steps.setdefault(int(round(total * fr)), []).append(int(round(fr * 100)))
    git = T.git_info()
    step = 0
    wall_prev = 0.0
    if resuming:
        step = int(rmeta["step"])
        wall_prev = float(rmeta.get("wall_s", 0.0))
        model.set_opt(opt_m, opt_v)
        log.info("RESUME from step %d (wall %.1fs)", step, wall_prev)
    gpu = T.SmiMonitor(be, args.smi_interval)
    gpu_info = gpu.info() if hasattr(gpu, "info") else {}
    if not resuming:
        status = T.collect_environment(run_dir, be, args, cfg)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({"mode": "sem_only", "args": vars(args), "model": cfg.to_dict(), "backend": args.backend,
                       "device": getattr(be, "device_name", None), "pci_bus_id": getattr(be, "pci_bus_id", None),
                       "data": {"dir": args.data, "train": tr_path, "n_train": tr.n, "lp": tr.lp, "lc": tr.lc},
                       "plan": {"total_steps": total, "checkpoint_steps": {str(k): v for k, v in ck_steps.items()}},
                       "optimizer": {"name": "AdamW", "lr": args.lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": args.wd,
                                     "clip_grad_norm": args.clip},
                       "sem": {"enabled": True, "cand_weight": lam, "sem_dim": cfg.sem_dim, "emb_dir": args.sem_emb, "info": sem_info,
                               "semfix": hook.describe(), "judge_loss": "none (sem-only)"},
                       "gpu_monitor": gpu_info, "git_commit": git["commit"], "init": args.init, **status}, f, indent=1, ensure_ascii=False)
        if git["diff"]:
            with open(os.path.join(run_dir, "git.diff"), "w") as f:
                f.write(git["diff"])
    log.info("sem-only backend=%s device=%s steps=%d M=%d loss=%s λ=%g pool=%s", args.backend, getattr(be, "device_name", None), total,
             args.sem_batch, "+".join(hook.spec.terms), lam, hook.pool.counts())

    csv_path = os.path.join(run_dir, "metrics.csv")
    cols = ["step", "sem_loss", "gnorm", "step_ms", "temp_c", "wall_s"] + [f"part_{k}" for k in hook.spec.terms]
    new_csv = not (resuming and os.path.exists(csv_path))
    csv_f = open(csv_path, "w" if new_csv else "a", newline="")
    cw = csv.writer(csv_f)
    if new_csv:
        cw.writerow(cols)
    t_start = time.perf_counter()

    def wall():
        return wall_prev + time.perf_counter() - t_start

    def meta_now(pct=None):
        return {"config": cfg.to_dict(), "step": step, "total_steps": total, "pct": pct, "seed": args.seed, "wall_s": wall(),
                "mode": "sem_only", "args": vars(args), "git_commit": git["commit"],
                "dataset_position": {"note": "sample は (seed, step) だけで決まる（状態なし）"}}

    def temp_guard():
        if not gpu.enabled or args.no_temp_guard:
            return
        gpu.poll()
        if gpu.temp is None or gpu.temp < args.temp_pause:
            return
        log.warning("TEMP PAUSE at step %d: %.0fC", step, gpu.temp)
        while True:
            time.sleep(5.0)
            gpu.poll(force=True)
            if gpu.temp is None or gpu.temp <= args.temp_resume:
                break
        log.warning("TEMP RESUME at step %d", step)

    if not resuming:
        if 0 in ck_steps:
            for pct in ck_steps[0]:
                T.save_ckpt(os.path.join(run_dir, "ckpt", f"p{pct:03d}"), model, meta_now(pct))
        T.save_ckpt(last_path, model, meta_now())
    acc = {"loss": 0.0, "n": 0, "t": 0.0, "g": 0.0}
    interrupted = False
    restore = T.install_sigterm_as_interrupt()
    try:
        while step < total:
            temp_guard()
            t0 = time.perf_counter()
            st = SF.sem_only_step(model, args.lr, step + 1, wd=args.wd, clip=args.clip)
            dt = time.perf_counter() - t0
            step += 1
            if not math.isfinite(st["sem"]):
                raise RuntimeError(f"non-finite sem loss at step {step}: {st}")
            acc["loss"] += st["sem"]
            acc["g"] += st["gnorm"]
            acc["n"] += 1
            acc["t"] += dt
            if step % args.log_every == 0 or step == total:
                gpu.poll()
                pm = hook.pop_parts() or {}
                log.info("step %d/%d sem_loss %.4f gnorm %.3f %.1f ms/step temp %s parts %s", step, total, acc["loss"] / acc["n"],
                         acc["g"] / acc["n"], acc["t"] / acc["n"] * 1e3, gpu.temp, {k: round(v, 4) for k, v in pm.items()})
                row = {"step": step, "sem_loss": acc["loss"] / acc["n"], "gnorm": acc["g"] / acc["n"], "step_ms": acc["t"] / acc["n"] * 1e3,
                       "temp_c": gpu.temp, "wall_s": round(wall(), 2), **{f"part_{k}": v for k, v in pm.items()}}
                cw.writerow(["" if row.get(c) is None else (f"{row[c]:.6g}" if isinstance(row.get(c), float) else row.get(c)) for c in cols])
                csv_f.flush()
                acc = {"loss": 0.0, "n": 0, "t": 0.0, "g": 0.0}
            if step in ck_steps and step != 0:
                for pct in ck_steps[step]:
                    T.save_ckpt(os.path.join(run_dir, "ckpt", f"p{pct:03d}"), model, meta_now(pct))
                    log.info("checkpoint p%03d at step %d", pct, step)
            if args.ckpt_every and step % args.ckpt_every == 0 and step != total:
                T.save_ckpt(last_path, model, meta_now())
    except KeyboardInterrupt:
        interrupted = True
        log.info("interrupted at step %d; saving last checkpoint", step)
    finally:
        restore()
    T.save_ckpt(last_path, model, meta_now())
    csv_f.close()
    if not interrupted and step >= total:
        T.save_ckpt(os.path.join(run_dir, "ckpt", "best"), model, meta_now("final") | {"note": "sem-only: best = 最終 weight"})
    summary = {"steps_done": step, "total_steps": total, "interrupted": interrupted, "wall_s": wall(), "vram_max_mb": gpu.vram_max,
               "temp_max_c": gpu.temp_max}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log.info("done: %s", json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
