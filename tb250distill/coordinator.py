"""学習 run の起動・監視（coordinator）。GPU ごとに別プロセス（setsid で切り離し）で student.train を走らせる。

  python -m tb250distill.coordinator init   --config common_s --seed 0 --out runs/common/init.npz
  python -m tb250distill.coordinator launch --plan common|optimized --data data/tok/<name> [--epochs N] \
        [--gpus "GT 430,GT 710,GT 730"] [--backend cl|np] [--dry-run]
  python -m tb250distill.coordinator status [--plan common|optimized] [--json]
  python -m tb250distill.coordinator eval   --plan common|optimized [--split test] [--robust]

方針（DESIGN.md / 指示書 §11-§18, §21）:
  - plan common   : 全 GPU が同じ preset(common_s)・同一 init npz・同一 seed/batch/lr/データ。run dir = runs/common/<gpu>/。
  - plan optimized: gt430->s430, gt710->s710, gt730->s730（--presets で変更可）。init は preset ごとに seed から決定的に生成して
                    runs/optimized/init_<preset>.npz に保存（既存なら上書きしない）。run dir = runs/optimized/<gpu>/。
  - 子プロセスは setsid + sh ラッパで切り離す（coordinator が終了しても学習は続く）。終了コードは <run_dir>/exit_code に残る。
    stdout/stderr は <run_dir>/stdout.log / stderr.log。pid・コマンドライン・開始時刻は runs/coordinator/<plan>.json に記録。
  - 1 つの run が起動不能・失敗しても他は止めない（子同士は独立。起動時の失敗も run 単位で記録して続行）。
  - TB250_GIT_COMMIT を子に渡す（環境変数があればそれ、git rev-parse HEAD が取れればそれ、無ければ uncommitted-<ソース内容 hash>）。
  - GPU 固定は --backend cl の --device <名前の部分一致>。np backend では device を無視する。
  - 本ファイルは GPU を一切使わない（numpy のみ import）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

from .student import model as M

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GPUS = "GT 430,GT 710,GT 730"
DEFAULT_PRESETS = {"gt430": "s430", "gt710": "s710", "gt730": "s730"}
COMMON_CONFIG = "common_s"


# --------------------------------------------------------------------------------------
# 小物
# --------------------------------------------------------------------------------------

def gpu_key(name):
    """"GT 430" -> "gt430"（run dir 名）。"""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def source_hash(root=REPO_ROOT):
    """tb250distill/ 以下の *.py の相対パス + 内容から作る短い hash（Mac と tb250 で同一ソースなら同じ値）。"""
    h = hashlib.sha256()
    base = os.path.join(root, "tb250distill")
    for dp, dn, fn in os.walk(base):
        dn[:] = sorted(d for d in dn if d != "__pycache__")
        for f in sorted(fn):
            if f.endswith(".py"):
                p = os.path.join(dp, f)
                h.update(os.path.relpath(p, root).encode())
                with open(p, "rb") as fh:
                    h.update(fh.read())
    return h.hexdigest()[:12]


def git_commit_value(root=REPO_ROOT):
    env = os.environ.get("TB250_GIT_COMMIT")
    if env:
        return env.strip()
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "uncommitted-" + source_hash(root)


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return default


def spec_name(spec):
    """preset 名 / JSON ファイル / JSON 文字列 -> init ファイル名に使う短い名前。"""
    if spec in M.PRESETS:
        return spec
    if os.path.isfile(str(spec)):
        return os.path.splitext(os.path.basename(spec))[0]
    return "custom_" + hashlib.sha256(str(spec).encode()).hexdigest()[:8]


def make_init(spec, seed, out, overwrite=False):
    """seed から決定的に初期 weight を作って npz へ。既存なら（overwrite=False）そのまま使う。戻り値 (path, created, cfg)。"""
    if os.path.exists(out) and not overwrite:
        cfg, _, meta = M.load_params_npz(out)
        want = M.get_config(spec)
        if (cfg.vocab, cfg.emb, cfg.hidden, cfg.layers) != (want.vocab, want.emb, want.hidden, want.layers):
            raise SystemExit(f"{out} は既存だが構造が {spec!r} と異なる: {cfg.to_dict()} vs {want.to_dict()}"
                             "（上書きしない。別パスを使うか手動で退避）")
        return out, False, cfg
    cfg = M.get_config(spec)
    params = M.init_params(cfg, seed)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp.npz"
    M.save_params_npz(tmp, cfg, params, {"seed": seed, "spec": str(spec), "created": now_iso(),
                                         "n_params": M.param_count(cfg)})
    os.replace(tmp, out)
    return out, True, cfg


# --------------------------------------------------------------------------------------
# プロセス管理
# --------------------------------------------------------------------------------------

_WRAP = 'rd="$1"; shift; "$@"; echo $? > "$rd/exit_code"'


def spawn_detached(argv, run_dir, env):
    """setsid で切り離して起動。pid は sh ラッパの pid（= セッション leader）。"""
    os.makedirs(run_dir, exist_ok=True)
    ec = os.path.join(run_dir, "exit_code")
    if os.path.exists(ec):
        os.remove(ec)
    out = open(os.path.join(run_dir, "stdout.log"), "ab")
    err = open(os.path.join(run_dir, "stderr.log"), "ab")
    try:
        p = subprocess.Popen(["/bin/sh", "-c", _WRAP, "sh", run_dir, *argv], cwd=REPO_ROOT, env=env,
                             stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
    finally:
        out.close()
        err.close()
    return p.pid


def pid_alive(pid, run_dir=None):
    """exit_code が無く、pid が存在し、zombie でなければ生きている。"""
    if run_dir and os.path.exists(os.path.join(run_dir, "exit_code")):
        return False
    if not pid:
        return False
    try:
        os.waitpid(pid, os.WNOHANG)   # 自分の子ならここで回収（zombie 対策）
    except ChildProcessError:
        pass
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        r = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=5)
        if r.stdout.strip().startswith("Z"):
            return False
    except Exception:  # noqa: BLE001
        pass
    return True


def run_state(rec):
    """plan 記録の 1 run から状態を判定する。"""
    rd = rec.get("run_dir")
    if rec.get("launch_status") != "started":
        return rec.get("launch_status", "unknown")
    if pid_alive(rec.get("pid"), rd):
        return "running"
    ec = os.path.join(rd, "exit_code")
    if os.path.exists(ec):
        try:
            rc = int(open(ec).read().strip())
        except ValueError:
            rc = -1
        return "finished" if rc == 0 else f"failed(rc={rc})"
    return "dead(no exit_code)"


# --------------------------------------------------------------------------------------
# plan / run の組み立て
# --------------------------------------------------------------------------------------

def parse_presets(s):
    if not s:
        return dict(DEFAULT_PRESETS)
    out = {}
    for tok in s.split(","):
        k, v = tok.split("=", 1)
        out[gpu_key(k)] = v.strip()
    return out


def plan_file(runs_dir, plan):
    return os.path.join(runs_dir, "coordinator", f"{plan}.json")


def build_runs(a):
    """plan から run のリスト（gpu, key, run_dir, spec, init, train argv）を作る。init の生成は呼び出し側。"""
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    keys = [gpu_key(g) for g in gpus]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"--gpus に重複がある: {gpus}")
    presets = parse_presets(a.presets)
    runs = []
    for g, k in zip(gpus, keys):
        if a.plan == "common":
            spec = a.common_config
            init = os.path.join(a.runs_dir, "common", "init.npz")
        else:
            if k not in presets:
                raise SystemExit(f"plan optimized: GPU {g!r}（{k}）の preset が無い。--presets \"{k}=s430,...\" で指定する")
            spec = presets[k]
            init = os.path.join(a.runs_dir, "optimized", f"init_{spec_name(spec)}.npz")
        runs.append({"gpu": g, "key": k, "spec": spec, "init": os.path.abspath(init),
                     "run_dir": os.path.abspath(os.path.join(a.runs_dir, a.plan, k))})
    return runs


def train_argv(a, r, lp):
    argv = [sys.executable, "-m", "tb250distill.student.train", "--backend", a.backend,
            "--data", os.path.abspath(a.data), "--run-dir", r["run_dir"], "--config", str(r["spec"]),
            "--init", r["init"], "--batch-size", str(a.batch_size), "--seed", str(a.seed), "--lr", str(a.lr)]
    if a.backend == "cl":
        argv += ["--device", r["gpu"]]
    if lp:
        argv += ["--lp", str(lp)]
    if a.epochs is not None:
        argv += ["--epochs", str(a.epochs)]
    if a.max_steps:
        argv += ["--max-steps", str(a.max_steps)]
    if a.eval_every:
        argv += ["--eval-every", str(a.eval_every)]
    if a.log_every:
        argv += ["--log-every", str(a.log_every)]
    if a.limit_train:
        argv += ["--limit-train", str(a.limit_train)]
    if a.replay_db:
        argv += ["--replay-db", os.path.abspath(a.replay_db)]
    if a.resume:
        argv += ["--resume"]
    if a.extra:
        argv += shlex.split(a.extra)
    return argv


# --------------------------------------------------------------------------------------
# サブコマンド
# --------------------------------------------------------------------------------------

def cmd_init(a):
    path, created, cfg = make_init(a.config, a.seed, a.out, overwrite=False)
    print(f"{'created' if created else 'exists (not overwritten)'}: {path} config={cfg.to_dict()} "
          f"params={M.param_count(cfg)}")
    return 0


def cmd_launch(a):
    commit = git_commit_value()
    runs = build_runs(a)
    shard_ok = M.find_shard(a.data, "train") is not None
    if not shard_ok:
        msg = f"train shard が見つからない: {a.data}"
        if a.dry_run:
            print("WARNING:", msg)
        else:
            raise SystemExit(msg)
    plan_path = plan_file(a.runs_dir, a.plan)
    prev = read_json(plan_path, {}) or {}
    env = dict(os.environ)
    env["TB250_GIT_COMMIT"] = commit
    env["PYTHONPATH"] = REPO_ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONUNBUFFERED", "1")

    rec_runs = {}
    dry_inits = set()
    for r in runs:
        rec = {"gpu": r["gpu"], "key": r["key"], "run_dir": r["run_dir"], "config": str(r["spec"]),
               "init": r["init"], "backend": a.backend, "data": os.path.abspath(a.data)}
        try:
            # 既に生きている run は触らない
            old = (prev.get("runs") or {}).get(r["key"])
            if old and old.get("launch_status") == "started" and pid_alive(old.get("pid"), old.get("run_dir")):
                raise RuntimeError(f"既に実行中 (pid {old.get('pid')})。起動しない")
            if os.path.exists(os.path.join(r["run_dir"], "ckpt", "last.npz")) and not a.resume:
                raise RuntimeError("run dir に既存 checkpoint がある。--resume で再開するか別の --runs-dir を使う")
            # init
            if a.dry_run:
                if not os.path.exists(r["init"]) and r["init"] not in dry_inits:
                    dry_inits.add(r["init"])
                    print(f"[dry-run] init を作る: {r['init']} (spec={r['spec']} seed={a.seed})")
                cfg = M.get_config(r["spec"])
            else:
                _, created, cfg = make_init(r["spec"], a.seed, r["init"])
                rec["init_created"] = created
            lp = cfg.lp
            rec["model"] = cfg.to_dict()
            rec["n_params"] = M.param_count(cfg)
            argv = train_argv(a, r, lp)
            rec["argv"] = argv
            rec["cmdline"] = " ".join(shlex.quote(x) for x in argv)
            rec["env_TB250_GIT_COMMIT"] = commit
            if a.dry_run:
                rec["launch_status"] = "dry-run"
                print(f"[dry-run] {r['gpu']:>7} -> {r['run_dir']}\n    TB250_GIT_COMMIT={commit}\n    {rec['cmdline']}")
            else:
                rec["pid"] = spawn_detached(argv, r["run_dir"], env)
                rec["started_at"] = now_iso()
                rec["started_at_epoch"] = time.time()
                rec["launch_status"] = "started"
                print(f"started {r['gpu']:>7} pid={rec['pid']} run_dir={r['run_dir']}")
        except SystemExit as e:
            rec["launch_status"] = "skipped"
            rec["error"] = f"{e}"
            print(f"SKIP {r['gpu']}: {e}")
        except Exception as e:  # noqa: BLE001  1 つの失敗で他を止めない
            rec["launch_status"] = "skipped"
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"SKIP {r['gpu']}: {rec['error']}")
        rec_runs[r["key"]] = rec

    if a.dry_run:
        return 0
    hist = list(prev.get("history", []))
    final = {}
    for k, rec in rec_runs.items():
        old = (prev.get("runs") or {}).get(k)
        if rec["launch_status"] == "skipped" and old and old.get("launch_status") == "started":
            # 起動できなかった run は前回記録を残す（生きている run の pid を失わない）
            final[k] = dict(old, last_launch_error=rec.get("error"))
            continue
        if old:
            hist.append(old)
        final[k] = rec
    for k, old in (prev.get("runs") or {}).items():
        if k not in final:
            final[k] = old
    plan = {"plan": a.plan, "created_at": now_iso(), "git_commit": commit, "data": os.path.abspath(a.data),
            "runs_dir": os.path.abspath(a.runs_dir), "seed": a.seed, "batch_size": a.batch_size, "lr": a.lr,
            "epochs": a.epochs, "backend": a.backend, "coordinator_argv": sys.argv, "runs": final, "history": hist}
    write_json(plan_path, plan)
    print("plan:", plan_path)
    return 0 if any(r["launch_status"] == "started" for r in rec_runs.values()) else 1


def read_metrics_summary(run_dir):
    """metrics.csv の最終行と、各列の最後の非空値。"""
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.isfile(path):
        return None
    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    last = rows[-1]
    lastv = {}
    for r in rows:
        for k, v in r.items():
            if v not in ("", None):
                lastv[k] = v
    return {"last": last, "last_nonempty": lastv, "n_rows": len(rows)}


def load_plans(runs_dir, plan=None):
    names = [plan] if plan else ["common", "optimized"]
    out = {}
    for n in names:
        d = read_json(plan_file(runs_dir, n))
        if d:
            out[n] = d
    return out


def _f(x, fmt):
    try:
        return format(float(x), fmt)
    except (TypeError, ValueError):
        return "-"


def status_rows(runs_dir, plan=None):
    rows = []
    for pname, plan_d in load_plans(runs_dir, plan).items():
        for key, rec in plan_d["runs"].items():
            st = run_state(rec)
            ms = read_metrics_summary(rec["run_dir"]) if rec.get("run_dir") else None
            cfgd = read_json(os.path.join(rec["run_dir"], "config.json"), {}) if rec.get("run_dir") else {}
            total = ((cfgd.get("plan") or {}).get("total_steps"))
            row = {"plan": pname, "gpu": rec["gpu"], "key": key, "state": st, "pid": rec.get("pid"),
                   "run_dir": rec.get("run_dir"), "total_steps": total, "error": rec.get("error")}
            if ms:
                lv = ms["last_nonempty"]
                row.update({"step": ms["last"].get("step"), "val_kl": lv.get("val_kl"), "val_agree": lv.get("val_agree"),
                            "samples_per_s": lv.get("samples_per_s"), "temp_c": lv.get("temp_c"),
                            "paused_s": ms["last"].get("paused_s")})
            if st.startswith(("failed", "dead")):
                try:
                    with open(os.path.join(rec["run_dir"], "stderr.log"), "rb") as f:
                        f.seek(0, 2)
                        f.seek(max(0, f.tell() - 600))
                        tail = f.read().decode("utf-8", "replace").strip().splitlines()
                    row["stderr_tail"] = tail[-1] if tail else None
                except OSError:
                    pass
            rows.append(row)
    return rows


def cmd_status(a):
    rows = status_rows(a.runs_dir, a.plan)
    if a.json:
        print(json.dumps(rows, indent=1, ensure_ascii=False))
        return 0
    if not rows:
        print("plan 記録が無い（先に launch する）")
        return 0
    hdr = f"{'plan':9} {'gpu':8} {'state':18} {'pid':>7} {'step':>11} {'val_kl':>8} {'val_agree':>9} {'samples/s':>9} {'temp':>5} {'paused_s':>8}"
    print(hdr)
    for r in rows:
        stp = f"{r.get('step', '-')}/{r.get('total_steps') or '?'}"
        print(f"{r['plan']:9} {r['gpu']:8} {r['state']:18} {str(r['pid'] or '-'):>7} {stp:>11} "
              f"{_f(r.get('val_kl'), '.4f'):>8} {_f(r.get('val_agree'), '.4f'):>9} "
              f"{_f(r.get('samples_per_s'), '.1f'):>9} {_f(r.get('temp_c'), '.0f'):>5} {_f(r.get('paused_s'), '.1f'):>8}")
        if r.get("error"):
            print(f"    launch error: {r['error']}")
        if r.get("stderr_tail"):
            print(f"    stderr: {r['stderr_tail']}")
    return 0


def cmd_eval(a):
    plans = load_plans(a.runs_dir, a.plan)
    if not plans:
        raise SystemExit("plan 記録が無い（先に launch する）")
    splits = ["val"] + [s for s in a.split if s != "val"]
    if a.robust and "robust" not in splits:
        splits.append("robust")
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONUNBUFFERED", "1")
    jobs = []
    for pname, plan_d in plans.items():
        for key, rec in plan_d["runs"].items():
            rd = rec.get("run_dir")
            if rec.get("launch_status") != "started":
                print(f"SKIP {pname}/{key}: 起動されていない run")
                continue
            st = run_state(rec)
            if st == "running":
                print(f"SKIP {pname}/{key}: 学習中（GPU を共有しない。終了後に実行）")
                continue
            ck = os.path.join(rd, "ckpt", "best.npz")
            if not os.path.exists(ck):
                ck = os.path.join(rd, "ckpt", "last.npz")
            if not os.path.exists(ck):
                print(f"SKIP {pname}/{key}: checkpoint が無い")
                continue
            cfgd = read_json(os.path.join(rd, "config.json"), {})
            lp = (cfgd.get("data") or {}).get("lp")
            backend = a.backend or rec.get("backend", "np")
            db = a.replay_db or (cfgd.get("args") or {}).get("replay_db")
            ev = os.path.join(rd, "eval.json")
            keep = os.path.join(rd, "eval_train_final.json")
            old = read_json(ev)
            if old is not None and "meta" not in old and not os.path.exists(keep):
                shutil.copy2(ev, keep)   # train.py の最終 weight での eval.json を退避
            argv = [sys.executable, "-m", "tb250distill.student.evaluate", "--backend", backend,
                    "--data", rec.get("data") or plan_d["data"], "--ckpt", ck, "--splits", *splits,
                    "--out", ev, "--dump-preds", os.path.join(rd, "preds.npz")]
            if backend == "cl":
                argv += ["--device", rec["gpu"]]
            if lp:
                argv += ["--lp", str(lp)]
            if db:
                argv += ["--replay-db", os.path.abspath(db)]
            if a.dump_latency_n:
                argv += ["--dump-latency-n", str(a.dump_latency_n)]
            if a.extra:
                argv += shlex.split(a.extra)
            jobs.append((pname, key, rd, argv))
    if a.dry_run:
        for pname, key, rd, argv in jobs:
            print(f"[dry-run] {pname}/{key}: {' '.join(shlex.quote(x) for x in argv)}")
        return 0
    procs = []
    for pname, key, rd, argv in jobs:
        lf = open(os.path.join(rd, "eval.log"), "ab")
        p = subprocess.Popen(argv, cwd=REPO_ROOT, env=env, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT)
        lf.close()
        procs.append((pname, key, rd, p))
        print(f"eval started {pname}/{key} pid={p.pid}")
    rc_all = 0
    for pname, key, rd, p in procs:   # GPU ごと別プロセスで並列実行し、全部待つ
        rc = p.wait()
        print(f"eval {pname}/{key}: rc={rc} -> {os.path.join(rd, 'eval.json')}")
        rc_all = rc_all or rc
    return rc_all


# --------------------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(description="学習 run の起動・監視", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="全 GPU 共通 init を一度だけ作る（既存なら上書きしない）")
    p.add_argument("--config", default=COMMON_CONFIG, help="preset / JSON ファイル / JSON 文字列")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "runs", "common", "init.npz"))
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("launch", help="GPU ごとに学習プロセスを起動（切り離し）")
    p.add_argument("--plan", required=True, choices=["common", "optimized"])
    p.add_argument("--data", required=True, help="data/tok/<name>")
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--gpus", default=DEFAULT_GPUS, help="カンマ区切りの GPU 名（cl は OpenCL デバイス名の部分一致）")
    p.add_argument("--backend", default="cl", choices=["cl", "np"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--replay-db", default=None, help="robust variant 情報の replay.sqlite（train の最終 eval 用）")
    p.add_argument("--common-config", default=COMMON_CONFIG, help="plan common のモデル（preset/JSON）")
    p.add_argument("--presets", default=None, help='plan optimized の GPU->preset（既定 "gt430=s430,gt710=s710,gt730=s730"）')
    p.add_argument("--resume", action="store_true", help="既存 checkpoint から再開")
    p.add_argument("--extra", default="", help="train.py に渡す追加引数（文字列）")
    p.add_argument("--dry-run", action="store_true", help="コマンドだけ表示（ファイルを作らない）")
    p.set_defaults(fn=cmd_launch)

    p = sub.add_parser("status", help="各 run の生死と metrics.csv 最終値")
    p.add_argument("--plan", choices=["common", "optimized"], default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("eval", help="best checkpoint で evaluate.py を GPU ごと別プロセスで実行（eval.json + preds.npz）")
    p.add_argument("--plan", choices=["common", "optimized"], default=None)
    p.add_argument("--split", nargs="+", default=["test"], help="評価 split（val は常に含める。cascade の τ 決定に必要）")
    p.add_argument("--robust", action="store_true", help="robust split も評価")
    p.add_argument("--backend", default=None, choices=["cl", "np"], help="既定は launch 時の backend")
    p.add_argument("--replay-db", default=None)
    p.add_argument("--dump-latency-n", type=int, default=0, help="per-item latency を測る先頭 item 数（0=全件）")
    p.add_argument("--extra", default="")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_eval)
    for sp in sub.choices.values():
        sp.add_argument("--runs-dir", default=os.path.join(REPO_ROOT, "runs"), help="runs ルート")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.runs_dir = os.path.abspath(a.runs_dir)
    if a.cmd == "init" and not os.path.isabs(a.out):
        a.out = os.path.abspath(a.out)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
