"""raw item → teacher → replay DB（再開可能・Student を待たない独立プロセス）。

  nohup python -m tb250distill.teacher.produce --db data/replay.sqlite --splits val,test,train,robust \
      >> runs/teacher/produce.out 2>&1 &

--splits は優先順。"split[:上限][@source]" 形式で「その split（の指定 source）の採点済みが上限件数に達するまで」
の段階指定ができる（例 val:200,test:200,train:2000,val,test,train,robust）。@source は '+' 区切りで複数可、
'public' は synth 以外すべて（例 val@public,test@public,train@public,val,test,train,robust =
公開データの val/test → 公開 train → 残り全部の順）。teacher 行が既にある item は飛ばすので、
途中で止めて再実行すれば続きから再開する。採点に失敗した item は teacher_errors に記録して先へ進む
（attempts が --max-attempts に達するまで次回 run で再試行）。llama-server が落ちたら復旧を待つ。
ログ: runs/teacher/produce.log（人間向け）, runs/teacher/progress.jsonl（samples/s・latency・GPU 温度/VRAM）。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import statistics
import sys
import time
from collections import deque
from pathlib import Path

from tb250distill import replay
from tb250distill.teacher import server as srv
from tb250distill.teacher.scorer import Scorer, ScorerConfig, ScorerConnectionError, ScorerError

STOP = False


def _on_signal(signum, _frame):
    global STOP
    STOP = True


def gpu_stats(device_id: str = "0x743f") -> dict:
    """RX 6400 の温度・VRAM・使用率を sysfs から読む（取れない項目は省略）。"""
    out: dict = {}
    base = Path("/sys/class/drm")
    try:
        cards = sorted(base.glob("card[0-9]*"))
    except Exception:
        return out
    for c in cards:
        d = c / "device"
        try:
            if (d / "vendor").read_text().strip() != "0x1002" or (d / "device").read_text().strip().lower() != device_id.lower():
                continue
        except Exception:
            continue

        def rd(p, scale=1.0):
            try:
                return float(Path(p).read_text().strip()) / scale
            except Exception:
                return None

        out["vram_used_mb"] = round((rd(d / "mem_info_vram_used") or 0) / 1048576, 1)
        out["busy_pct"] = rd(d / "gpu_busy_percent")
        for h in (d / "hwmon").glob("hwmon*"):
            t = rd(h / "temp1_input", 1000.0)
            if t is not None:
                out["temp_c"] = t
            p = rd(h / "power1_average", 1e6)
            if p is not None:
                out["power_w"] = round(p, 1)
            f = rd(h / "freq1_input", 1e6)
            if f is not None:
                out["sclk_mhz"] = round(f, 0)
        break
    return out


def parse_stages(spec: str) -> list[tuple]:
    """'split[:cap][@src1+src2]' のカンマ区切り -> [(split, cap|None, sources|None), ...]。"""
    stages = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        head, _, srcs = part.partition("@")
        name, _, cap = head.partition(":")
        if name not in replay.SPLITS:
            raise SystemExit(f"unknown split: {name}")
        sources = [x for x in srcs.split("+") if x] or None
        for x in sources or ():
            if x != "public" and x not in replay.SOURCES:
                raise SystemExit(f"unknown source: {x}")
        stages.append((name, int(cap) if cap else None, sources))
    return stages


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


class Producer:
    def __init__(self, args, log: logging.Logger):
        self.args = args
        self.log = log
        self.run_dir = Path(args.log_dir)
        self.conn = replay.connect(args.db)
        cfg = ScorerConfig(base_url=args.url, max_prompt_tokens=args.max_prompt_tokens, n_probs=args.n_probs, perms=args.perms)
        self.scorer = Scorer(cfg)
        self.skipped: set[int] = set()
        self.session_done = 0
        self.session_err = 0
        self.t_start = time.time()
        self.win: deque = deque(maxlen=200)  # (t_finish, latency_ms)
        self.last_report = 0.0
        self.pending_commit = 0

    def wait_for_server(self) -> bool:
        waited, delay = 0.0, 2.0
        while not STOP and waited < self.args.server_wait:
            if srv.health(self.args.url):
                return True
            if self.args.start_server and waited == 0.0:
                try:
                    srv.start(srv.ServerConfig(max_prompt_tokens=self.args.max_prompt_tokens, port=self.args.port), wait=120)
                    continue
                except Exception as e:
                    self.log.error("server start failed: %s", e)
            time.sleep(delay)
            waited += delay
            delay = min(delay * 1.5, 30.0)
        return srv.health(self.args.url)

    def report(self, force: bool = False):
        now = time.time()
        if not force and now - self.last_report < self.args.report_every:
            return
        self.last_report = now
        el = now - self.t_start
        lats = [l for _, l in self.win]
        win_rate = None
        if len(self.win) >= 2 and self.win[-1][0] > self.win[0][0]:
            win_rate = (len(self.win) - 1) / (self.win[-1][0] - self.win[0][0])
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_s": round(el, 1), "session_done": self.session_done,
            "session_errors": self.session_err, "samples_per_s_session": round(self.session_done / el, 3) if el > 0 else None,
            "samples_per_s_window": round(win_rate, 3) if win_rate else None,
            "latency_ms_mean": round(statistics.fmean(lats), 1) if lats else None,
            "latency_ms_p50": round(pct(lats, 0.5), 1) if lats else None,
            "latency_ms_p95": round(pct(lats, 0.95), 1) if lats else None,
            "counts": replay.counts(self.conn), "gpu": gpu_stats(self.args.gpu_device_id),
        }
        if self.args.by_source:
            rec["counts_by_source"] = replay.counts_by_source(self.conn)
        with open(self.run_dir / "progress.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        c = rec["counts"]
        tot = sum(v["total"] for v in c.values())
        sc = sum(v["scored"] for v in c.values())
        eta = None
        if rec["samples_per_s_window"]:
            eta = (tot - sc) / rec["samples_per_s_window"] / 60
        self.log.info(
            "scored %d/%d (+%d this run, %d err) %.2f/s win %.2f/s lat mean %s p95 %s ms | gpu %s | eta %s min | %s",
            sc, tot, self.session_done, self.session_err, rec["samples_per_s_session"] or 0, rec["samples_per_s_window"] or 0,
            rec["latency_ms_mean"], rec["latency_ms_p95"], rec["gpu"], None if eta is None else round(eta, 1),
            {k: f"{v['scored']}/{v['total']}" for k, v in c.items()},
        )

    def commit(self, force: bool = False):
        if self.pending_commit and (force or self.pending_commit >= self.args.commit_every):
            self.conn.commit()
            self.pending_commit = 0

    def process(self, it: dict, model: str) -> bool:
        """1 item を採点して保存。サーバ断なら False（再試行させる）。"""
        t0 = time.perf_counter()
        try:
            res = self.scorer.score(it["context"], it["question"], it["candidates"], lang=it["lang"])
            lat = res["latency_ms"]
            replay.write_teacher(self.conn, it["item_id"], model, self.scorer.method, res["logits"], res["probs"],
                                 res["raw"], lat, commit=False)
            self.pending_commit += 1
            self.session_done += 1
            self.win.append((time.time(), lat))
            return True
        except ScorerConnectionError as e:
            self.log.warning("server error on item %s: %s", it["item_id"], e)
            return False
        except Exception as e:  # item 固有の失敗: 記録して先へ
            self.commit(force=True)
            n = replay.write_error(self.conn, it["item_id"], f"{type(e).__name__}: {e}")
            self.skipped.add(it["item_id"])
            self.session_err += 1
            self.log.error("item %s failed (attempt %d, %.0f ms): %s: %s", it["item_id"], n, (time.perf_counter() - t0) * 1000, type(e).__name__, e)
            return True

    def run(self) -> int:
        a = self.args
        if not self.wait_for_server():
            self.log.error("llama-server not healthy at %s", a.url)
            return 3
        model = self.scorer.model_file()
        self.log.info("start: model=%s method=%s db=%s stages=%s max_prompt_tokens=%d", model, self.scorer.method, a.db, a.splits, a.max_prompt_tokens)
        stages = parse_stages(a.splits)
        done_all = False
        while not STOP and not done_all:
            progressed = False
            for split, cap, sources in stages:
                while not STOP:
                    want = a.batch
                    if cap is not None:
                        left = cap - replay.scored_count(self.conn, split, sources)
                        if left <= 0:
                            break
                        want = min(want, left)
                    if a.limit and self.session_done >= a.limit:
                        self.log.info("limit %d reached", a.limit)
                        self.commit(force=True)
                        self.report(force=True)
                        return 0
                    items = replay.pending_items(self.conn, want, [split], a.max_attempts, self.skipped, sources)
                    if not items:
                        break
                    for it in items:
                        if STOP:
                            break
                        while not self.process(it, model):
                            self.commit(force=True)
                            if not self.wait_for_server():
                                self.log.error("server did not come back; exiting")
                                self.report(force=True)
                                return 3
                        progressed = True
                        self.commit()
                        self.report()
                        if a.limit and self.session_done >= a.limit:
                            break
            self.commit(force=True)
            if not a.follow:
                done_all = True
            elif not progressed:
                time.sleep(30)
        self.commit(force=True)
        self.report(force=True)
        self.log.info("finished (stop=%s): %d scored this run, %d errors", STOP, self.session_done, self.session_err)
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/replay.sqlite")
    ap.add_argument("--splits", default="val,test,train,robust", help="優先順。split[:採点済み上限][@source] をカンマ区切り")
    ap.add_argument("--by-source", action="store_true", help="progress.jsonl に source 別の total/scored も記録")
    ap.add_argument("--port", type=int, default=srv.DEFAULT_PORT)
    ap.add_argument("--url", default=None, help="既定 http://127.0.0.1:<port>")
    ap.add_argument("--max-prompt-tokens", type=int, default=256)
    ap.add_argument("--n-probs", type=int, default=50)
    ap.add_argument("--perms", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--commit-every", type=int, default=20)
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="この run で採点する最大件数（0=無制限）")
    ap.add_argument("--follow", action="store_true", help="未処理が無くなっても待ち続け、追加 item を拾う")
    ap.add_argument("--start-server", action="store_true", help="llama-server が落ちていれば起動する")
    ap.add_argument("--server-wait", type=float, default=600.0, help="サーバ復旧待ちの最大秒")
    ap.add_argument("--report-every", type=float, default=30.0)
    ap.add_argument("--log-dir", default=str(srv.RUN_DIR))
    ap.add_argument("--gpu-device-id", default="0x743f")
    args = ap.parse_args(argv)
    args.url = args.url or f"http://127.0.0.1:{args.port}"

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("produce")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(Path(args.log_dir) / "produce.log"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    (Path(args.log_dir) / "produce.pid").write_text(str(os.getpid()))
    (Path(args.log_dir) / "produce_config.json").write_text(json.dumps(vars(args), indent=1))
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    try:
        return Producer(args, log).run()
    finally:
        try:
            (Path(args.log_dir) / "produce.pid").unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    sys.exit(main())
