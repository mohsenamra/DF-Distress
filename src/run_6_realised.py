"""Data-calibrated environment (Iran): cal_train.py for the six folds with realised outcomes (4 in parallel), then
cal_eval.py. Progress: results/logC_<fold>.txt and results/run_status_cal.txt. Safe to stop and restart."""
import os, sys, subprocess, threading, queue, time
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(os.path.dirname(HERE), "results")
PY = sys.executable; FOLDS = ["1397", "1398", "1399", "1400", "1401", "1402"]
WORKERS = int(os.environ.get("CAL_WORKERS", max(1, min((os.cpu_count() or 2) - 1, 6))))
lock = threading.Lock()


def status(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    with lock:
        print(line, flush=True)
        with open(os.path.join(RES, "run_status_cal.txt"), "a", encoding="utf-8") as f: f.write(line + "\n")


def job(args, tag):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1")
    with open(os.path.join(RES, f"logC_{tag}.txt"), "a", encoding="utf-8") as f:
        r = subprocess.run([PY] + args, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, env=env)
    status(f"{'done' if r.returncode == 0 else 'FAILED (see logC_' + tag + '.txt)'}: {' '.join(args)}")


if __name__ == "__main__":
    status(f"calibrated environment, {WORKERS} parallel jobs")
    q = queue.Queue()
    for f in FOLDS: q.put(f)
    def worker():
        while True:
            try: f = q.get_nowait()
            except queue.Empty: return
            status(f"start: fold {f}"); job(["cal_train.py", f], f)
    ts = [threading.Thread(target=worker) for _ in range(WORKERS)]
    for t in ts: t.start(); time.sleep(2)
    for t in ts: t.join()
    status("start: evaluation"); job(["cal_eval.py"], "eval")
    status("ALL DONE (calibrated environment). You can close this window.")
