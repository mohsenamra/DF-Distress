"""Robustness experiments (resumable). Runs the jobs of s3_train.py and s3_eval.py on several cores in parallel.
Progress: results/logS3_<dataset>_<job>.txt and results/run_status_s3.txt. Safe to stop and restart at any time."""
import os, sys, subprocess, threading, queue, time
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(os.path.dirname(HERE), "results")
PY = sys.executable
SETS = ["IR", "US", "PL", "TW"]
TRAIN = ["m16", "rnd300", "ntr120", "strict", "l8", "eps025", "eps1", "mu01", "mu1", "ntr20", "ntr40", "m4"]
WORKERS = int(os.environ.get("S3_WORKERS", max(1, min((os.cpu_count() or 2) - 1, 16))))
lock = threading.Lock()


def status(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    with lock:
        print(line, flush=True)
        with open(os.path.join(RES, "run_status_s3.txt"), "a", encoding="utf-8") as f: f.write(line + "\n")


def job(args, tag):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1")
    log = os.path.join(RES, f"logS3_{tag}.txt")
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(args)}\n"); f.flush()
        r = subprocess.run([PY] + args, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, env=env)
        f.write(f"=== exit code {r.returncode}\n")
    status(f"{'done' if r.returncode == 0 else 'FAILED (see logS3_' + tag + '.txt)'}: {' '.join(args)}")


def run_queue(jobs):
    q = queue.Queue()
    for j in jobs: q.put(j)
    def worker():
        while True:
            try: args, tag = q.get_nowait()
            except queue.Empty: return
            status(f"start: {' '.join(args)}"); job(args, tag)
    ts = [threading.Thread(target=worker) for _ in range(WORKERS)]
    for t in ts: t.start(); time.sleep(2)
    for t in ts: t.join()


if __name__ == "__main__":
    status(f"robustness experiments, {WORKERS} parallel jobs")
    jobs = [(["s3_eval.py", "env", ds], f"{ds}_env") for ds in SETS]
    for key in TRAIN:
        for ds in SETS: jobs.append((["s3_train.py", ds, key], f"{ds}_{key}"))
    jobs += [(["s3_train.py", "IR", "lag1"], "IR_lag1"), (["s3_train.py", "IR", "negeq"], "IR_negeq")]
    jobs.sort(key=lambda j: (j[0][0] != "s3_train.py" or j[0][2] not in ("m16", "rnd300", "ntr120"),))
    run_queue(jobs)
    run_queue([(["s3_eval.py", "score", ds], f"{ds}_score") for ds in SETS])
    open(os.path.join(RES, "ALL_DONE_S3.txt"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    status("ALL DONE (robustness experiments). You can close this window.")
