"""Comparator experiments (resumable): rev_train.py then rev_eval.py, one lane per dataset (4 lanes in parallel).
Progress: results/logR_<dataset>.txt. Safe to stop and restart at any time; finished folds are kept."""
import os, sys, subprocess, threading, time
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(os.path.dirname(HERE), "results")
PY = sys.executable

def step(log, args):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1")
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(args)}\n"); f.flush()
        r = subprocess.run([PY] + args, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, env=env)
        f.write(f"=== exit code {r.returncode}\n")
    return r.returncode == 0

def lane(ds):
    log = os.path.join(RES, f"logR_{ds}.txt")
    for args in (["rev_train.py", ds], ["rev_eval.py", ds]):
        print(f"[{ds}] start: {' '.join(args)}", flush=True)
        ok = step(log, args)
        print(f"[{ds}] {'done' if ok else 'FAILED (see ' + log + ')'}: {' '.join(args)}", flush=True)
        if not ok: return

if __name__ == "__main__":
    sets = sys.argv[1:] or ["IR", "US", "PL", "TW"]
    ts = [threading.Thread(target=lane, args=(ds,)) for ds in sets]
    for t in ts: t.start(); time.sleep(5)
    for t in ts: t.join()
    open(os.path.join(RES, "ALL_DONE_REV.txt"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("ALL DONE (comparator experiments). You can close this window.")
