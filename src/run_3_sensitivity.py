"""Sensitivity (resumable): sensitivity analysis (one lane per dataset) and the SAA scenario-size study (after TW).
Progress: results/log3_<dataset>.txt. Safe to stop and restart at any time; finished folds are kept."""
import os, sys, subprocess, threading, time
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(os.path.dirname(HERE), "results")
PY = sys.executable

def step(log, args, env_extra=None):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", **(env_extra or {}))
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(args)} {env_extra or ''}\n"); f.flush()
        r = subprocess.run([PY] + args, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, env=env)
        f.write(f"=== exit code {r.returncode}\n")
    return r.returncode == 0

def lane(ds):
    log = os.path.join(RES, f"log3_{ds}.txt")
    steps = [(["sensitivity.py", ds], None)] + ([(["saa.py"], None)] if ds == "TW" else [])
    for args, env in steps:
        print(f"[{ds}] start: {' '.join(args)}", flush=True)
        ok = step(log, args, env)
        print(f"[{ds}] {'done' if ok else 'FAILED (see ' + log + ')'}: {' '.join(args)}", flush=True)
        if not ok: return

if __name__ == "__main__":
    ts = [threading.Thread(target=lane, args=(ds,)) for ds in ("IR", "US", "PL", "TW")]
    for t in ts: t.start(); time.sleep(5)
    for t in ts: t.join()
    open(os.path.join(RES, "ALL_DONE_3.txt"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("ALL DONE (sensitivity). You can close this window.")
