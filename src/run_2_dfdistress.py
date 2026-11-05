"""DF-Distress training (resumable): DF-Distress-M2 and its ablations, then decision scoring. One lane per dataset (4 lanes).
Progress: results/log2_<dataset>.txt. Safe to stop and restart at any time; finished folds are kept."""
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
    log = os.path.join(RES, f"log2_{ds}.txt")
    for args, env in ((["run_dfm2.py", ds], None), (["evaluate_plugin.py", ds], {"PREDS": "_dfm2"})):
        print(f"[{ds}] start: {' '.join(args)}", flush=True)
        ok = step(log, args, env)
        print(f"[{ds}] {'done' if ok else 'FAILED (see ' + log + ')'}: {' '.join(args)}", flush=True)
        if not ok: return

if __name__ == "__main__":
    ts = [threading.Thread(target=lane, args=(ds,)) for ds in ("IR", "US", "PL", "TW")]
    for t in ts: t.start(); time.sleep(5)
    for t in ts: t.join()
    open(os.path.join(RES, "ALL_DONE_2.txt"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("ALL DONE (DF-Distress). You can close this window.")
