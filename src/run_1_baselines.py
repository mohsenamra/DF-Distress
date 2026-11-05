"""Run the remaining experiments locally (resumable). Two parallel lanes; each step skips folds already finished.
  Lane A: DF training IR, US  -> decision scoring of DF predictions IR, US
  Lane B: baseline decision scoring TW (remaining folds) -> DF training PL, TW -> decision scoring PL, TW
Progress is written to results/log_laneA.txt and results/log_laneB.txt. Safe to stop and restart at any time.
"""
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

def lane(name, steps):
    log = os.path.join(RES, f"log_lane{name}.txt")
    for args, env in steps:
        print(f"[lane {name}] start: {' '.join(args)} {env or ''}", flush=True)
        ok = step(log, args, env)
        print(f"[lane {name}] {'done' if ok else 'FAILED (see ' + log + ')'}: {' '.join(args)}", flush=True)
        if not ok: return

A = [(["run_df.py", "IR", "US"], None), (["evaluate_plugin.py", "IR", "US"], {"PREDS": "_df"})]
B = [(["evaluate_plugin.py", "TW"], None), (["run_df.py", "PL", "TW"], None),
     (["evaluate_plugin.py", "PL", "TW"], {"PREDS": "_df"})]

if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    ts = [threading.Thread(target=lane, args=("A", A)), threading.Thread(target=lane, args=("B", B))]
    for t in ts: t.start()
    for t in ts: t.join()
    open(os.path.join(RES, "ALL_DONE.txt"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("ALL DONE. You can close this window.")
