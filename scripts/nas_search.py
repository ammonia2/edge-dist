"""Hardware-aware evolutionary NAS on CIFAR-10.

    python scripts/nas_search.py [--profile quick|full] [--max-latency-ms 3 --max-size-mb 2 --max-mem-mb 25]

Outputs:
    results/nas_history.csv / .json  - every candidate (config, latency, size, memory, status, accuracy)
    results/nas_best_arch.json       - best deployable architecture
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgekd.config import NAS_CONSTRAINTS, get_profile  # noqa: E402
from edgekd.data import get_loaders  # noqa: E402
from edgekd.nas import EvolutionarySearch  # noqa: E402
from edgekd.utils import RESULTS_DIR, save_json, seed_all, set_threads  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="quick")
    ap.add_argument("--max-latency-ms", type=float, default=NAS_CONSTRAINTS["max_latency_ms"])
    ap.add_argument("--max-size-mb", type=float, default=NAS_CONSTRAINTS["max_size_mb"])
    ap.add_argument("--max-mem-mb", type=float, default=NAS_CONSTRAINTS["max_peak_mem_mb"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    P = get_profile(args.profile); N = P["nas"]
    seed_all(args.seed); set_threads(P["threads"])
    constraints = {"max_latency_ms": args.max_latency_ms, "max_size_mb": args.max_size_mb,
                   "max_peak_mem_mb": args.max_mem_mb}
    print("constraints:", constraints, flush=True)

    train, val, _ = get_loaders("cifar10", P["batch_size"], train_subset=N["proxy_train"], val_split=N["val"],
                                test_subset=1, seed=args.seed)
    search = EvolutionarySearch(train, val, constraints, N["population"], N["generations"], N["proxy_epochs"],
                                N["lr"], seed=args.seed, log=lambda s: print(s, flush=True))
    t0 = time.time()
    best, history = search.run()
    elapsed = time.time() - t0

    rows = [{**{k: v for k, v in r.items() if k != "arch"}, "arch": json.dumps(r["arch"]),
             "resolution": r["arch"]["resolution"], "num_stages": len(r["arch"]["stages"]),
             "depth": sum(s["blocks"] for s in r["arch"]["stages"])} for r in history]
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "nas_history.csv"), index=False)
    save_json({"constraints": constraints, "search_time_s": elapsed, "history": history},
              os.path.join(RESULTS_DIR, "nas_history.json"))
    save_json({"arch": best["arch"], "proxy_val_acc": best["val_acc"], "latency_ms": best["latency_ms"],
               "size_mb": best["size_mb"], "peak_mem_mb": best["peak_mem_mb"], "params": best["params"],
               "constraints": constraints}, os.path.join(RESULTS_DIR, "nas_best_arch.json"))
    n_rej = sum(r["status"] == "rejected" for r in history)
    print(f"\nsearch done in {elapsed / 60:.1f} min: {len(history)} candidates, {n_rej} rejected by constraints")
    print("best:", json.dumps(best["arch"]))
    print(f"proxy val acc {best['val_acc']:.2f}%  latency {best['latency_ms']:.2f}ms  size {best['size_mb']:.2f}MB")


if __name__ == "__main__":
    main()
