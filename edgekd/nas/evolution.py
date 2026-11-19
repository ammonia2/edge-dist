"""Hardware-aware evolutionary architecture search ((mu + lambda) with
tournament selection, uniform crossover and mutation).

Each candidate is first profiled on CPU (latency / size / peak memory).
Candidates violating the deployment constraints are rejected *before* training,
so no compute is spent on undeployable networks. Deployable candidates are
trained briefly on a CIFAR-10 proxy split and scored by validation accuracy.
"""
import random
import time

from ..utils import count_params, evaluate, train_supervised
from .generator import build_model
from .hardware import check_constraints, profile_model
from .search_space import arch_key, crossover, mutate, sample_architecture


class EvolutionarySearch:
    def __init__(self, train_loader, val_loader, constraints, population=8, generations=4, proxy_epochs=1,
                 lr=0.05, seed=0, log=print):
        self.train, self.val = train_loader, val_loader
        self.constraints = constraints
        self.P, self.G, self.E, self.lr = population, generations, proxy_epochs, lr
        self.rng = random.Random(seed)
        self.cache = {}       # arch_key -> record
        self.history = []     # every evaluated candidate, in order
        self.log = log

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, arch, generation):
        key = arch_key(arch)
        if key in self.cache:
            return self.cache[key]
        model = build_model(arch, 10)
        rec = {"id": len(self.history), "generation": generation, "arch": arch, "params": count_params(model)}
        hw = profile_model(model, arch["resolution"])
        rec.update(latency_ms=hw["latency_ms"], size_mb=hw["size_mb"], peak_mem_mb=hw["peak_mem_mb"])
        violations = check_constraints(hw, self.constraints)
        if violations:
            rec.update(status="rejected", violations="; ".join(violations), val_acc=None, fitness=float("-inf"),
                       train_time_s=0.0)
        else:
            t0 = time.time()
            train_supervised(model, self.train, None, self.E, self.lr, log_name=None)
            acc = evaluate(model, self.val)
            rec.update(status="accepted", violations="", val_acc=acc, fitness=acc, train_time_s=time.time() - t0)
        self.cache[key] = rec
        self.history.append(rec)
        self.log(f"  [gen {generation}] #{rec['id']:02d} {rec['status']:8s} "
                 f"params={rec['params'] / 1e3:6.1f}K lat={rec['latency_ms']:.2f}ms size={rec['size_mb']:.2f}MB "
                 f"mem={rec['peak_mem_mb'] or 0:.1f}MB "
                 + (f"val_acc={rec['val_acc']:.2f}%" if rec["val_acc"] is not None else rec["violations"]))
        return rec

    # ------------------------------------------------------------------ operators
    def _tournament(self, pop, k=3):
        return max(self.rng.sample(pop, min(k, len(pop))), key=lambda r: r["fitness"])

    def _offspring(self, pop):
        if self.rng.random() < 0.5 and len(pop) > 1:
            child = crossover(self._tournament(pop)["arch"], self._tournament(pop)["arch"], self.rng)
            child = mutate(child, self.rng, n_mutations=1)
        else:
            child = mutate(self._tournament(pop)["arch"], self.rng)
        return child

    # ------------------------------------------------------------------ main loop
    def run(self):
        self.log(f"generation 0: sampling {self.P} deployable architectures")
        pop, tries = [], 0
        while len(pop) < self.P and tries < self.P * 6:
            tries += 1
            rec = self.evaluate(sample_architecture(self.rng), 0)
            if rec["status"] == "accepted" and rec not in pop:
                pop.append(rec)
        for g in range(1, self.G + 1):
            self.log(f"generation {g}: best so far {max(r['fitness'] for r in pop):.2f}%")
            children, tries = [], 0
            while len(children) < self.P and tries < self.P * 6:
                tries += 1
                arch = self._offspring(pop)
                if arch_key(arch) in self.cache:
                    continue
                rec = self.evaluate(arch, g)
                if rec["status"] == "accepted":
                    children.append(rec)
            pop = sorted(pop + children, key=lambda r: r["fitness"], reverse=True)[: self.P]  # (mu + lambda)
        best = max(pop, key=lambda r: r["fitness"])
        return best, self.history
