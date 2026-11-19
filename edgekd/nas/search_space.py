"""Searchable CNN architecture space.

An architecture is a JSON-serialisable dict:

    {
      "resolution": 32,                       # variable input resolution
      "stem_channels": 24,
      "stages": [                             # variable depth (#stages, #blocks per stage)
        {"blocks": 2, "channels": 32,          # variable filters
         "kernel": 3,                          # 3 / 5 / 7
         "block": "dwsep",                     # conv | dwsep (depthwise-separable) | mbconv
         "expand": 4,                          # only used by mbconv
         "act": "hardswish"},                  # relu | relu6 | hardswish | silu | leaky_relu
        ...
      ]
    }

Every stage starts with a stride-2 block, so the final spatial size is
resolution / 2**num_stages.
"""
import copy
import json
import random

SPACE = {
    "resolution": [24, 28, 32],
    "stem_channels": [16, 24, 32],
    "num_stages": [2, 3, 4],
    "blocks": [1, 2, 3],
    "channels": [16, 24, 32, 48, 64, 96, 128],
    "kernel": [3, 5, 7],
    "block": ["conv", "dwsep", "mbconv"],
    "expand": [2, 4, 6],
    "act": ["relu", "relu6", "hardswish", "silu", "leaky_relu"],
}


def _sample_stage(rng, prev_channels):
    # channels are non-decreasing across stages (standard CNN design prior)
    ch = rng.choice([c for c in SPACE["channels"] if c >= prev_channels] or [SPACE["channels"][-1]])
    return {"blocks": rng.choice(SPACE["blocks"]), "channels": ch, "kernel": rng.choice(SPACE["kernel"]),
            "block": rng.choice(SPACE["block"]), "expand": rng.choice(SPACE["expand"]),
            "act": rng.choice(SPACE["act"])}


def sample_architecture(rng=random):
    arch = {"resolution": rng.choice(SPACE["resolution"]), "stem_channels": rng.choice(SPACE["stem_channels"]),
            "stages": []}
    prev = arch["stem_channels"]
    for _ in range(rng.choice(SPACE["num_stages"])):
        st = _sample_stage(rng, prev)
        arch["stages"].append(st)
        prev = st["channels"]
    return arch


def _sort_channels(arch):
    chans = sorted(s["channels"] for s in arch["stages"])
    for s, c in zip(arch["stages"], chans):
        s["channels"] = c
    return arch


def mutate(arch, rng=random, n_mutations=None):
    """Randomly change 1-2 genes (a global gene, a stage gene, or add/remove a stage)."""
    a = copy.deepcopy(arch)
    for _ in range(n_mutations or rng.choice([1, 2])):
        r = rng.random()
        if r < 0.15:
            a["resolution"] = rng.choice(SPACE["resolution"])
        elif r < 0.25:
            a["stem_channels"] = rng.choice(SPACE["stem_channels"])
        elif r < 0.35:
            n = len(a["stages"])
            if n < max(SPACE["num_stages"]) and (rng.random() < 0.5 or n == min(SPACE["num_stages"])):
                a["stages"].append(_sample_stage(rng, a["stages"][-1]["channels"]))
            elif n > min(SPACE["num_stages"]):
                a["stages"].pop(rng.randrange(n))
        else:
            st = rng.choice(a["stages"])
            key = rng.choice(["blocks", "channels", "kernel", "block", "expand", "act"])
            st[key] = rng.choice(SPACE[key])
    return _sort_channels(a)


def crossover(a, b, rng=random):
    """Uniform crossover: global genes and aligned stages are inherited from either parent."""
    child = {"resolution": rng.choice([a, b])["resolution"],
             "stem_channels": rng.choice([a, b])["stem_channels"]}
    n = rng.choice([len(a["stages"]), len(b["stages"])])
    stages = []
    for i in range(n):
        opts = [p["stages"][i] for p in (a, b) if i < len(p["stages"])]
        stages.append(copy.deepcopy(rng.choice(opts)))
    child["stages"] = stages
    return _sort_channels(child)


def arch_key(arch):
    """Canonical string used to de-duplicate architectures."""
    a = copy.deepcopy(arch)
    for s in a["stages"]:
        if s["block"] != "mbconv":
            s["expand"] = None
    return json.dumps(a, sort_keys=True)
