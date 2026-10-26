"""Record numpy's small dense linear algebra (as numpy's BLAS / LAPACK on
this platform computes it) for tests/linalg_parity.rs.

Run with ome-zarr-py's environment:  python scripts/gen_linalg_fixtures.py
"""

import json
import pathlib

import numpy as np
import transformnd as tnd

rng = np.random.default_rng(7)
cases = []
for k in (1, 2, 3, 4):
    for m in range(1, 10):
        if k == 4 and m % 2:
            continue  # odd row counts with 4 terms: kernel tails differ by machine
        a, b = rng.standard_normal((m, k)), rng.standard_normal((k, k))
        cases.append({"kind": "matmul", "a": a.tolist(), "b": b.tolist(), "out": (a @ b).tolist()})
    for m in (1, 2, 3, 4):
        if k > 3:
            continue
        a, v = rng.standard_normal((m, k)), rng.standard_normal(k)
        cases.append({"kind": "matvec", "a": a.tolist(), "v": v.tolist(), "out": np.dot(a, v).tolist()})
for n in (2, 3, 4):
    for _ in range(20):
        a = rng.standard_normal((n, n))
        cases.append({"kind": "inv", "a": a.tolist(), "out": np.linalg.inv(a).tolist()})
for d in (2, 3):
    for npts in (1, 2, 5, 33):
        lm = rng.standard_normal((d, d)); t = rng.standard_normal(d)
        m = np.eye(d + 1); m[:d, :d] = lm; m[:d, d] = t
        pts = rng.standard_normal((npts, d))
        cases.append({"kind": "apply", "matrix": m.tolist(), "points": pts.tolist(), "out": tnd.transforms.Affine(m).apply(pts).tolist()})
out = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "linalg.json"
out.write_text(json.dumps(cases))
print(len(cases), "cases")
