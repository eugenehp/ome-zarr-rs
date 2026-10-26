"""Generate reference outputs from scipy.ndimage / scikit-image for the Rust port.

Run with the ome-zarr-py environment:  python scripts/gen_ndimage_fixtures.py
"""
import json
import pathlib

import numpy as np
import scipy.ndimage as ndi
from skimage.transform import (
    downscale_local_mean,
    pyramid_gaussian,
    pyramid_laplacian,
    resize,
)

rng = np.random.default_rng(42)
cases = []


def arr(a):
    return {"dtype": str(a.dtype), "shape": list(a.shape), "data": a.ravel().tolist()}


def add(kind, inp, out, **params):
    cases.append({"kind": kind, "input": arr(inp), "output": arr(np.asarray(out)), "params": params})


for shape, out_shape in [((37, 53), (18, 26)), ((5, 64, 41), (5, 32, 20)), ((9, 9), (4, 4)), ((3, 2, 7, 20), (3, 2, 7, 10)), ((6, 6), (3, 3))]:
    for dtype in ["uint8", "uint16", "float32", "int8"]:
        a = rng.poisson(10, size=shape).astype(dtype)
        r = resize(a, out_shape, order=1, mode="reflect", anti_aliasing=True, preserve_range=True)
        add("resize", a, r, output_shape=list(out_shape), order=1, anti_aliasing=True, preserve_range=True)
        r0 = resize(a, out_shape, order=0, mode="reflect", anti_aliasing=False, preserve_range=True)
        add("resize", a, r0, output_shape=list(out_shape), order=0, anti_aliasing=False, preserve_range=True)
        f = np.ceil(np.array(shape) / np.array(out_shape)).astype(int)
        add("local_mean", a, downscale_local_mean(a, tuple(int(x) for x in f)), factors=[int(x) for x in f])
    a = rng.poisson(10, size=shape).astype("uint8")
    z = np.array(out_shape) / np.array(shape)
    add("zoom", a, ndi.zoom(a, z, order=1), zoom=z.tolist(), order=1, mode="constant", grid_mode=False)
    af = rng.normal(size=shape)
    add("zoom", af, ndi.zoom(af, z, order=3), zoom=z.tolist(), order=3, mode="constant", grid_mode=False)
    add("zoom", af, ndi.zoom(af, 1.7, order=2, mode="nearest"), zoom=[1.7] * len(shape), order=2, mode="nearest", grid_mode=False)
    add("zoom", af, ndi.zoom(af, z, order=3, mode="grid-constant", cval=2.0, grid_mode=True), zoom=z.tolist(), order=3, mode="grid-constant", grid_mode=True, cval=2.0)
    for mode in ["reflect", "mirror", "nearest", "constant", "wrap"]:
        for dtype in ["float64", "float32"]:
            g = af.astype(dtype)
            sig = [0.5 + i for i in range(len(shape))]
            add("gaussian", g, ndi.gaussian_filter(g, sig, mode=mode, cval=0.5), sigma=sig, mode=mode, cval=0.5)

img = rng.normal(size=(20, 17))
coords = rng.uniform(-3, 22, size=(2, 200))
for order in [0, 1, 2, 3, 5]:
    for mode in ["constant", "mirror", "reflect", "nearest", "grid-constant", "wrap", "grid-wrap"]:
        out = ndi.map_coordinates(img, coords, order=order, mode=mode, cval=np.nan)
        add("map_coordinates", img, out, coords=coords.tolist(), order=order, mode=mode, cval=None)

a = rng.poisson(30, size=(1, 40, 33)).astype("uint8")
for i, lvl in enumerate(pyramid_gaussian(a, max_layer=3, downscale=2, channel_axis=None)):
    add("pyramid_gaussian", a, lvl, level=i, max_layer=3)
for i, lvl in enumerate(pyramid_laplacian(a, max_layer=2, downscale=2, channel_axis=None)):
    add("pyramid_laplacian", a, lvl, level=i, max_layer=2)

out = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "ndimage.json"
out.write_text(json.dumps(cases, allow_nan=True).replace("NaN", "null"))
print(f"wrote {len(cases)} cases to {out}")
