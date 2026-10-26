"""Record `ZarrLocation`'s accessors for relative paths into the parity stores
(run after gen_parity_fixtures.py):  python scripts/gen_location_fixtures.py
"""

import json
import logging
import os
import pathlib

from ome_zarr.io import ZarrLocation, parse_url

logging.disable(logging.WARNING)
fx = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"
os.chdir(fx / "parity" / "stores")
out = {}
for p in ["coins_0.4.zarr", "coins_0.4.zarr/", "./coins_0.5.zarr", "coins_0.4.zarr/labels", "coins_0.5.zarr/labels/coins",
          "plate_0.4.zarr/A/1", "scene.zarr", "nope.zarr", "coins_0.4.zarr/s0", "coins_0.4.zarr/labels/../s0"]:
    loc = ZarrLocation(p)
    r = {"repr": repr(loc), "str": str(loc), "path": loc.path, "basename": loc.basename(), "parts": loc.parts(),
         "version": loc.version, "exists": loc.exists(), "subpath": loc.subpath("x"), "subpath_up": loc.subpath("../y"),
         "subpath_empty": loc.subpath(""), "root_attrs": sorted(loc.root_attrs.keys())}
    if loc.exists():
        r["create_labels"] = repr(loc.create("labels"))
    r["parse_url"] = repr(parse_url(p)) if parse_url(p) else None
    out[p] = r
(fx / "zarr_location.json").write_text(json.dumps({"cwd": os.getcwd(), "cases": out}, indent=1))
