"""Record what the ome_zarr CLI prints for argument errors and --help.

Run with ome-zarr-py installed: python scripts/gen_cli_fixtures.py
Writes tests/fixtures/cli_argparse.json: argv -> stdout, stderr, exit code.
"""

import contextlib
import io
import json
import os
import pathlib
import sys

os.environ["COLUMNS"] = "80"
from ome_zarr.cli import main  # noqa: E402

CASES = [
    [], ["-h"], ["--help"], ["frobnicate"], ["--bogus", "info", "x"], ["-V"],
    ["info"], ["info", "-h"], ["info", "a", "b"], ["info", "a", "--nope"], ["info", "a", "b", "--nope", "c"],
    ["download"], ["download", "-h"], ["download", "x", "--output"], ["download", "x", "--out"],
    ["view"], ["view", "-h"], ["view", "x", "--port", "y"], ["view", "x", "--port=abc"],
    ["finder"], ["finder", "-h"],
    ["create"], ["create", "-h"], ["create", "--method=nope", "x"], ["create", "--method", "nope", "x"],
    ["create", "--m", "coins"], ["create", "--f"],
    ["scale"], ["scale", "-h"], ["scale", "a", "b"], ["scale", "a", "b", "c", "--method", "nope"],
    ["scale", "a", "b", "c", "--max_layer", "x"], ["scale", "a", "b", "c", "--m", "zoom"],
    ["scale", "a", "b", "c", "--downscale"],
    ["csv_to_labels"], ["csv_to_labels", "-h"], ["csv_to_labels", "a", "b"],
    ["-v", "-x", "info", "p"], ["-vvq"], ["-vvq", "info"],
]


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    code = 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        sys.argv = ["ome_zarr", *argv]
        try:
            main(argv)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return {"argv": argv, "stdout": out.getvalue(), "stderr": err.getvalue(), "code": code}


results = [run(a) for a in CASES]
dst = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "cli_argparse.json"
dst.write_text(json.dumps(results, indent=1))
for r in results:
    print(r["argv"], r["code"], repr((r["stdout"] + r["stderr"])[:90]))
