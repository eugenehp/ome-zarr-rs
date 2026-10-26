"""Export ome-zarr-models' pydantic core schemas and record its validation
errors for a corpus of inputs.

Run with ome-zarr-py installed:  python scripts/gen_pydantic_fixtures.py
Writes src/models/pydantic_schemas.json (embedded by the Rust validator) and
tests/fixtures/pydantic.json (input -> error message or OK).
"""

import copy
import json
import pathlib
import re
import warnings

from pydantic import TypeAdapter, ValidationError

from ome_zarr_models.common.image_label_types import LabelBase
from ome_zarr_models.common.omero import Omero
from ome_zarr_models.v06.coordinate_transforms import AnyTransform, CoordinateSystem
from ome_zarr_models.v06.multiscales import Multiscale
from ome_zarr_models.v06.scene import BaseSceneAttrs, SceneAttrs
from ome_zarr_models.v04.multiscales import Multiscale as MultiscaleV04
from ome_zarr_models.v05.multiscales import Multiscale as MultiscaleV05

ROOT = pathlib.Path(__file__).resolve().parent.parent
TA = TypeAdapter(AnyTransform)
from pydantic import BaseModel  # noqa: E402

from ome_zarr_models.common.coordinate_transformations import Transform as CommonTransform  # noqa: E402


class Transforms(BaseModel):
    """The throwaway model of v04 / v05 `Dataset._ensure_scale_translation`."""

    transforms: list[CommonTransform]


for _m in (Transforms, CoordinateSystem, Multiscale, Omero, LabelBase, SceneAttrs, BaseSceneAttrs, MultiscaleV04, MultiscaleV05):
    _m.model_rebuild()  # forward references: build the real core schema
ENTRIES = {
    "CoordinateSystem": (CoordinateSystem.__pydantic_core_schema__, CoordinateSystem.model_validate),
    "Transform": (TA.core_schema, TA.validate_python),
    "Multiscale": (Multiscale.__pydantic_core_schema__, Multiscale.model_validate),
    "Omero": (Omero.__pydantic_core_schema__, Omero.model_validate),
    "LabelBase": (LabelBase.__pydantic_core_schema__, LabelBase.model_validate),
    "SceneAttrs": (SceneAttrs.__pydantic_core_schema__, SceneAttrs.model_validate),
    "BaseSceneAttrs": (BaseSceneAttrs.__pydantic_core_schema__, BaseSceneAttrs.model_validate),
    "MultiscaleV04": (MultiscaleV04.__pydantic_core_schema__, MultiscaleV04.model_validate),
    "MultiscaleV05": (MultiscaleV05.__pydantic_core_schema__, MultiscaleV05.model_validate),
    "Transforms": (Transforms.__pydantic_core_schema__, Transforms.model_validate),
}


def ref(r):
    return re.sub(r":\d+$", "", r)


def clean(s, defs):
    if isinstance(s, (list, tuple)):
        return [clean(x, defs) for x in s]
    if not isinstance(s, dict):
        return s if isinstance(s, (str, int, float, bool, type(None))) else str(s)
    t = s.get("type")
    if t == "definitions":
        for d in s["definitions"]:
            defs[ref(d["ref"])] = clean(d, defs)
        return clean(s["schema"], defs)
    out = {}
    for k, v in s.items():
        if k in ("metadata", "serialization", "config", "custom_init", "post_init", "root_model", "computed_fields",
                 "serialization_alias", "from_attributes", "cls"):
            continue
        if k == "ref":
            out[k] = ref(v)
        elif k == "schema_ref":
            out[k] = ref(v)
        elif k == "function":
            f = v.get("function") if isinstance(v, dict) else v
            out[k] = f"{f.__module__}.{f.__qualname__}"
        elif k == "discriminator":
            out[k] = v if isinstance(v, str) else "<python type name>"
        elif k == "choices" and isinstance(v, dict):
            out[k] = {str(kk): clean(vv, defs) for kk, vv in v.items()}
        else:
            out[k] = clean(v, defs)
    if t == "model":
        out["cls"] = s["cls"].__name__
    if "ref" in s and t != "definition-ref":
        defs.setdefault(ref(s["ref"]), out)
    return out


schemas = {}
for name, (cs, validate) in ENTRIES.items():
    defs = {}
    sch = clean(cs, defs)
    try:
        validate(5)
    except ValidationError as e:
        title = e.title
    schemas[name] = {"title": title, "schema": sch, "definitions": defs}
(ROOT / "src" / "models" / "pydantic_schemas.json").write_text(json.dumps(schemas, indent=1))

# ---------------------------------------------------------------- corpus
cs = {"name": "physical", "axes": [{"name": "y", "type": "space", "unit": "micrometer"},
                                    {"name": "x", "type": "space", "discrete": False, "longName": "X"}]}
io_ = {"input": {"name": "a"}, "output": {"name": "b", "path": "p"}}
transforms = [
    {"type": "identity", **io_}, {"type": "mapAxis", "mapAxis": [1, 0], **io_},
    {"type": "projectAxis", "createdOutputs": [0], "droppedInputs": [1, 2]},
    {"type": "translation", "translation": [1, 2.5]}, {"type": "scale", "scale": [2, 0.5], **io_},
    {"type": "affine", "affine": [[1, 0, 2], [0, 1, 3]]}, {"type": "affine", "path": "m"},
    {"type": "rotation", "rotation": [[0, -1], [1, 0]]}, {"type": "rotation", "path": "r", **io_},
    {"type": "sequence", "transformations": [{"type": "scale", "scale": [1, 2]}, {"type": "translation", "translation": [3, 4]}]},
    {"type": "displacements", "path": "f", "interpolation": "linear"}, {"type": "coordinates", "path": "c"},
    {"type": "bijection", "forward": {"type": "scale", "scale": [2]}, "inverse": {"type": "scale", "scale": [0.5]}},
    {"type": "byDimension", "transformations": [{"transformation": {"type": "scale", "scale": [2]}, "inputAxes": [0], "outputAxes": [1]}]},
]
ms_ome = json.loads((ROOT / "tests/fixtures/parity/stores/scene.zarr/a/zarr.json").read_text())["attributes"]["ome"]
scene_ome = json.loads((ROOT / "tests/fixtures/parity/stores/scene.zarr/zarr.json").read_text())["attributes"]["ome"]
valid = {
    "CoordinateSystem": [cs],
    "Transform": transforms,
    "Multiscale": [ms_ome["multiscales"][0]],
    "Omero": [ms_ome["omero"]],
    "LabelBase": [{"colors": [{"label-value": 1, "rgba": [1, 2, 3, 255]}, {"label-value": 2, "rgba": [0, 0, 0, 0]}],
                   "properties": [{"label-value": 1, "area": 3}, {"label-value": 2}], "source": {"image": "../.."}, "version": "0.4"}],
    "SceneAttrs": [scene_ome["scene"]],
    "BaseSceneAttrs": [scene_ome],
    "MultiscaleV04": [json.loads((ROOT / "tests/fixtures/parity/stores/coins_0.4.zarr/.zattrs").read_text())["multiscales"][0],
                      json.loads((ROOT / "tests/fixtures/parity/stores/multiscales_metadata_0.4.zarr/.zattrs").read_text())["multiscales"][0]],
    "Transforms": [{"transforms": [{"type": "scale", "scale": [1, 2]}, {"type": "translation"}]}],
    "MultiscaleV05": [json.loads((ROOT / "tests/fixtures/parity/stores/coins_0.5.zarr/zarr.json").read_text())["attributes"]["ome"]["multiscales"][0]],
}
WRONG = [None, 1, 2.5, -3, "x", "", True, [], [1], {}, {"a": 1}]


def mutations(v, depth=0):
    """(description, mutated value) pairs: delete keys, swap values for wrong types."""
    out = []
    if depth > 4:
        return out
    if isinstance(v, dict):
        for k in list(v):
            d = copy.deepcopy(v); del d[k]
            out.append(d)
            for w in WRONG:
                d = copy.deepcopy(v); d[k] = w
                out.append(d)
            for m in mutations(v[k], depth + 1):
                d = copy.deepcopy(v); d[k] = m
                out.append(d)
        d = copy.deepcopy(v); d["extra_key"] = 1
        out.append(d)
    elif isinstance(v, list) and v:
        out.append([])
        for i in range(min(len(v), 2)):
            for w in WRONG[:6]:
                d = copy.deepcopy(v); d[i] = w
                out.append(d)
            for m in mutations(v[i], depth + 1):
                d = copy.deepcopy(v); d[i] = m
                out.append(d)
        out.append(v + [copy.deepcopy(v[0])])
    return out


special = {
    "CoordinateSystem": [{"name": "", "axes": cs["axes"]}, {"name": "a", "axes": [{"name": "x"}, {"name": "x"}]},
                         {"name": "a", "axes": [{"name": "x", "type": "nope"}]}, {"name": "a", "axes": [{"name": None}]},
                         {"name": "a" * 80, "axes": [{"name": "x" * 70}]}, 5, "x", None, [cs]],
    "Transform": [{"type": "bogus"}, {}, {"scale": [1]}, {"type": 5}, {"type": "mapAxis", "mapAxis": [0, 2]},
                  {"type": "mapAxis", "mapAxis": [0, 0]}, {"type": "projectAxis"}, {"type": "projectAxis", "createdOutputs": [1, 1]},
                  {"type": "projectAxis", "droppedInputs": [0, 0]}, {"type": "affine"}, {"type": "affine", "affine": [[1, 2], [3]]},
                  {"type": "rotation"}, {"type": "rotation", "rotation": [[1, 1], [0, 1]]}, {"type": "rotation", "rotation": [[2.0]]},
                  {"type": "scale", "scale": [1], "input": {"name": "a"}}, {"type": "scale", "scale": [1], "output": {"path": "p"}},
                  {"type": "scale", "scale": [1], "input": {}, "output": {}}, {"type": "scale", "scale": ["1", "2.5", "x"]},
                  {"type": "scale", "scale": [True, 1e308]}, {"type": "mapAxis", "mapAxis": [1.0, 0.5]},
                  {"type": "mapAxis", "mapAxis": ["1", "0"]}, {"type": "sequence", "transformations": [{"type": "nope"}]},
                  {"type": "byDimension", "transformations": [{"transformation": {"type": "scale"}}]},
                  {"type": "bijection", "forward": {"type": "scale", "scale": [2]}}, 5, None, [],
                  {"type": "scale", "scale": [1, 2], "name": 5}, {"type": "displacements"}, {"type": "coordinates", "path": 1}],
    "LabelBase": [{"colors": [{"label-value": 1, "rgba": [1, 2, 3, 255]}, {"label-value": 1, "rgba": [1, 2, 3, 255]}]},
                  {"colors": [{"label-value": 1, "rgba": [1, 2, 3, 255]}], "properties": [{"label-value": 2}]},
                  {"colors": [{"label-value": 1.5, "rgba": [1, 2, 3, 256]}]}, {"colors": [{"label-value": 1, "rgba": [1, 2, 3]}]},
                  {"colors": [{"label-value": "1", "rgba": [-1, 2, 3, 4, 5]}]}, {"version": 0.4}, {}],
    "Omero": [{"channels": [{"color": "FF000", "window": {"min": 0, "max": 1, "start": 0, "end": 1}}]},
              {"channels": [{"color": "GGGGGG", "window": {"min": "0", "max": None, "start": 0, "end": 1}}]}, {"channels": None}],
    "MultiscaleV04": [{"axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}], "datasets": [{"path": "0", "coordinateTransformations": t}]}
                      for t in ([{"type": "translation", "translation": [1, 1]}, {"type": "scale", "scale": [1, 1]}],
                                [{"type": "scale", "scale": [1, 1]}, {"type": "translation", "translation": [1, 1, 1]}],
                                [{"type": "scale", "path": "p"}], [{"type": "identity"}], [], [{"type": "scale"}] * 3, "x", None)]
    + [{"axes": "yx", "datasets": []}, {"axes": [{"name": "x", "type": "space"}] * 2, "datasets": []},
       {"axes": [{"name": "t", "type": "time"}, {"name": "c", "type": "channel"}, {"name": "y", "type": "space"}, {"name": "x", "type": "space"}, {"name": "q"}, {"name": "r"}], "datasets": []},
       {"axes": [{"name": "y", "type": "space"}, {"name": "c", "type": "channel"}, {"name": "x", "type": "space"}], "datasets": []},
       {"axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
        "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [2, 2]}]}, {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [1, 1]}]}]},
       {"axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}], "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [2, 2, 2]}]}],
        "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1]}]}],
    "BaseSceneAttrs": [{"version": "0.5", "scene": {}}, {"version": "0.6.dev4", "scene": {}}, {"version": 6, "scene": {}}],
}
cases = []
seen = set()
for name, (_, validate) in ENTRIES.items():
    inputs = []
    for v in valid[name]:
        inputs.append(v)
        inputs.extend(mutations(v))
    inputs.extend(special.get(name, []) + (special["MultiscaleV04"] if name == "MultiscaleV05" else []))
    for inp in inputs:
        key = (name, json.dumps(inp, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                validate(copy.deepcopy(inp))
                cases.append({"entry": name, "input": inp, "ok": True})
            except ValidationError as e:
                cases.append({"entry": name, "input": inp, "ok": False, "error": str(e)})
            except Exception as e:  # noqa: BLE001
                cases.append({"entry": name, "input": inp, "ok": False, "other": f"{type(e).__name__}: {e}"})
(ROOT / "tests" / "fixtures" / "pydantic.json").write_text(json.dumps(cases, allow_nan=False))
import collections  # noqa: E402
print(collections.Counter((c["entry"], "ok" if c["ok"] else ("err" if "error" in c else "other")) for c in cases))
print([c for c in cases if "other" in c][:5])
