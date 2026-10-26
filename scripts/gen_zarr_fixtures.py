"""Write reference OME-Zarr stores with ome-zarr-py for the Rust parity tests.

    python scripts/gen_zarr_fixtures.py
"""
import pathlib
import shutil

import dask.array as da
import numpy as np
import zarr

from ome_zarr import OMEZarrImage, OMEZarrLabels, OMEZarrMultiscale
from ome_zarr.format import FormatV04, FormatV05
from ome_zarr.writer import (
    write_image,
    write_labels,
    write_plate_metadata,
    write_well_metadata,
)

root = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "py"
if root.exists():
    shutil.rmtree(root)
root.mkdir(parents=True)

rng = np.random.default_rng(0)
img = rng.poisson(10, size=(2, 5, 48, 40)).astype(np.uint8)
lbl = (img > 10).astype(np.uint8)
img16 = rng.poisson(1000, size=(3, 70, 53)).astype(np.uint16)
f32 = rng.normal(size=(37, 41)).astype(np.float32)

# class API, all versions and methods
for v in ["0.4", "0.5", "0.6"]:
    for method in ["resize", "nearest", "local_mean", "zoom"]:
        i = OMEZarrImage(data=img, axes="czyx", scale={"z": 2.0, "y": 0.5, "x": 0.5},
                         axes_units={"y": "micrometer", "x": "micrometer"})
        ms = OMEZarrMultiscale(i, scale_factors=[2, 4], method=method, channel_names=["A", "B"],
                               channel_colors=["FF0000", [0, 255, 0]], contrast_limits=[(0, 20), (1, 30)])
        ms.to_ome_zarr(str(root / f"class_{v}_{method}.zarr"), version=v, overwrite=True)

# chunked dask input (block boundaries matter for downsampling)
d = da.from_array(img16, chunks=(1, 32, 32))
for method in ["resize", "nearest", "local_mean", "zoom"]:
    i = OMEZarrImage(data=d, axes="cyx")
    ms = OMEZarrMultiscale(i, scale_factors=[{"y": 2, "x": 2}, {"y": 4, "x": 3}], method=method)
    ms.to_ome_zarr(str(root / f"chunked_{method}.zarr"), version="0.5", overwrite=True)

# float32, z downsampling
i = OMEZarrImage(data=f32, axes="yx")
OMEZarrMultiscale(i, scale_factors=[2, 4, 8]).to_ome_zarr(str(root / "float32.zarr"), version="0.6", overwrite=True)

# labels via class API
li = OMEZarrImage(data=lbl, axes="czyx", name="lab")
OMEZarrMultiscale(OMEZarrImage(data=img, axes="czyx"), scale_factors=[2],
                  labels=OMEZarrLabels(li, scale_factors=[2])).to_ome_zarr(
    str(root / "with_labels.zarr"), version="0.5", overwrite=True)

# writer API
for fmt in [FormatV04(), FormatV05()]:
    g = zarr.open_group(str(root / f"writer_{fmt.version}.zarr"), mode="w", zarr_format=fmt.zarr_format)
    write_image(img, g, axes="czyx", storage_options={"chunks": (16, 16)}, scale={"y": 0.25, "x": 0.25}, fmt=fmt)
    write_labels(lbl, g, name="cells", axes="czyx", fmt=fmt)
    # plate
    p = zarr.open_group(str(root / f"plate_{fmt.version}.zarr"), mode="w", zarr_format=fmt.zarr_format)
    write_plate_metadata(p, ["A", "B"], ["1", "2", "3"], ["A/1", "A/3", "B/2"], fmt=fmt, name="plate")
    for wp in ["A/1", "A/3", "B/2"]:
        r, c = wp.split("/")
        well = p.require_group(r).require_group(c)
        write_well_metadata(well, ["0", "1"], fmt=fmt)
        for f in range(2):
            write_image(rng.poisson(10, size=(1, 32, 32)).astype(np.uint8), well.require_group(str(f)),
                        axes="cyx", fmt=fmt, scale_factors=[2])

# sharded
g = zarr.open_group(str(root / "sharded.zarr"), mode="w", zarr_format=3)
write_image(img16, g, axes="cyx", storage_options={"chunks": (16, 16), "shards": (32, 32)}, scale_factors=[2])
print("wrote fixtures to", root)

# sample data (coins segmentation) for data.rs parity
from ome_zarr.data import coins as _coins
_img, _lbl = _coins()
g = zarr.open_group(str(root / "coins_reference.zarr"), mode="w", zarr_format=3)
g.create_array("image", data=np.asarray(_img.images[0].data), chunks=(203, 284))
g.create_array("labels", data=np.asarray(_lbl.images[0].data), chunks=(203, 284))
