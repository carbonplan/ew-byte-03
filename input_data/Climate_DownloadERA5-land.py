"""
ERA5-Land Climate Extractor — MIN3P site inputs
=================================================
Fetches ERA5-Land data from Earth Data Hub (EDH) for a user-defined set of
site coordinates (read from a CSV with columns: lat, lon, site_id) and saves
a single Zarr store directly to S3. (Built with support from Claude Code).

More on variable units / accumulation / conventions here: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-timeseries?tab=overview
Full documentation here: https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation
Accumulation convention: https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation#ERA5Land:datadocumentation-accumulationsAccumulations

Variables fetched:
  t2m  – 2-metre temperature                    (K)
  d2m  – 2-metre dewpoint temperature           (K)
  u10  – 10-metre U wind component              (m s⁻¹)
  v10  – 10-metre V wind component              (m s⁻¹)
  str  – surface net thermal (LW) radiation *   (W m⁻²) (converted from J m⁻² daily accumulation)
  ssr  – surface net solar   (SW) radiation *   (W m⁻²) (converted from J m⁻² daily accumulation)
  tp   – total precipitation *                  (mm day⁻¹) (converted from m daily accumulation)
  sp   – surface pressure                       (Pa)

  * str, ssr, and tp are accumulated from 00:00 UTC over steps 1–24 in ERA5-Land.
    Step 24 (valid at 00:00 UTC next day) is the full 24-hour accumulation.
    Set DEACCUMULATE_RADIATION = True to convert to per-hour values via a
    forward difference (the step-1 value at 01:00 UTC is kept as-is since it
    is already a single-hour accumulation).
    - str/ssr: divided by 3600 s hr⁻¹ to convert J m⁻² hr⁻¹ → W m⁻²
    - tp: multiplied by 1000 mm m⁻¹ × 24 hr day⁻¹ to express each hourly
      increment as mm day⁻¹ (a rate, not a 24-hour total)

Output path:
  s3://carbonplan-carbon-removal/ew-workflows-data/min3p/input-data/era5-land/
      era5-land_{TIME_START}_{TIME_END}_{RESOLUTION}.zarr

Auth setup (one-time):
    Add to ~/.netrc:
        machine data.earthdatahub.destine.eu
            login edh
            password <your EDH personal access token>
    chmod 600 ~/.netrc

    AWS credentials with write access to carbonplan-carbon-removal must also
    be configured (e.g. via ~/.aws/credentials or environment variables).

Requirements:
    pip install xarray zarr dask aiohttp s3fs pandas coiled
"""
# %%
import aiohttp
import coiled
import dask
from dask.distributed import Client
import pandas as pd
import xarray as xr
import numpy as np
import s3fs

# ─────────────────────────────────────────────
# 1.  CONFIGURATION
# ─────────────────────────────────────────────

# Path to sites CSV — must have columns: lat, lon, site_id
SITES_CSV = (
    "/Users/tylerkukla/Documents/GitHub/min3p-model-bytes/input_data/raw_data/site-locations.csv"
)

TIME_START = "2001-01-01"
TIME_END   = "2020-12-31"
RESOLUTION = "hourly"   # descriptive label used in the output zarr filename

# Deaccumulate accumulated variables (radiation → W m⁻², precipitation → mm day⁻¹)
DEACCUMULATE_RADIATION = True

OUT_S3_BASE = "s3://carbonplan-carbon-removal/ew-workflows-data/min3p/input-data/era5-land"

atoken = "edh_pat_5006183c5b85f3a21566968ca1710fe2f600be9e71841e2f750adeb4ee818ae35b2595d1d470d641df0560f317133278"
EDH_URL = (
    f"https://edh:{atoken}@data.earthdatahub.destine.eu"
    "/era5/reanalysis-era5-land-no-antartica-v0.zarr"
)

# %% 
# Variable metadata: units, long/short names, and whether the variable is
# a daily accumulation that needs deaccumulation before use
VARIABLE_META = {
    "t2m": {
        "units": "K",
        "long_name": "2 metre temperature",
        "short_name": "t2m",
        "is_accumulation": False,
    },
    "d2m": {
        "units": "K",
        "long_name": "2 metre dewpoint temperature",
        "short_name": "d2m",
        "is_accumulation": False,
    },
    "u10": {
        "units": "m s**-1",
        "long_name": "10 metre U wind component",
        "short_name": "u10",
        "is_accumulation": False,
    },
    "v10": {
        "units": "m s**-1",
        "long_name": "10 metre V wind component",
        "short_name": "v10",
        "is_accumulation": False,
    },
    "str": {
        "units": "J m**-2",
        "long_name": "Surface net thermal radiation",
        "short_name": "str",
        "is_accumulation": True,
        "deaccum_scale": 1 / 3600,   # J m⁻² hr⁻¹ → W m⁻²
        "deaccum_units": "W m**-2",
        "deaccum_processing": "deaccumulated from daily accumulation to per-hour values; divided by 3600 s hr⁻¹ to convert J m⁻² → W m⁻²",
    },
    "ssr": {
        "units": "J m**-2",
        "long_name": "Surface net solar radiation",
        "short_name": "ssr",
        "is_accumulation": True,
        "deaccum_scale": 1 / 3600,   # J m⁻² hr⁻¹ → W m⁻²
        "deaccum_units": "W m**-2",
        "deaccum_processing": "deaccumulated from daily accumulation to per-hour values; divided by 3600 s hr⁻¹ to convert J m⁻² → W m⁻²",
    },
    "tp": {
        "units": "m",
        "long_name": "Total precipitation",
        "short_name": "tp",
        "is_accumulation": True,
        "deaccum_scale": 1000 * 24,  # m hr⁻¹ → mm day⁻¹
        "deaccum_units": "mm day**-1",
        "deaccum_processing": "deaccumulated from daily accumulation to per-hour increments; multiplied by 1000 mm m⁻¹ × 24 hr day⁻¹ to express as mm day⁻¹",
    },
    "sp": {
        "units": "Pa",
        "long_name": "Surface pressure",
        "short_name": "sp",
        "is_accumulation": False,
    },
}

VARIABLES  = list(VARIABLE_META.keys())
ACCUM_VARS = [v for v, m in VARIABLE_META.items() if m["is_accumulation"]]

# Derived output path
out_zarr = (
    f"{OUT_S3_BASE}/"
    f"era5-land_{TIME_START}_{TIME_END}_{RESOLUTION}.zarr"
)

# %%
# ─────────────────────────────────────────────
# 2.  LOAD SITES
# ─────────────────────────────────────────────
# CSV must have columns: lat, lon, site_id

print("Reading sites CSV ...")
if SITES_CSV.startswith("s3://"):
    fs = s3fs.S3FileSystem(anon=False)
    with fs.open(SITES_CSV) as f:
        sites = pd.read_csv(f)
else:
    sites = pd.read_csv(SITES_CSV)

required_cols = {"lat", "lon", "site_id"}
if not required_cols.issubset(sites.columns):
    raise ValueError(
        f"Sites CSV must have columns {required_cols}. Found: {set(sites.columns)}"
    )

print(f"  {len(sites)} sites loaded")
print(sites[["site_id", "lat", "lon"]].to_string(index=False))

lats     = xr.DataArray(sites["lat"].values, dims="site")
lons_180 = xr.DataArray(sites["lon"].values, dims="site")
lons     = lons_180 % 360   # EDH uses 0–360 longitude convention

# %%
# ─────────────────────────────────────────────
# 3.  OPEN THE ZARR STORE (lazy — nothing downloaded yet)
# ─────────────────────────────────────────────

print("Opening ERA5-Land Zarr store (lazy) ...")
ds_full = xr.open_dataset(
    EDH_URL,
    storage_options={
        "client_kwargs": {
            "trust_env": True,  # uses ~/.netrc
            "timeout": aiohttp.ClientTimeout(total=600, connect=60),
        }
    },
    chunks={},
    engine="zarr",
)
print(f"  Variables available: {list(ds_full.data_vars)}")

# %%
# ─────────────────────────────────────────────
# 4.  SUBSET: variables + time range + sites
# ─────────────────────────────────────────────

ds = ds_full[VARIABLES]

ds = ds.sel(
    latitude=lats,
    longitude=lons,
    method="nearest",
).sel(valid_time=slice(TIME_START, TIME_END))

# Attach human-readable site coordinates as non-index coords
ds = ds.assign_coords(
    site_id=("site", sites["site_id"].values.astype(str)),
    target_lat=("site", sites["lat"].values),
    target_lon=("site", sites["lon"].values),
)

# %%
# ─────────────────────────────────────────────
# 5.  DEACCUMULATE RADIATION VARIABLES (optional)
# ─────────────────────────────────────────────
# ERA5-Land runs one 24-hour forecast per day, starting at 00:00 UTC (steps 1–24).
# Accumulations are from 00:00 UTC to the end of the step, so:
#   step  1 → valid 01:00 UTC: 1-hour accumulation  ← first step (already hourly)
#   step 23 → valid 23:00 UTC: 23-hour accumulation
#   step 24 → valid 00:00 UTC next day: full 24-hour accumulation
# For hourly values: diff(t) = accum(t) - accum(t-1) works for all steps except
# step 1 (01:00 UTC), where the raw value IS the single-hour accumulation.
# Reference: https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation
#            #ERA5Land:datadocumentation-accumulationsAccumulations

if DEACCUMULATE_RADIATION and ACCUM_VARS:
    print(f"Deaccumulating variables: {ACCUM_VARS}")
    for var in ACCUM_VARS:
        da = ds[var]
        meta = VARIABLE_META[var]
        # diff() drops the very first timestamp (produces N-1 steps);
        # reindex back to the original coords so shapes align.
        da_diff = da.diff("valid_time", label="upper")
        da_diff = da_diff.reindex(valid_time=da.valid_time)
        # At 01:00 UTC (step 1 of each day) the raw value is already a
        # single-hour accumulation; diff would subtract the full previous
        # day's accumulated value, so replace with the raw value instead.
        is_step1 = da["valid_time.hour"] == 1
        da_deaccum = da_diff.where(~is_step1, other=da)
        ds[var] = da_deaccum * meta["deaccum_scale"]

# %%
# ─────────────────────────────────────────────
# 6.  ATTACH VARIABLE AND DATASET METADATA
# ─────────────────────────────────────────────

for var, meta in VARIABLE_META.items():
    attrs = {
        "units": meta["units"],
        "long_name": meta["long_name"],
        "short_name": meta["short_name"],
    }
    if meta["is_accumulation"] and DEACCUMULATE_RADIATION:
        attrs["units"] = meta["deaccum_units"]
        attrs["processing"] = meta["deaccum_processing"]
    elif meta["is_accumulation"]:
        attrs["processing"] = "raw daily accumulation from 00:00 UTC; deaccumulate before use"
    ds[var].attrs = attrs

ds.attrs = {
    "source": "ERA5-Land via Earth Data Hub (EDH)",
    "edh_dataset": "reanalysis-era5-land-no-antartica-v0",
    "time_start": TIME_START,
    "time_end": TIME_END,
    "temporal_resolution": RESOLUTION,
    "variables": ", ".join(VARIABLES),
    "sites_csv": SITES_CSV,
}

# %%
# ─────────────────────────────────────────────
# 7.  COMPUTE AND WRITE TO S3
# ─────────────────────────────────────────────

if fs.exists(out_zarr):
    print(f"Output already exists at {out_zarr} — overwriting")
    mode = "w"
else:
    mode = "w-"

# Rechunk to uniform sizes; diff+reindex fragments the original EDH chunks.
# 8760 = one year of hourly steps, a natural boundary for this data.
ds = ds.chunk({"valid_time": 8760})

cluster = coiled.Cluster(n_workers=4, region="us-east-1")
client  = Client(cluster)

print(f"\nDownloading and writing to:\n  {out_zarr}")
with dask.config.set({"distributed.scheduler.allowed-failures": 5}):
    ds.to_zarr(
        out_zarr,
        mode=mode,
        storage_options={"anon": False},
        zarr_format=2,
        encoding={"site_id": {"dtype": "str"}},
    )
print("Done.")

client.close()
cluster.close()

# %%
# ─────────────────────────────────────────────
# 8.  OPTIONAL: READ BACK TO VERIFY
# ─────────────────────────────────────────────

# ds_check = xr.open_zarr(out_zarr, storage_options={"anon": False})
# print(ds_check)
