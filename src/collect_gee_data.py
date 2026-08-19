"""
collect_gee_data.py
Collect Sentinel-2 spectral composites from Google Earth Engine for
Pakistani agricultural field sites and save them to CSV.

Requirements:
  pip install earthengine-api
  earthengine authenticate          # interactive browser auth
  # OR use a service account key:
  # export GEE_KEY_FILE=/path/to/service-account-key.json

Usage:
    python src/collect_gee_data.py --out data/gee_collected.csv

Output CSV columns:
  location_id, location_name, province, season, season_type, crop_type,
  lat, lon, n_images, ndvi, evi, ndwi, ndre, lai, bsi,
  ndvi_std, ndvi_min, ndvi_max, b2, b3, b4, b5, b8, b8a, b11

Collection method:
  Collection : COPERNICUS/S2_SR_HARMONIZED
  Cloud mask : QA60 bit 10 (opaque) and bit 11 (cirrus)
  Composite  : median over season window
  Extraction : reduceRegion with 1 km buffer, scale=10m
  Seasons    : Rabi (Nov 15 prev year – Apr 30), Kharif (May 1 – Nov 14)
  Period     : 2015 – 2024
"""

import os
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Field sites ───────────────────────────────────────────────────────────────
# 116 sites across Punjab (PB), Sindh (SD), Balochistan (BL), KPK (KP)
# Coordinates are district/city centroids — publicly available.
LOCATIONS = [
    # ── Punjab ────────────────────────────────────────────────────────────────
    ("PB01", "Lahore",      "Punjab",      31.5204, 74.3587, "wheat"),
    ("PB02", "Faisalabad",  "Punjab",      31.4504, 73.1350, "cotton"),
    ("PB03", "Multan",      "Punjab",      30.1978, 71.4711, "wheat"),
    ("PB04", "Gujranwala",  "Punjab",      32.1877, 74.1945, "rice"),
    ("PB05", "Rawalpindi",  "Punjab",      33.5651, 73.0169, "wheat"),
    ("PB06", "Bahawalpur",  "Punjab",      29.3956, 71.6722, "cotton"),
    ("PB07", "Sialkot",     "Punjab",      32.4945, 74.5229, "rice"),
    ("PB08", "Sargodha",    "Punjab",      32.0836, 72.6711, "wheat"),
    ("PB09", "Jhang",       "Punjab",      31.2681, 72.3181, "sugarcane"),
    ("PB10", "Sahiwal",     "Punjab",      30.6706, 73.1064, "rice"),
    ("PB11", "Okara",       "Punjab",      30.8107, 73.4597, "wheat"),
    ("PB12", "Pakpattan",   "Punjab",      30.3437, 73.3870, "cotton"),
    ("PB13", "Narowal",     "Punjab",      32.1022, 74.8755, "rice"),
    ("PB14", "Sheikhupura", "Punjab",      31.7167, 73.9850, "sugarcane"),
    ("PB15", "Hafizabad",   "Punjab",      32.0711, 73.6883, "rice"),
    ("PB16", "Chakwal",     "Punjab",      32.9328, 72.8560, "wheat"),
    ("PB17", "Khushab",     "Punjab",      32.2975, 72.3517, "wheat"),
    ("PB18", "Mianwali",    "Punjab",      32.5854, 71.5433, "wheat"),
    ("PB19", "Bhakkar",     "Punjab",      31.6264, 71.0641, "wheat"),
    ("PB20", "Layyah",      "Punjab",      30.9605, 70.9389, "cotton"),
    ("PB21", "DG Khan",     "Punjab",      30.0514, 70.6347, "wheat"),
    ("PB22", "Muzaffargarh","Punjab",      30.0728, 71.1932, "cotton"),
    ("PB23", "Lodhran",     "Punjab",      29.5328, 71.6322, "cotton"),
    ("PB24", "Vehari",      "Punjab",      30.0450, 72.3511, "cotton"),
    ("PB25", "Khanewal",    "Punjab",      30.3017, 71.9322, "wheat"),
    ("PB26", "Toba Tek Singh","Punjab",    30.9736, 72.4819, "sugarcane"),
    ("PB27", "Chiniot",     "Punjab",      31.7208, 72.9783, "wheat"),
    ("PB28", "Nankana Sahib","Punjab",     31.4508, 73.7094, "rice"),
    ("PB29", "Kasur",       "Punjab",      31.1186, 74.4505, "rice"),
    ("PB30", "Attock",      "Punjab",      33.7667, 72.3600, "wheat"),
    ("PB31", "Jhelum",      "Punjab",      32.9425, 73.7258, "wheat"),
    ("PB32", "Gujrat",      "Punjab",      32.5736, 74.0786, "rice"),
    ("PB33", "Mandi Bahauddin","Punjab",   32.5869, 73.4917, "rice"),
    ("PB34", "Hafizabad2",  "Punjab",      32.0711, 73.7500, "wheat"),
    # ── Sindh ─────────────────────────────────────────────────────────────────
    ("SD01", "Karachi",     "Sindh",       24.8607, 67.0011, "cotton"),
    ("SD02", "Hyderabad",   "Sindh",       25.3960, 68.3578, "wheat"),
    ("SD03", "Sukkur",      "Sindh",       27.7052, 68.8574, "wheat"),
    ("SD04", "Larkana",     "Sindh",       27.5580, 68.2122, "rice"),
    ("SD05", "Nawabshah",   "Sindh",       26.2440, 68.4100, "sugarcane"),
    ("SD06", "Mirpurkhas",  "Sindh",       25.5264, 69.0139, "cotton"),
    ("SD07", "Jacobabad",   "Sindh",       28.2769, 68.4514, "wheat"),
    ("SD08", "Shikarpur",   "Sindh",       27.9558, 68.6378, "wheat"),
    ("SD09", "Khairpur",    "Sindh",       27.5295, 68.7592, "rice"),
    ("SD10", "Badin",       "Sindh",       24.6559, 68.8375, "rice"),
    ("SD11", "Thatta",      "Sindh",       24.7464, 67.9228, "cotton"),
    ("SD12", "Dadu",        "Sindh",       26.7311, 67.7764, "wheat"),
    ("SD13", "Sanghar",     "Sindh",       26.0461, 68.9492, "cotton"),
    ("SD14", "Matiari",     "Sindh",       25.5967, 68.4628, "sugarcane"),
    ("SD15", "Umerkot",     "Sindh",       25.3617, 69.7367, "cotton"),
    ("SD16", "Tharparkar",  "Sindh",       24.7161, 70.2439, "wheat"),
    ("SD17", "Sujawal",     "Sindh",       24.5994, 68.0736, "rice"),
    ("SD18", "Jamshoro",    "Sindh",       25.4317, 68.2800, "sugarcane"),
    ("SD19", "Qamber",      "Sindh",       27.2833, 68.0167, "rice"),
    ("SD20", "Kashmor",     "Sindh",       28.4333, 69.5833, "wheat"),
    # ── Balochistan ───────────────────────────────────────────────────────────
    ("BL01", "Quetta",      "Balochistan", 30.1798, 66.9750, "wheat"),
    ("BL02", "Turbat",      "Balochistan", 26.0025, 63.0439, "wheat"),
    ("BL03", "Khuzdar",     "Balochistan", 27.8000, 66.6167, "wheat"),
    ("BL04", "Hub",         "Balochistan", 25.0603, 66.8950, "cotton"),
    ("BL05", "Chaman",      "Balochistan", 30.9214, 66.4508, "wheat"),
    ("BL06", "Sibi",        "Balochistan", 29.5433, 67.8775, "wheat"),
    ("BL07", "Zhob",        "Balochistan", 31.3417, 69.4486, "wheat"),
    ("BL08", "Mastung",     "Balochistan", 29.8000, 66.8500, "wheat"),
    ("BL09", "Nushki",      "Balochistan", 29.5525, 66.0200, "wheat"),
    ("BL10", "Loralai",     "Balochistan", 30.3706, 68.5919, "wheat"),
    ("BL11", "Dera Bugti",  "Balochistan", 29.0333, 69.1667, "wheat"),
    ("BL12", "Kalat",       "Balochistan", 29.0236, 66.5889, "wheat"),
    ("BL13", "Pishin",      "Balochistan", 30.5808, 66.9964, "mango"),
    ("BL14", "Killa Saifullah","Balochistan",30.6928,68.3625,"mango"),
    ("BL15", "Kharan",      "Balochistan", 28.5833, 65.4167, "wheat"),
    ("BL16", "Washuk",      "Balochistan", 27.7667, 64.6500, "wheat"),
    ("BL17", "Panjgur",     "Balochistan", 26.9667, 64.1000, "wheat"),
    ("BL18", "Awaran",      "Balochistan", 26.8333, 65.2500, "cotton"),
    ("BL19", "Musakhel",    "Balochistan", 29.8667, 69.7500, "mango"),
    ("BL20", "Harnai",      "Balochistan", 30.0989, 67.9322, "wheat"),
    ("BL21", "Barkhan",     "Balochistan", 29.9000, 69.5167, "wheat"),
    ("BL22", "Sherani",     "Balochistan", 31.3333, 69.8333, "mango"),
    # ── KPK ───────────────────────────────────────────────────────────────────
    ("KP01", "Peshawar",    "KPK",         34.0151, 71.5249, "wheat"),
    ("KP02", "Mardan",      "KPK",         34.2014, 72.0440, "sugarcane"),
    ("KP03", "Abbottabad",  "KPK",         34.1558, 73.2194, "wheat"),
    ("KP04", "Swat",        "KPK",         35.2219, 72.4258, "rice"),
    ("KP05", "Charsadda",   "KPK",         34.1483, 71.7314, "sugarcane"),
    ("KP06", "Nowshera",    "KPK",         34.0153, 71.9747, "sugarcane"),
    ("KP07", "Kohat",       "KPK",         33.5867, 71.4422, "wheat"),
    ("KP08", "Bannu",       "KPK",         32.9886, 70.5986, "wheat"),
    ("KP09", "Mansehra",    "KPK",         34.3331, 73.1972, "rice"),
    ("KP10", "Haripur",     "KPK",         33.9942, 72.9353, "wheat"),
    ("KP11", "Swabi",       "KPK",         34.1200, 72.4700, "sugarcane"),
    ("KP12", "Malakand",    "KPK",         34.5650, 71.9303, "rice"),
    ("KP13", "Dir Upper",   "KPK",         35.2050, 72.0097, "rice"),
    ("KP14", "Chitral",     "KPK",         35.8517, 71.7861, "wheat"),
    ("KP15", "Shangla",     "KPK",         34.8319, 72.8089, "rice"),
    ("KP16", "Buner",       "KPK",         34.5114, 72.4992, "mango"),
    ("KP17", "Lakki Marwat","KPK",         32.6072, 70.9128, "wheat"),
    ("KP18", "Tank",        "KPK",         32.2186, 70.3781, "wheat"),
    ("KP19", "South Waziristan","KPK",     32.3167, 69.7333, "wheat"),
    ("KP20", "North Waziristan","KPK",     33.0000, 70.0833, "wheat"),
    ("KP21", "Kurram",      "KPK",         33.5667, 70.1000, "wheat"),
    ("KP22", "Karak",       "KPK",         33.1167, 71.0833, "wheat"),
    ("KP23", "Hangu",       "KPK",         33.5319, 71.0592, "mango"),
]

SEASON_WINDOWS = {
    "rabi":   ("11-15", "04-30"),  # sowing Nov → harvest Apr
    "kharif": ("05-01", "11-14"),  # sowing May → harvest Nov
}
YEARS = range(2015, 2025)


def ee_init():
    import ee
    key_file = os.environ.get("GEE_KEY_FILE")
    if key_file and Path(key_file).exists():
        credentials = ee.ServiceAccountCredentials(
            email=None, key_file=key_file)
        ee.Initialize(credentials)
    else:
        ee.Authenticate()
        ee.Initialize(project=os.environ.get("GEE_PROJECT", "your-gcp-project-id"))


def mask_s2_clouds(image):
    import ee
    qa   = image.select("QA60")
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(mask).divide(10000)


def fetch_location(loc, season_type, year):
    """
    Fetch median S2 composite for one location × season × year.
    Returns a dict of spectral values, or None if no clear images found.
    """
    import ee
    loc_id, name, province, lat, lon, crop = loc

    start_raw, end_raw = SEASON_WINDOWS[season_type]
    if season_type == "rabi":
        start = f"{year - 1}-{start_raw}"
        end   = f"{year}-{end_raw}"
        label = f"Rabi_{year}"
    else:
        start = f"{year}-{start_raw}"
        end   = f"{year}-{end_raw}"
        label = f"Kharif_{year}"

    point  = ee.Geometry.Point([lon, lat]).buffer(1000)
    col    = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(point)
                .filterDate(start, end)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
                .map(mask_s2_clouds)
                .select(["B2", "B3", "B4", "B5", "B8", "B8A", "B11"]))

    n_images = col.size().getInfo()
    if n_images == 0:
        return None

    composite = col.median()

    # Compute indices in GEE
    eps  = 1e-9
    ndvi = composite.normalizedDifference(["B8", "B4"]).rename("ndvi")
    evi  = composite.expression(
        "2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)",
        {"NIR": composite.select("B8"), "RED": composite.select("B4"),
         "BLUE": composite.select("B2")}
    ).rename("evi")
    ndwi = composite.normalizedDifference(["B3", "B8"]).rename("ndwi")
    ndre = composite.normalizedDifference(["B8A", "B5"]).rename("ndre")  # correct
    bsi  = composite.expression(
        "((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))",
        {"SWIR": composite.select("B11"), "RED": composite.select("B4"),
         "NIR": composite.select("B8"),  "BLUE": composite.select("B2")}
    ).rename("bsi")
    lai  = evi.multiply(3.618).subtract(0.118).rename("lai")

    # Temporal NDVI stats across all images in the composite
    ndvi_col   = col.map(lambda img: img.normalizedDifference(["B8", "B4"]))
    ndvi_std   = ndvi_col.reduce(ee.Reducer.stdDev()).rename("ndvi_std")
    ndvi_minmax = ndvi_col.reduce(ee.Reducer.minMax()).rename(["ndvi_min", "ndvi_max"])

    full = (composite.select(["B2","B3","B4","B5","B8","B8A","B11"])
                     .rename(["b2","b3","b4","b5","b8","b8a","b11"])
                     .addBands([ndvi, evi, ndwi, ndre, lai, bsi,
                                ndvi_std, ndvi_minmax]))

    vals = full.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=10,
        maxPixels=1e9,
    ).getInfo()

    if any(v is None for v in vals.values()):
        return None

    return {
        "location_id": loc_id, "location_name": name, "province": province,
        "season": label, "season_type": season_type, "crop_type": crop,
        "lat": lat, "lon": lon, "n_images": n_images,
        **{k: round(v, 6) for k, v in vals.items()},
    }


def collect_all(out_path: Path, max_workers: int = 8):
    ee_init()

    tasks = [
        (loc, stype, yr)
        for yr in YEARS
        for stype in ("rabi", "kharif")
        for loc in LOCATIONS
    ]
    log.info(f"Total tasks: {len(tasks)}")

    rows = []
    if out_path.exists():
        rows = pd.read_csv(out_path).to_dict("records")
        done = {(r["location_id"], r["season"]) for r in rows}
        tasks = [(loc, st, yr) for loc, st, yr in tasks
                 if (loc[0], f"{'Rabi' if st == 'rabi' else 'Kharif'}_{yr}") not in done]
        log.info(f"Resuming — {len(done)} already collected, {len(tasks)} remaining")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_location, loc, st, yr): (loc[0], st, yr)
                   for loc, st, yr in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            loc_id, st, yr = futures[fut]
            try:
                result = fut.result()
                if result:
                    rows.append(result)
                    log.info(f"[{i}/{len(futures)}] OK   {loc_id} {st} {yr}")
                else:
                    log.warning(f"[{i}/{len(futures)}] SKIP {loc_id} {st} {yr} — no clear images")
            except Exception as exc:
                log.error(f"[{i}/{len(futures)}] FAIL {loc_id} {st} {yr} — {exc}")

            if i % 50 == 0:
                pd.DataFrame(rows).to_csv(out_path, index=False)
                log.info(f"  Checkpoint saved → {out_path} ({len(rows)} rows)")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    log.info(f"Done — {len(df)} observations saved to {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect GEE Sentinel-2 data")
    parser.add_argument("--out",     default="data/gee_collected.csv",
                        help="Output CSV path")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel GEE workers")
    args = parser.parse_args()
    collect_all(Path(args.out), max_workers=args.workers)
