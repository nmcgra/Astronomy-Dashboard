#!/usr/bin/env python3
"""
stack_fits.py — Stack M87 R-band light frames into a single master image.

Tuned for M87 on large sensors (ZWO ASI294MM Pro):
  - Kappa-sigma median stacking (κ=2.5) by default
  - Processes in horizontal tiles to keep RAM usage low and stay fast
  - No calibration (stack as-is)
  - Outputs a 32-bit float FITS with full header provenance

Dependencies:
    pip install numpy astropy

Usage:
  python3 stack_fits.py *.fit
  python3 stack_fits.py *.fit --output my_m87.fit
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits


# ---------------------------------------------------------------------------
# FITS I/O
# ---------------------------------------------------------------------------

def load_fits(path: Path) -> tuple[np.ndarray, fits.Header]:
    """Load a FITS file and return (float32 array, header)."""
    with fits.open(path, memmap=False) as hdul:
        hdr  = hdul[0].header.copy()
        data = hdul[0].data.astype(np.float32)
    bzero  = float(hdr.get("BZERO",  0.0))
    bscale = float(hdr.get("BSCALE", 1.0))
    if bzero != 0.0 or bscale != 1.0:
        data = data * bscale + bzero
    return data, hdr


def save_fits(data: np.ndarray, header: fits.Header, output: Path) -> None:
    hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=header)
    hdu.writeto(output, overwrite=True)
    print(f"\n✓  Saved → {output}")


# ---------------------------------------------------------------------------
# Tiled kappa-sigma median stacker
# ---------------------------------------------------------------------------

def kappa_sigma_median_tile(tile: np.ndarray, kappa: float, iterations: int) -> np.ndarray:
    """
    Stack a tile (N, rows, cols) using kappa-sigma median rejection.
    Uses MAD-based sigma estimate — robust against bright cores and gradients.
    """
    masked = tile.astype(np.float32)

    for _ in range(iterations):
        median  = np.nanmedian(masked, axis=0, keepdims=True)
        mad     = np.nanmedian(np.abs(masked - median), axis=0, keepdims=True)
        sigma   = mad * 1.4826
        outlier = np.abs(masked - median) > kappa * sigma
        masked[outlier] = np.nan

    return np.nanmedian(masked, axis=0)


def stack_tiled(frames: list, kappa: float, iterations: int,
                tile_rows: int = 256) -> np.ndarray:
    """
    Stack all frames using tiled kappa-sigma median.
    Processes `tile_rows` rows at a time to keep memory usage manageable.
    """
    n  = len(frames)
    h  = frames[0].shape[0]
    w  = frames[0].shape[1]
    result   = np.empty((h, w), dtype=np.float32)
    n_tiles  = (h + tile_rows - 1) // tile_rows
    total_clipped = 0
    total_pixels  = 0

    print(f"   Tiled kappa-sigma median  κ={kappa}  iters={iterations}  "
          f"tile={tile_rows} rows  ({n_tiles} tiles)")

    t_start = time.time()

    for t in range(n_tiles):
        row_start = t * tile_rows
        row_end   = min(row_start + tile_rows, h)

        # Build (N, tile_rows, W) cube for this tile only
        tile = np.stack([f[row_start:row_end, :] for f in frames], axis=0)

        # Stack this tile
        stacked_tile = kappa_sigma_median_tile(tile, kappa, iterations)
        result[row_start:row_end, :] = stacked_tile

        # Count rejected pixels for the report
        masked = tile.astype(np.float32)
        for _ in range(iterations):
            median  = np.nanmedian(masked, axis=0, keepdims=True)
            mad     = np.nanmedian(np.abs(masked - median), axis=0, keepdims=True)
            sigma   = mad * 1.4826
            masked[np.abs(masked - median) > kappa * sigma] = np.nan
        total_clipped += int(np.isnan(masked).sum())
        total_pixels  += masked.size

        # Progress bar
        pct     = (t + 1) / n_tiles * 100
        elapsed = time.time() - t_start
        eta     = (elapsed / (t + 1)) * (n_tiles - t - 1)
        bar     = "#" * int(pct // 5)
        print(f"   Tile {t+1:>3}/{n_tiles}  [{bar:<20}] {pct:5.1f}%  "
              f"elapsed {elapsed:5.1f}s  ETA {eta:5.1f}s",
              end="\r", flush=True)

    print()
    print(f"   Rejected {total_clipped / total_pixels * 100:.2f}% of pixels total")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stack M87 FITS light frames (tiled, memory-efficient).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("lights", nargs="+", type=Path,
                        help="Input .fit / .fits light frame paths")
    parser.add_argument("--kappa", type=float, default=2.5,
                        help="Rejection threshold in sigma (default: 2.5)")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Rejection iterations (default: 3)")
    parser.add_argument("--tile-rows", type=int, default=256,
                        help="Rows per processing tile (default: 256; lower = less RAM)")
    parser.add_argument("--output", type=Path, default=Path("m87_stacked_R.fit"),
                        help="Output file (default: m87_stacked_R.fit)")
    args = parser.parse_args()

    # Validate inputs
    light_paths = sorted(set(args.lights))
    missing = [p for p in light_paths if not p.exists()]
    if missing:
        print(f"ERROR: {len(missing)} file(s) not found:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    n = len(light_paths)
    if n < 2:
        print("ERROR: Need at least 2 frames to stack.")
        sys.exit(1)

    print(f"\nStack FITS — {n} light frames  κ={args.kappa}  iters={args.iterations}")
    print("-" * 60)

    # Load all frames
    frames = []
    ref_header = None
    shape = None
    t0 = time.time()

    for i, path in enumerate(light_paths, 1):
        print(f"  [{i:>2}/{n}] {path.name}", end="  ", flush=True)
        data, hdr = load_fits(path)

        if ref_header is None:
            ref_header = hdr
            shape = data.shape

        if data.shape != shape:
            print(f"shape mismatch {data.shape} vs {shape} — SKIPPED")
            continue

        frames.append(data)
        print(f"{data.shape}  min={data.min():.0f}  max={data.max():.0f}")

    print(f"\nLoaded {len(frames)} frames in {time.time() - t0:.1f}s")

    # Stack
    print(f"\nStacking ...")
    t1 = time.time()
    stacked = stack_tiled(frames, kappa=args.kappa, iterations=args.iterations,
                          tile_rows=args.tile_rows)
    print(f"   Stacking done in {time.time() - t1:.1f}s")
    print(f"   Result  min={stacked.min():.1f}  max={stacked.max():.1f}  mean={stacked.mean():.1f}")

    # Update header
    ref_header["BITPIX"]   = -32
    ref_header["STACKCNT"] = (n,              "Frames stacked")
    ref_header["STACKMTH"] = ("kappa-median", "Stacking method")
    ref_header["STACKKAP"] = (args.kappa,     "Rejection kappa")
    ref_header["STACKITR"] = (args.iterations,"Rejection iterations")
    for kw in ("BZERO", "BSCALE"):
        ref_header.remove(kw, ignore_missing=True)

    save_fits(stacked, ref_header, args.output)
    print(f"   Shape: {stacked.shape}  dtype: float32\n")


if __name__ == "__main__":
    main()
