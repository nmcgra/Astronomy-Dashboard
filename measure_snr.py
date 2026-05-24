#!/usr/bin/env python3
"""
measure_snr.py — Measure Signal-to-Noise ratio of M87 from a stacked FITS image.

Method:
  1. Auto-detects M87 as the brightest source in the image
  2. Measures signal using aperture photometry (circular aperture around core)
  3. Estimates noise from a sky annulus surrounding the source
  4. Computes S/N = source_counts / (sqrt(source_counts + n_pix * sky_noise^2))
  5. Saves a draft report you can submit

Dependencies:
    pip install numpy astropy matplotlib

Usage:
  python3 measure_snr.py m87_stacked_R.fit
  python3 measure_snr.py m87_stacked_R.fit --aperture 40 --inner 60 --outer 90
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats


# ---------------------------------------------------------------------------
# FITS loader
# ---------------------------------------------------------------------------

def load_fits(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        hdr  = hdul[0].header.copy()
        data = hdul[0].data.astype(np.float32)
    bzero  = float(hdr.get("BZERO",  0.0))
    bscale = float(hdr.get("BSCALE", 1.0))
    if bzero != 0.0 or bscale != 1.0:
        data = data * bscale + bzero
    return data, hdr


# ---------------------------------------------------------------------------
# Source detection — find brightest source (M87 core)
# ---------------------------------------------------------------------------

def find_brightest_source(data: np.ndarray, box: int = 50) -> tuple[int, int]:
    """
    Smooth the image with a box average and return (row, col) of the peak.
    Box averaging reduces hot pixel false detections.
    """
    from astropy.convolution import convolve, Box2DKernel
    print("   Smoothing image to locate brightest source ...")
    smoothed = convolve(data, Box2DKernel(box), normalize_kernel=True,
                        nan_treatment="interpolate", preserve_nan=False)
    idx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    row, col = int(idx[0]), int(idx[1])
    print(f"   Brightest source found at pixel  col={col}  row={row}")
    return row, col


# ---------------------------------------------------------------------------
# Circular aperture mask helpers
# ---------------------------------------------------------------------------

def circular_mask(shape, cy, cx, radius):
    """Boolean mask — True inside circle of given radius centred at (cy, cx)."""
    y, x = np.ogrid[:shape[0], :shape[1]]
    return (x - cx)**2 + (y - cy)**2 <= radius**2


def annulus_mask(shape, cy, cx, r_inner, r_outer):
    """Boolean mask — True in annulus between r_inner and r_outer."""
    y, x = np.ogrid[:shape[0], :shape[1]]
    dist2 = (x - cx)**2 + (y - cy)**2
    return (dist2 >= r_inner**2) & (dist2 <= r_outer**2)


# ---------------------------------------------------------------------------
# S/N measurement
# ---------------------------------------------------------------------------

def measure_snr(data: np.ndarray, cy: int, cx: int,
                ap_r: int, sky_inner: int, sky_outer: int) -> dict:
    """
    Aperture photometry S/N measurement.

    Signal  = sum of (background-subtracted) counts inside aperture
    Noise   = sqrt(signal + N_pix * sky_sigma^2)
              (Poisson noise from source + background noise per pixel)
    S/N     = Signal / Noise
    """
    h, w = data.shape

    # Aperture
    ap_mask  = circular_mask((h, w), cy, cx, ap_r)
    ap_pixels = data[ap_mask]
    n_ap      = ap_pixels.size

    # Sky annulus — sigma-clip to reject stars in the annulus
    sky_mask    = annulus_mask((h, w), cy, cx, sky_inner, sky_outer)
    sky_pixels  = data[sky_mask]
    sky_mean, sky_median, sky_sigma = sigma_clipped_stats(sky_pixels, sigma=3.0)

    # Background-subtracted source counts
    source_counts = float(np.sum(ap_pixels) - n_ap * sky_median)

    # Combined noise (Poisson + background)
    noise = float(np.sqrt(abs(source_counts) + n_ap * sky_sigma**2))

    snr = source_counts / noise if noise > 0 else 0.0

    return {
        "cx": cx, "cy": cy,
        "aperture_radius_px":  ap_r,
        "sky_inner_px":        sky_inner,
        "sky_outer_px":        sky_outer,
        "n_aperture_pixels":   n_ap,
        "n_sky_pixels":        sky_mask.sum(),
        "sky_median":          float(sky_median),
        "sky_sigma":           float(sky_sigma),
        "raw_aperture_sum":    float(np.sum(ap_pixels)),
        "source_counts":       source_counts,
        "noise":               noise,
        "snr":                 snr,
    }


# ---------------------------------------------------------------------------
# Optional diagnostic plot
# ---------------------------------------------------------------------------

def save_plot(data: np.ndarray, cy: int, cx: int,
              ap_r: int, sky_inner: int, sky_outer: int,
              output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from astropy.visualization import ZScaleInterval

        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(data)

        # Crop a region around the source for display
        pad  = sky_outer + 50
        y0, y1 = max(0, cy - pad), min(data.shape[0], cy + pad)
        x0, x1 = max(0, cx - pad), min(data.shape[1], cx + pad)
        crop = data[y0:y1, x0:x1]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(crop, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
                  extent=[x0, x1, y0, y1])

        # Draw aperture and annulus
        for r, color, label in [
            (ap_r,     "cyan",   f"Aperture r={ap_r}px"),
            (sky_inner,"yellow", f"Sky inner r={sky_inner}px"),
            (sky_outer,"orange", f"Sky outer r={sky_outer}px"),
        ]:
            circle = plt.Circle((cx, cy), r, color=color, fill=False, linewidth=1.5,
                                 label=label)
            ax.add_patch(circle)

        ax.plot(cx, cy, "+", color="cyan", markersize=12, markeredgewidth=2)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title("M87 — Aperture Photometry", fontsize=13)
        ax.set_xlabel("X pixel"); ax.set_ylabel("Y pixel")
        plt.tight_layout()
        plt.savefig(output, dpi=150)
        plt.close()
        print(f"   Diagnostic plot saved → {output}")
    except Exception as e:
        print(f"   (Plot skipped: {e})")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(result: dict, hdr: fits.Header, fits_path: Path,
                 output: Path, n_frames: int) -> None:
    exptime  = hdr.get("EXPTIME",  "unknown")
    gain     = hdr.get("GAIN",     "unknown")
    ccdtemp  = hdr.get("CCD-TEMP", "unknown")
    dateobs  = hdr.get("DATE-OBS", "unknown")
    instrume = hdr.get("INSTRUME", "unknown")
    filt     = hdr.get("FILTER",   "R")

    lines = [
        "=" * 65,
        "  M87 Signal-to-Noise Ratio — Draft Report",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        "",
        "OBSERVATION SUMMARY",
        "-" * 40,
        f"  Target           : M87 (Virgo A)",
        f"  Filter           : {filt}-band",
        f"  Instrument       : {instrume}",
        f"  Frames stacked   : {n_frames}",
        f"  Exposure / frame : {exptime} s",
        f"  Total exposure   : {float(exptime) * n_frames:.0f} s  ({float(exptime) * n_frames / 60:.1f} min)"
          if str(exptime).replace('.','').isdigit() else f"  Total exposure   : N/A",
        f"  CCD temperature  : {ccdtemp} °C",
        f"  Gain             : {gain}",
        f"  Observation date : {dateobs}",
        f"  Input file       : {fits_path.name}",
        "",
        "APERTURE PHOTOMETRY SETUP",
        "-" * 40,
        f"  Source position  : col={result['cx']}  row={result['cy']}  (auto-detected peak)",
        f"  Aperture radius  : {result['aperture_radius_px']} px  ({result['n_aperture_pixels']} pixels)",
        f"  Sky annulus      : {result['sky_inner_px']} – {result['sky_outer_px']} px  "
        f"({result['n_sky_pixels']} pixels)",
        "",
        "PHOTOMETRY RESULTS",
        "-" * 40,
        f"  Sky background (median)  : {result['sky_median']:.2f} counts/px",
        f"  Sky noise (sigma)        : {result['sky_sigma']:.2f} counts/px",
        f"  Raw aperture sum         : {result['raw_aperture_sum']:.0f} counts",
        f"  Background-subtracted    : {result['source_counts']:.0f} counts",
        f"  Total noise estimate     : {result['noise']:.0f} counts",
        "",
        f"  ╔══════════════════════════════╗",
        f"  ║   S/N  =  {result['snr']:>8.1f}             ║",
        f"  ╚══════════════════════════════╝",
        "",
        "METHOD SUMMARY",
        "-" * 40,
        f"  The {n_frames} R-band light frames were stacked using kappa-sigma",
        f"  median combination (κ=2.5, 3 iterations) to produce a master",
        f"  image. Signal-to-noise was measured via aperture photometry:",
        f"  source counts were summed within a circular aperture of radius",
        f"  {result['aperture_radius_px']} px centred on the M87 core (auto-detected as the",
        f"  brightest source). Sky background was estimated from a sigma-",
        f"  clipped annulus ({result['sky_inner_px']}–{result['sky_outer_px']} px) and subtracted.",
        f"  Noise was computed as:",
        f"    N = sqrt(source_counts + N_pix * sky_sigma^2)",
        f"  combining Poisson noise from the source with per-pixel",
        f"  background noise scaled by aperture area.",
        "",
        "  NOTE: Calibration frames (darks, flats, bias) have not yet",
        f"  been applied. The final S/N reported in the presentation",
        f"  will be updated after full calibration.",
        "",
        "=" * 65,
    ]

    with open(output, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"   Report saved → {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Measure S/N of M87 from a stacked FITS image.")
    parser.add_argument("fits", type=Path, help="Stacked FITS file (e.g. m87_stacked_R.fit)")
    parser.add_argument("--aperture", type=int, default=40,
                        help="Aperture radius in pixels (default: 40)")
    parser.add_argument("--inner", type=int, default=60,
                        help="Sky annulus inner radius in pixels (default: 60)")
    parser.add_argument("--outer", type=int, default=90,
                        help="Sky annulus outer radius in pixels (default: 90)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip saving the diagnostic plot")
    args = parser.parse_args()

    if not args.fits.exists():
        print(f"ERROR: File not found: {args.fits}")
        sys.exit(1)

    print(f"\nMeasure S/N — {args.fits.name}")
    print("-" * 60)

    # Load
    print("Loading FITS ...")
    data, hdr = load_fits(args.fits)
    print(f"   Image size: {data.shape}  min={data.min():.1f}  max={data.max():.1f}")

    n_frames = int(hdr.get("STACKCNT", 1))

    # Find M87
    print("\nLocating M87 ...")
    cy, cx = find_brightest_source(data)

    # Measure S/N
    print("\nMeasuring S/N ...")
    result = measure_snr(data, cy, cx,
                         ap_r=args.aperture,
                         sky_inner=args.inner,
                         sky_outer=args.outer)

    # Print summary
    print(f"\n{'='*50}")
    print(f"  Source position : col={cx}  row={cy}")
    print(f"  Source counts   : {result['source_counts']:.0f}")
    print(f"  Sky median      : {result['sky_median']:.2f} cts/px")
    print(f"  Sky sigma       : {result['sky_sigma']:.2f} cts/px")
    print(f"  Noise           : {result['noise']:.0f}")
    print(f"  S/N             : {result['snr']:.1f}")
    print(f"{'='*50}\n")

    # Save outputs
    stem = args.fits.stem
    report_path = args.fits.parent / f"{stem}_snr_report.txt"
    plot_path   = args.fits.parent / f"{stem}_aperture.png"

    write_report(result, hdr, args.fits, report_path, n_frames)

    if not args.no_plot:
        print("   Saving diagnostic plot ...")
        save_plot(data, cy, cx, args.aperture, args.inner, args.outer, plot_path)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
