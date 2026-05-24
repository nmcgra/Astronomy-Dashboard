#!/usr/bin/env python3
"""
surface_brightness.py — Measure and plot the surface brightness profile of M87.

Measures mean intensity in concentric annuli from the core outward, then plots:
  1. Intensity vs radius (linear)
  2. log(I) vs r^(1/4) — tests de Vaucouleurs law (should be straight line)
  3. S/N per annulus — shows where signal drops into noise

Dependencies:
    pip install numpy astropy matplotlib scipy

Usage:
  python3 surface_brightness.py m87_stacked_R.fit
  python3 surface_brightness.py m87_stacked_R.fit --cx 3131 --cy 757
  python3 surface_brightness.py m87_stacked_R.fit --max-radius 300
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy import stats


# ---------------------------------------------------------------------------
# FITS loader
# ---------------------------------------------------------------------------

def load_fits(path):
    with fits.open(path, memmap=False) as hdul:
        hdr  = hdul[0].header.copy()
        data = hdul[0].data.astype(np.float32)
    bzero  = float(hdr.get("BZERO",  0.0))
    bscale = float(hdr.get("BSCALE", 1.0))
    if bzero != 0.0 or bscale != 1.0:
        data = data * bscale + bzero
    return data, hdr


# ---------------------------------------------------------------------------
# Find brightest source
# ---------------------------------------------------------------------------

def find_peak(data, box=50):
    from astropy.convolution import convolve, Box2DKernel
    print("   Locating M87 core (smoothed peak) ...")
    smoothed = convolve(data, Box2DKernel(box), normalize_kernel=True,
                        nan_treatment="interpolate", preserve_nan=False)
    idx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    cy, cx = int(idx[0]), int(idx[1])
    print(f"   Core found at col={cx}  row={cy}")
    return cy, cx


# ---------------------------------------------------------------------------
# Surface brightness profile
# ---------------------------------------------------------------------------

def measure_profile(data, cy, cx, max_radius, step=2):
    """
    Measure mean intensity and noise in concentric annuli.
    Returns arrays of radii, mean intensity, std, and number of pixels.
    """
    h, w   = data.shape
    # Pre-compute distance map from core
    y, x   = np.ogrid[:h, :w]
    dist   = np.sqrt((x - cx)**2 + (y - cy)**2)

    radii, intensity, noise, npix = [], [], [], []

    # Estimate sky background far from source
    sky_mask = (dist > max_radius * 1.1) & (dist < max_radius * 1.3)
    if sky_mask.sum() > 100:
        sky_mean, sky_median, sky_sigma = sigma_clipped_stats(data[sky_mask], sigma=3.0)
    else:
        # Fall back to image corners
        corner = data[:100, :100]
        sky_mean, sky_median, sky_sigma = sigma_clipped_stats(corner, sigma=3.0)

    print(f"   Sky background: {sky_median:.1f} cts/px  sigma={sky_sigma:.2f}")

    r = step
    while r <= max_radius:
        r_inner = max(0, r - step)
        r_outer = r
        mask = (dist >= r_inner) & (dist < r_outer)
        n = mask.sum()
        if n < 5:
            r += step
            continue
        pixels = data[mask]
        mean_val = float(np.mean(pixels)) - float(sky_median)
        std_val  = float(np.std(pixels))
        radii.append(r)
        intensity.append(mean_val)
        noise.append(sky_sigma)
        npix.append(n)
        r += step

    return (np.array(radii), np.array(intensity),
            np.array(noise), np.array(npix),
            sky_median, sky_sigma)


# ---------------------------------------------------------------------------
# de Vaucouleurs fit
# ---------------------------------------------------------------------------

def fit_devaucouleurs(radii, intensity):
    """
    Fit log(I) vs r^(1/4) with a linear regression.
    Only fit points where intensity > 0.
    Returns slope, intercept, r_value, and fit arrays.
    """
    mask = intensity > 0
    if mask.sum() < 5:
        return None

    r_quarter = radii[mask] ** 0.25
    log_i     = np.log10(intensity[mask])

    slope, intercept, r_value, p_value, std_err = stats.linregress(r_quarter, log_i)
    fit_y = slope * r_quarter + intercept

    return {
        "slope": slope, "intercept": intercept,
        "r_value": r_value, "r_squared": r_value**2,
        "r_quarter": r_quarter, "log_i": log_i, "fit_y": fit_y,
        "mask": mask,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(radii, intensity, noise, npix, sky_sigma, fit, output):
    snr_per_annulus = intensity / sky_sigma

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("M87 — Surface Brightness Profile", fontsize=15, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # --- Plot 1: Intensity vs radius (linear) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(radii, intensity, "b-", linewidth=1.5, label="M87")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Sky level")
    ax1.axhline(3 * sky_sigma, color="red", linestyle=":", linewidth=1,
                label=f"3σ sky ({3*sky_sigma:.1f} cts)")
    ax1.set_xlabel("Radius (pixels)")
    ax1.set_ylabel("Background-subtracted intensity (cts/px)")
    ax1.set_title("Intensity Profile (linear)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Intensity vs radius (log scale) ---
    ax2 = fig.add_subplot(gs[0, 1])
    pos_mask = intensity > 0
    ax2.semilogy(radii[pos_mask], intensity[pos_mask], "b-", linewidth=1.5)
    ax2.axhline(3 * sky_sigma, color="red", linestyle=":", linewidth=1,
                label=f"3σ sky noise")
    ax2.set_xlabel("Radius (pixels)")
    ax2.set_ylabel("Intensity (cts/px, log scale)")
    ax2.set_title("Intensity Profile (log scale)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, which="both")

    # --- Plot 3: de Vaucouleurs r^(1/4) law ---
    ax3 = fig.add_subplot(gs[1, 0])
    if fit:
        ax3.scatter(fit["r_quarter"], fit["log_i"], s=8, color="blue",
                    alpha=0.6, label="Data")
        ax3.plot(fit["r_quarter"], fit["fit_y"], "r-", linewidth=2,
                 label=f"Linear fit  R²={fit['r_squared']:.3f}")
        ax3.set_xlabel("r^(1/4)  (pixels^0.25)")
        ax3.set_ylabel("log₁₀(Intensity)")
        ax3.set_title("de Vaucouleurs r^(1/4) Law Test")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        # Annotation
        ax3.text(0.05, 0.08,
                 f"slope = {fit['slope']:.3f}\nR² = {fit['r_squared']:.3f}",
                 transform=ax3.transAxes, fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    else:
        ax3.text(0.5, 0.5, "Not enough positive-intensity\ndata points for fit",
                 ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("de Vaucouleurs r^(1/4) Law Test")

    # --- Plot 4: S/N per annulus ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(radii, snr_per_annulus, "g-", linewidth=1.5)
    ax4.axhline(3, color="red", linestyle="--", linewidth=1, label="S/N = 3 (detection limit)")
    ax4.axhline(1, color="orange", linestyle=":", linewidth=1, label="S/N = 1")
    # Mark where signal drops below 3-sigma
    below3 = np.where(snr_per_annulus < 3)[0]
    if len(below3) > 0:
        r_limit = radii[below3[0]]
        ax4.axvline(r_limit, color="red", linestyle="-", linewidth=0.8, alpha=0.5)
        ax4.text(r_limit + 2, ax4.get_ylim()[1] * 0.85 if ax4.get_ylim()[1] > 0 else 5,
                 f"Signal lost\nat r={r_limit}px", fontsize=8, color="red")
    ax4.set_xlabel("Radius (pixels)")
    ax4.set_ylabel("S/N per annulus")
    ax4.set_title("S/N vs Radius")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)

    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Plot saved → {output}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(radii, intensity, noise, sky_median, sky_sigma, fit, hdr, output):
    snr = intensity / sky_sigma
    # Find detection limit radius
    below3 = np.where(snr < 3)[0]
    r_limit = radii[below3[0]] if len(below3) > 0 else radii[-1]

    lines = [
        "=" * 65,
        "  M87 Surface Brightness Profile — Report",
        "=" * 65,
        "",
        "BACKGROUND ESTIMATE",
        "-" * 40,
        f"  Sky median       : {sky_median:.2f} cts/px",
        f"  Sky sigma (noise): {sky_sigma:.2f} cts/px",
        f"  3σ limit         : {3*sky_sigma:.2f} cts/px",
        "",
        "PROFILE SUMMARY",
        "-" * 40,
        f"  Innermost radius : {radii[0]:.0f} px",
        f"  Outermost radius : {radii[-1]:.0f} px",
        f"  Peak intensity   : {intensity.max():.1f} cts/px  at r={radii[intensity.argmax()]:.0f}px",
        f"  Detection limit  : r = {r_limit:.0f} px  (S/N drops below 3σ here)",
        "",
        "de VAUCOULEURS FIT  (log I vs r^1/4)",
        "-" * 40,
    ]

    if fit:
        lines += [
            f"  Slope            : {fit['slope']:.4f}",
            f"  Intercept        : {fit['intercept']:.4f}",
            f"  R²               : {fit['r_squared']:.4f}",
            f"",
            f"  Interpretation:",
            f"  An R² close to 1.0 indicates the brightness profile",
            f"  follows the de Vaucouleurs r^(1/4) law, confirming M87's",
            f"  character as a giant elliptical galaxy. R²={fit['r_squared']:.3f}",
            f"  {'strongly supports' if fit['r_squared'] > 0.95 else 'is consistent with' if fit['r_squared'] > 0.85 else 'partially supports'}",
            f"  this interpretation.",
        ]
    else:
        lines.append("  Insufficient data for fit.")

    lines += [
        "",
        "ANNULUS DATA  (first 20 radii)",
        "-" * 40,
        f"  {'Radius':>8} {'Intensity':>12} {'S/N':>8} {'N pixels':>10}",
        "  " + "-" * 44,
    ]
    for i in range(min(20, len(radii))):
        lines.append(f"  {radii[i]:>8.1f} {intensity[i]:>12.2f} {snr[i]:>8.2f} {noise[i]:>10.0f}")

    lines += ["  ...", "", "=" * 65]

    with open(output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"   Report saved → {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="M87 surface brightness profile.")
    parser.add_argument("fits", type=Path, help="Stacked FITS file")
    parser.add_argument("--cx", type=int, default=None, help="Core X pixel (auto-detect if omitted)")
    parser.add_argument("--cy", type=int, default=None, help="Core Y pixel (auto-detect if omitted)")
    parser.add_argument("--max-radius", type=int, default=250,
                        help="Max radius in pixels to measure (default: 250)")
    parser.add_argument("--step", type=int, default=2,
                        help="Annulus width in pixels (default: 2)")
    args = parser.parse_args()

    if not args.fits.exists():
        print(f"ERROR: {args.fits} not found"); sys.exit(1)

    print(f"\nSurface Brightness Profile — {args.fits.name}")
    print("-" * 60)

    print("Loading FITS ...")
    data, hdr = load_fits(args.fits)
    print(f"   Image: {data.shape}  min={data.min():.0f}  max={data.max():.0f}")

    # Find core
    if args.cx is not None and args.cy is not None:
        cy, cx = args.cy, args.cx
        print(f"   Using provided core position: col={cx} row={cy}")
    else:
        cy, cx = find_peak(data)

    # Measure profile
    print(f"\nMeasuring profile out to r={args.max_radius}px ...")
    radii, intensity, noise, npix, sky_med, sky_sig = measure_profile(
        data, cy, cx, args.max_radius, args.step)
    print(f"   Measured {len(radii)} annuli")

    # de Vaucouleurs fit
    print("\nFitting de Vaucouleurs r^(1/4) law ...")
    fit = fit_devaucouleurs(radii, intensity)
    if fit:
        print(f"   R² = {fit['r_squared']:.4f}  slope = {fit['slope']:.4f}")
    else:
        print("   Not enough data for fit")

    # Save outputs
    stem        = args.fits.stem
    plot_path   = args.fits.parent / f"{stem}_sb_profile.png"
    report_path = args.fits.parent / f"{stem}_sb_profile.txt"

    print("\nSaving outputs ...")
    make_plots(radii, intensity, noise, npix, sky_sig, fit, plot_path)
    write_report(radii, intensity, noise, sky_med, sky_sig, fit, hdr, report_path)

    snr = intensity / sky_sig
    below3 = np.where(snr < 3)[0]
    r_limit = radii[below3[0]] if len(below3) > 0 else radii[-1]

    print(f"\n{'='*50}")
    print(f"  Core position   : col={cx}  row={cy}")
    print(f"  Detection limit : r = {r_limit:.0f} px")
    if fit:
        print(f"  de Vauc. R²     : {fit['r_squared']:.4f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
