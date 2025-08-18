#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 15:36:57 2025

@author: jason
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Showcase plot: Radio Luminosity vs Star Formation Rate (your real data)
-----------------------------------------------------------------------
What it makes:
- log10(SFR) vs log10(L_1.4GHz) scatter (your IR-based SFRs against radio luminosities)
- Overlays your measured best-fit slope line (fit to YOUR data)
- ±1σ shaded region around the best-fit (from residual scatter)
- Overlays the canonical unity-slope line (m=1) for comparison
- Optionally annotates the fraction of sources with S_1.4 < 0.2 mJy (if flux column available)
- Saves figure as PNG and PDF

How to run:
    python plot_rl_vs_sfr.py --input your_data.csv --outfig rl_vs_sfr_showcase

Expected columns (it will try to auto-detect common names):
    Radio luminosity (log10): one of
        - logL_1p4GHz, logL1p4, log10_L1p4, Radio_Luminosity_1.4GHz, L1p4_log, log10_L_1.4GHz
    SFR (linear units, Msun/yr) or log10(SFR): one of
        - SFR_IR, True_SFR, sfr_true, SFR, SFR_total, SFR_ir
        - or log10(SFR): logSFR, log10_SFR, log10SFR, log_SFR, log10_True_SFR
    Optional observed 1.4 GHz flux density (mJy): one of
        - S1p4_mJy, Flux_1p4_mJy, Observed_Flux_mJy, S1p4, S_1p4_mJy

If your data are in a FITS file, use "--fits" and give the HDU index or name if needed:
    python plot_rl_vs_sfr.py --input your_data.fits --fits 1
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

# Use serif fonts globally
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'

try:
    from astropy.table import Table
except Exception:
    Table = None

def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # try case-insensitive
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def load_table(path, fits_hdu=None):
    path = Path(path)
    if path.suffix.lower() in ['.csv', '.tsv']:
        df = pd.read_csv(path) if path.suffix.lower() == '.csv' else pd.read_table(path)
        return df
    elif path.suffix.lower() in ['.fits', '.fit', '.fz']:
        if Table is None:
            raise RuntimeError("astropy is required to read FITS files. pip install astropy")
        tab = Table.read(str(path), hdu=fits_hdu)
        return tab.to_pandas()
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Path to CSV or FITS containing your real data')
    ap.add_argument('--fits', default=None, help='FITS HDU index or name (if using FITS)')
    ap.add_argument('--outfig', default='rl_vs_sfr_showcase', help='Base name for output figure files')
    ap.add_argument('--title', default='Radio Luminosity vs. Star Formation Rate')

    # Optional explicit column-name overrides
    ap.add_argument('--col-logL', default=None, help='Name of log10(L1.4GHz) column')
    ap.add_argument('--col-sfr', default=None, help='Name of SFR (linear) column')
    ap.add_argument('--col-logsfr', default=None, help='Name of log10(SFR) column')
    ap.add_argument('--col-flux', default=None, help='Name of S_1.4GHz flux density column (mJy)')

    args = ap.parse_args()

    # Load
    df = load_table(args.input, args.fits)

    # Identify columns
    logL_candidates = ['logL_1p4GHz','logL1p4','log10_L1p4','Radio_Luminosity_1.4GHz','L1p4_log','log10_L_1.4GHz']
    sfr_candidates_lin = ['SFR_IR','True_SFR','sfr_true','SFR','SFR_total','SFR_ir']
    sfr_candidates_log = ['logSFR','log10_SFR','log10SFR','log_SFR','log10_True_SFR']
    flux_candidates = ['S1p4_mJy','Flux_1p4_mJy','Observed_Flux_mJy','S1p4','S_1p4_mJy']

    logL_col = args.col_logL or _pick_col(df, logL_candidates)
    sfr_col = args.col_sfr or _pick_col(df, sfr_candidates_lin)
    logSFR_col = args.col_logsfr or _pick_col(df, sfr_candidates_log)
    flux_col = args.col_flux or _pick_col(df, flux_candidates)

    if logL_col is None:
        raise KeyError(f"Could not find a radio luminosity (log10) column in {df.columns.tolist()}")
    if sfr_col is None and logSFR_col is None:
        raise KeyError(f"Could not find SFR columns (linear or log). Present columns: {df.columns.tolist()}")

    # Build working arrays
    logL = df[logL_col].astype(float).values

    if logSFR_col is not None:
        logSFR = df[logSFR_col].astype(float).values
    else:
        sfr = df[sfr_col].astype(float).values
        # Filter out non-positive values
        mask_pos = np.isfinite(sfr) & (sfr > 0)
        logL = logL[mask_pos]
        sfr = sfr[mask_pos]
        logSFR = np.log10(sfr)

    # Filter finite
    mfin = np.isfinite(logL) & np.isfinite(logSFR)
    logL, logSFR = logL[mfin], logSFR[mfin]

    # Fit best-fit slope and intercept: logSFR = m*logL + b
    m, b = np.polyfit(logL, logSFR, 1)

    # Residuals and scatter for ±1σ band
    residuals = logSFR - (m * logL + b)
    sigma = np.std(residuals)

    # Prepare line + 1-sigma envelope
    xgrid = np.linspace(np.nanmin(logL), np.nanmax(logL), 200)
    y_fit = m * xgrid + b
    y_fit_upper = y_fit + sigma
    y_fit_lower = y_fit - sigma

    # For unity slope, choose intercept so that line passes through the (median logL, median logSFR)
    x0, y0 = np.median(logL), np.median(logSFR)
    y_unity = 1.0 * xgrid + (y0 - x0)

    # Pearson correlation and R^2
    # Using numpy's corrcoef for Pearson r
    r = np.corrcoef(logL, logSFR)[0,1]
    r2 = r**2

    # Make plot
    plt.figure(figsize=(5,4))
    plt.scatter(logL, logSFR, s=10, alpha=0.35, label='Galaxies')
    plt.plot(xgrid, y_fit, label=f'Best-fit slope (m={m:.2f})', linewidth=2)
    plt.fill_between(xgrid, y_fit_lower, y_fit_upper, alpha=0.2, label=r'$\pm 1\sigma$ scatter')
    plt.plot(xgrid, y_unity, '--', label='Canonical unity slope (m=1)', linewidth=2)

    # Optional: annotate ~0.2 mJy fraction if flux column available
    if flux_col is not None:
        flux_all = df.loc[mfin, flux_col]
        # Some tables may store as string; convert carefully
        flux = pd.to_numeric(flux_all, errors='coerce').values
        frac_below = np.mean(np.isfinite(flux) & (flux < 0.2)) * 100.0
        txt_flux = f"S₁.₄ < 0.2 mJy: {frac_below:.1f}%"
    else:
        txt_flux = None

    # Add a small annotation with stats (top-right)
    
    '''
    stats_lines = [f"Pearson r = {r:.2f}", f"R² = {r2:.2f}", f"σ (residuals) = {sigma:.2f} dex"]
    if txt_flux:
        stats_lines.append(txt_flux)
    stats_text = "\n".join(stats_lines)
    plt.annotate(stats_text, xy=(0.97, 0.97), xycoords='axes fraction',
                 ha='right', va='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
    
    '''
    
    plt.xlabel(r'$\log_{10}\,L_{1.4\,\mathrm{GHz}}\ \mathrm{[W\,Hz^{-1}]}$')
    plt.ylabel(r'$\log_{10}\,\mathrm{SFR}\ \mathrm{[M_\odot\,yr^{-1}]}$')
    #plt.title(args.title)
    plt.grid(alpha=0.3)
    plt.legend()

    # Save
    out_png = f"{args.outfig}.png"
    out_pdf = f"{args.outfig}.pdf"
    plt.tight_layout()
    plt.savefig(out_png, dpi=400)
    plt.savefig(out_pdf)
    print(f"Saved: {out_png} and {out_pdf}")
    plt.show()

if __name__ == "__main__":
    main()
