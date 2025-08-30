#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:58:34 2020

@author: Kabelo McKabuza
"""

import os
import time
import random
import numpy as np
import pandas as pd
import astropy.units as u

from tqdm import tqdm
from astropy.constants import c
from astropy.table import Table
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy.stats import biweight_scale
from astropy.cosmology import Planck18 as cosmo


# Setup
vizier_catalog = 'II/328/allwise'
chunk_size = 100
input_file = '/Users/jason/Downloads/Table2_MGCLS_compactcat_DR1.csv'
output_dir = './Output/'
os.makedirs(output_dir, exist_ok=True)

# Clusters and redshifts
clusters = {
    'Abell 133': 0.057,
    'Abell 141': 0.2300,
    'Abell 68': 0.2546,
    'Abell 194': 0.018,
    'Abell 209': 0.21,
    'Abell 22': 0.206,
    'Abell 2485': 0.247,
    'Abell 2597': 0.085,
    'Abell 2645': 0.251,
    'Abell 2667': 0.230,
    'Abell 2744': 0.308,
    'Abell 2751': 0.107,
    'Abell 2811': 0.108,
    'Abell 2895': 0.227,
    'Abell 3365': 0.093,
    'Abell 3376': 0.046,
    'Abell 33': 0.280,
    'Abell 3558': 0.048,
    'Abell 3562': 0.049,
    'Abell 3667': 0.056,
    'Abell 370': 0.375,
    'Abell 4038': 0.028,
    'Abell 521': 0.253,
    'Abell 545': 0.154,
    'Abell 548': 0.042,
    'Abell 85': 0.055,
    'Abell S1063': 0.348,
    'Abell S1121': 0.190,
    'Abell S295': 0.300,
    'ElGordo': 0.870,
    'J0014.3-6604': 0.155,
    'J0027.3-5015': 0.145,
    'J0051.1-4833': 0.187,
    'J0108.5-4020': 0.143,
    'J0117.8-5455': 0.251,
    'J0145.0-5300': 0.188,
    'J0145.2-6033': 0.184,
    'J0212.8-4707': 0.115,
    'J0216.3-4816': 0.163,
    'J0217.2-5244': 0.343,
    'J0225.9-4154': 0.220,
    'J0232.2-4420': 0.284,
    'J0303.7-7752': 0.274,
    'J0314.3-4525': 0.073,
    'J0317.9-4414': 0.075,
    'J0328.6-5542': 0.086,
    'J0336.3-4037': 0.062,
    'J0342.8-5338': 0.060,
    'J0351.1-8212': 0.061,
    'J0352.4-7401': 0.127,
    'J0406.7-7116': 0.229,
    'J0416.7-5525': 0.365,
    'J0431.4-6126': 0.059,
    'J0449.9-4440': 0.172,
    'J0510.2-4519': 0.200,
    'J0516.6-5430': 0.297,
    'J0525.8-4715': 0.191,
    'J0528.9-3927': 0.284,
    'J0540.1-4050': 0.036,
    'J0540.1-4322': 0.085,
    'J0542.8-4100': 0.640,
    'J0543.4-4430': 0.164,
    'J0545.5-4756': 0.130,
    'J0600.8-5835': 0.037,
    'J0607.0-4928': 0.056,
    'J0610.5-4848': 0.243,
    'J0616.8-4748': 0.116,
    'J0625.2-5521': 0.121,
    'J0626.3-5341': 0.051,
    'J0627.2-5428': 0.051,
    'J0631.3-5610': 0.054,
    'J0637.3-4828': 0.203,
    'J0638.7-5358': 0.233,
    'J0645.4-5413': 0.167,
    'J0658.5-5556': 0.296,
    'J0712.0-6030': 0.032,
    'J0738.1-7506': 0.111,
    'J0745.1-5404': 0.074,
    'J0757.7-5315': 0.043,
    'J0812.5-5714': 0.062,
    'J0820.9-5704': 0.061,
    'J0943.4-7619': 0.199,
    'J0948.6-8327': 0.198,
    'J1040.7-7047': 0.061,
    'J1130.0-4213': 0.155,
    'J1145.6-5420': 0.155,
    'J1201.0-4623': 0.118,
    'J1240.2-4825': 0.152,
    'J1358.9-4750': 0.074,
    'J1410.4-4246': 0.049,
    'J1423.7-5412': 0.300,
    'J1518.3-4632': 0.056,
    'J1535.1-4658': 0.036,
    'J1539.5-8335': 0.073,
    'J1601.7-7544': 0.153,
    'J1645.4-7334': 0.069,
    'J1653.0-5943': 0.048,
    'J1705.1-8210': 0.074,
    'J1840.6-7709': 0.019,
    'J2023.4-5535': 0.232,
    'J2104.9-8243': 0.097,
    'J2222.2-5235': 0.174,
    'J2319.2-6750': 0.029,
    'J2340.1-8510': 0.193,
    'MACSJ 0025.4-1222B': 0.584,
    'MACS J0257.6-2209': 0.322,
    'MACSJ0417.5-1155': 0.440,
    'PLCK G200.9-28.2': 0.220,
    'RXCJ0225.1-22928': 0.060,
    'RXCJ0510.7-0801': 0.220,

}


def convert_w4_mag_to_flux_density(magnitude_w4):
    zero_point_flux_jy = 8.283  # Wright+2010
    return zero_point_flux_jy * 10**(-magnitude_w4 / 2.5)

def compute_nuLnu_from_flux_density(flux_density_jy, redshift):
    D_L_cm = cosmo.luminosity_distance(redshift).to(u.cm).value
    flux_density_cgs = flux_density_jy * 1e-23  # Jy to erg/s/cm^2/Hz
    freq_hz = (c / (22 * u.micron)).to(u.Hz).value
    return freq_hz * (4 * np.pi * D_L_cm**2 * flux_density_cgs)

def estimate_sfr_from_luminosity(luminosity_nuLnu):
    return 2.04e-43 * luminosity_nuLnu  # Cluver+2017

def calculate_star_formation_rate(w4_mag, redshift):
    flux_density = convert_w4_mag_to_flux_density(w4_mag)
    nuLnu = compute_nuLnu_from_flux_density(flux_density, redshift)
    return estimate_sfr_from_luminosity(nuLnu)

def query_with_retry(coord, retries=5, base_delay=1.0, max_delay=30):
    for attempt in range(retries):
        try:
            result = Vizier.query_region(coord, radius=4*u.arcsec, catalog=vizier_catalog)
            time.sleep(0.1)  # polite delay after successful request
            return result
        except Exception as e:
            wait = min(max_delay, base_delay * 2 ** attempt + random.uniform(0, 1))
            print(f"Retry {attempt + 1}/{retries} after error: {e}. Waiting {wait:.1f}s.")
            time.sleep(wait)
    print(f"All {retries} retries failed for coordinate {coord}.")
    return None

# Load table
full_table = Table.read(input_file)

# Main loop per cluster
for cluster_name, z in clusters.items():
    print(f"\nProcessing cluster: {cluster_name} (z={z})")
    start_time = time.time()
    
    subset = full_table[full_table['Field'] == cluster_name]
    if len(subset) == 0:
        print(f"No sources found for {cluster_name}. Skipping.")
        continue

    all_entries = []

    for start_idx in range(0, len(subset), chunk_size):
        chunk = subset[start_idx:start_idx + chunk_size]
        meer_coords = SkyCoord(ra=chunk['RA_deg'] * u.deg, dec=chunk['Dec_deg'] * u.deg)
        all_matches = []
    
        print(f"Processing {len(meer_coords)} rows for {cluster_name}...")
        
        
        for i, coord in enumerate(tqdm(meer_coords, desc=f"{cluster_name} Chunk {start_idx}")):
            result = query_with_retry(coord)
            if result:
                r = result[0]
                r['match_RA_deg'] = coord.ra.deg
                r['match_Dec_deg'] = coord.dec.deg
                all_matches.append(r)
        
        
        if not all_matches:
            continue

        combined = Table(np.hstack(all_matches))
        unwise_coords = SkyCoord(ra=combined['RAJ2000'] * u.deg, dec=combined['DEJ2000'] * u.deg)
        match_coords = SkyCoord(ra=combined['match_RA_deg'] * u.deg, dec=combined['match_Dec_deg'] * u.deg)
        
        idx, d2d, _ = match_coords.match_to_catalog_sky(unwise_coords)
        sep_constraint = d2d <= 2 * u.arcsec
        matched = combined[sep_constraint]

        if len(matched) == 0:
            continue

        x = (matched['RAJ2000'] - matched['match_RA_deg']) * 3600
        y = (matched['DEJ2000'] - matched['match_Dec_deg']) * 3600
        
        for i in range(len(x)):
            w_mags = [float(matched[col][i]) if col in matched.colnames and matched[col][i] != '--' else np.nan
                      for col in ['W1mag', 'W2mag', 'W3mag', 'W4mag']]
        
            ra_wise = float(matched['RAJ2000'][i])
            dec_wise = float(matched['DEJ2000'][i])
        
            all_entries.append([
                cluster_name,
                round(x[i], 3), np.nan,
                round(y[i], 3), np.nan,
                ra_wise,
                dec_wise,
                *w_mags
            ])
        print(x)

    if not all_entries:
        print(f"No valid matches for {cluster_name}. Skipping SFR calculation.")
        continue

    colnames = ['Field', 'RAdiff_median (arcsec)', 'RAdiff_median_Err (arcsec)',
                'DECdiff_median (arcsec)', 'DECdiff_median_Err (arcsec)',
                'WISE_RA_deg', 'WISE_Dec_deg',
                'W1mag', 'W2mag', 'W3mag', 'W4mag']

    offsets_table = Table(rows=all_entries, names=colnames)

    cluster_basename = cluster_name.replace(' ', '_')
    offsets_path = os.path.join(output_dir, f'{cluster_basename}_offsets_copy.csv')
    sfr_path = os.path.join(output_dir, f'{cluster_basename}_sfr.csv')


    offsets_table.write(offsets_path, format='ascii.csv', overwrite=True)
    print(f"Offsets saved to {offsets_path}")

    # Calculate SFRs
    df = pd.read_csv(offsets_path)
    df['SFR_Msun_per_yr'] = df['W4mag'].apply(
        lambda mag: calculate_star_formation_rate(mag, z) if np.isfinite(mag) else np.nan
    )
    df.to_csv(sfr_path, index=False)
    print(f"SFRs saved to {sfr_path}")
    print(f"Finished {cluster_name} in {(time.time() - start_time) / 60:.2f} minutes")

print("\nAll clusters processed.")
