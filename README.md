# RL-as-a-star-formation-tracer

This repository contains the code I used to obtain results for my paper (in prep) titled 'Environmental Dependence of Radio Continuum as a Star Formation Tracer'.

We use machine learning techniques applied to observational data from VLA, COSMOS, and allWISE catalogues to invesitage the reliabity of 1.4GHz radio luminosity is as a tracer of star formation within galaxies in a multitude of environments; regardless of morphology, mass, luminosity, and other galaxy characteristics.

In short - we find that 1.4GHz radio luminosity is not an accurate tracer of star formation in any environment, however, the effectiveness of its tracer capabilities becomes even less reliable at flux densities above ~0.2mJy. This is the flux density where above which, we see a larger amout of AGN in the COSMOS sample of galaxies, and it is where we expect a stronger contribution from other non star forming processes that still contribute to the overall radio continuum emission.

**offsets_iter.py** - This script processes astronomical data for a list of galaxy clusters. It reads a catalog of radio sources, then queries the AllWISE catalog to find matching infrared sources. For each matched pair, the script calculates the positional offset and uses the WISE W4-band magnitude to estimate the star formation rate (SFR) of the source, based on the cluster's redshift. Finally, it saves the calculated offsets and SFRs into separate CSV files for each cluster. Originally written by Kabelo McKabuza, heavily modified by myself.

**Predict_IR_SFR_from_Radio_Data.py** - This script loads radio astronomy data from the VLA-COSMOS 3 GHz survey and uses various machine learning regression models — including Random Forest, Gradient Boosting, Neural Networks, Support Vector Regression, and others — to predict infrared star formation rates (IR SFR) from radio observations.

It includes utilities for:

Converting astronomical measurements (e.g., luminosity ↔ flux, angular distances)
Splitting and scaling datasets
Training and evaluating multiple regressors
Plotting predictions, residuals, and redshift trends

**plot_rl_vs_sfr.py** - This is a standalone script which creates a plot showing the correlation between 1.4GHz radio luminosity, and infrared–derived star formation rates for our VLA-COSMOS galaxies. The plot also contains information about the correlation strength, 1-sigma confidence band, our linear best fit, and the canonial linear relation between 1.4GHz radio luminosity and (IR derived) SFRs. This plot demonstrates how overestimated and underestimated SFRs are when using the canonical linear relation. It is run through the terminal, instructions are in the code comments.
