# Heat-Stress-Metrics-for-US-Census-Tracts-1998-2020
This project provides hourly, area-weighted and population-weighted Heat Index (HI), Wet-Bulb Globe Temperature (WBGT), and Universal Thermal Climate Index (UTCI) for U.S. Census tract boundaries across the contiguous United States from 1998–2020

## Code overview
**heatstress_prismday_to_tract.py**: reconstructs hourly PRISM-day (12Z–12Z) meteorology from ERA5-Land and PRISM inputs, interpolates meteorology and NSRDB   radiation to the PRISM 800 m grid, computes hourly Heat Index (HI), Wet-Bulb Globe Temperature (WBGT), and Universal Thermal Climate Index (UTCI), aggregates grid-cell values to U.S. Census tracts using area-weighted and population-weighted methods, and writes final tract-level exposure outputs.

**technical_validation.py**: aligns station observations to PRISM-day time windows, maps stations to nearest model grid cells using explicit distance thresholds, validates gridded meteorology against NOAA ISD-Lite, validates HI and WBGT against NOAA USCRN heat01, validates radiation-driven UTCI against SURFRAD station estimates, and produces manuscript-ready validation tables and summary statistics.
