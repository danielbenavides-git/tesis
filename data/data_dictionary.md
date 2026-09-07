# Data Dictionary

Source: XM S.A. E.S.P. - Sinergox API (`servapibi.xm.com.co`)
Downloaded via `pydataxm` (v0.3.14). No authentication required.
Download script: `scripts/download_xm_data.py`

## Raw Data Summary

All files use the `Sistema` entity (national aggregates).
Date range target: 2000-01-01 to 2026-07-31.

### Hourly files

One row per date, with columns `Values_Hour01` through `Values_Hour24`.
Additional columns: `Id` (always "Sistema"), `Values_code` ("Sistema"), `Date`.

| File | MetricId | Rows | Date range | Nulls | Unit | Description |
|---|---|---|---|---|---|---|
| `precio_bolsa_horario.csv` | PrecBolsNaci | 9,709 | 2000-01 to 2026-07 | 0 | $/kWh | Spot price per hour. Primary input for CWT scalograms. |
| `generacion_real_horaria.csv` | Gene | 9,709 | 2000-01 to 2026-07 | 0 | kWh | Total real generation dispatched to the SIN. |
| `demanda_real_horaria.csv` | DemaReal | 9,709 | 2000-01 to 2026-07 | 0 | kWh | Real demand from regulated and non-regulated users. |
| `max_precio_oferta_horario.csv` | MaxPrecOferNal | 6,185 | 2009-08 to 2026-07 | 19 | $/kWh | Highest offer price accepted in the dispatch. |

### Daily files

One row per date with columns: `Id` ("Sistema"), `Value`, `Date`.

| File | MetricId | Rows | Date range | Nulls | Unit | Description |
|---|---|---|---|---|---|---|
| `precio_bolsa_diario_ponderado.csv` | PPPrecBolsNaci | 9,709 | 2000-01 to 2026-07 | 0 | $/kWh | Demand-weighted average of hourly spot price. |
| `aportes_energia_diario.csv` | AporEner | 9,709 | 2000-01 to 2026-07 | 0 | kWh | Daily hydro inflows to the SIN in energy equivalent. |
| `porcentaje_volumen_util_diario.csv` | PorcVoluUtilDiar | 9,709 | 2000-01 to 2026-07 | 0 | % | Aggregate reservoir level as percentage of useful volume. |
| `volumen_util_diario.csv` | VoluUtilDiarEner | 9,709 | 2000-01 to 2026-07 | 0 | GWh | Aggregate reservoir level in energy equivalent. |
| `demanda_sin_diaria.csv` | DemaSIN | 9,709 | 2000-01 to 2026-07 | 0 | kWh | Total daily energy demand of the SIN. |
| `precio_escasez_superior.csv` | PrecEscaSup | 518 | 2025-03 to 2026-07 | 0 | $/kWh | Scarcity price threshold. |

## Role in the project

Each variable maps to a specific stage of the analysis.

**Primary input (CWT scalograms)**
`precio_bolsa_horario.csv` is the only input to the wavelet transform. Hourly spot prices are segmented into overlapping windows, transformed via CWT, and the resulting scalograms become inputs to the VAE.

**Regime validation (Interpretation and Validation)**
Once the VAE latent space is clustered into regimes, the following variables help identify what each regime represents:

- `aportes_energia_diario.csv` and `porcentaje_volumen_util_diario.csv` - direct proxies for hydrological stress. Low inflows and low reservoir levels correlate with El Nino episodes (2002-03, 2009-10, 2015-16, 2023-24) and historically precede price spikes.
- `generacion_real_horaria.csv` - reveals the generation mix shift. When hydro capacity drops, thermal plants increase output, which raises marginal costs and spot prices.
- `demanda_real_horaria.csv` and `demanda_sin_diaria.csv` - context for distinguishing price increases driven by supply constraints (regime-relevant) from those driven by demand growth (less relevant).
- `precio_bolsa_diario_ponderado.csv` - smoothed daily reference for quick time-series plots and correlation analysis.

**Limited coverage**
- `precio_escasez_superior.csv` covers only 2025-03 onward. Not usable for historical validation. Included for completeness.
- `max_precio_oferta_horario.csv` starts 2009-08. Usable for the second half of the dataset.

## Notes

- Hourly files encode hours as columns (`Values_Hour01`..`Values_Hour24`), not rows, subsequet data processing follows in the project
- All monetary values are in Colombian pesos (COP) per kWh, nominal (not inflation-adjusted)
- The API returns at most 30 days per request for hourly variables. `pydataxm` handles this internally
