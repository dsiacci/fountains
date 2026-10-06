# Data

Everything here comes from open sources and can be rebuilt with
`fountains build-data --all`. Each file keeps the license of its source.

| File | What | Source | License |
|---|---|---|---|
| `fountains-corsica.geojson` | Drinking-water points in Corsica (selection rule in the main README), with the tags the tool uses | OpenStreetMap via the Overpass API | ODbL 1.0, © OpenStreetMap contributors |
| `onde-observations.csv` | Every usable observation of the ONDE network in Corsica: station, date, flow class, label | OFB, Observatoire national des étiages (ONDE), via Hub'Eau (`/api/v1/ecoulement`) | Licence Ouverte 2.0 (etalab-2.0) |
| `gauge-normals-1991-2020.json` | For each Météo-France rain gauge with enough data, the mean rain of the 30, 90 and 180 days before each day of the year, over 1991-2020 | Computed from Météo-France « Données climatologiques de base - quotidiennes », department 20 | Licence Ouverte 2.0. Source: Météo-France |
| `training.csv` | The model's context: one row per ONDE observation, with the rain before that day at the station (interpolated from the gauges), the ratio to the 1991-2020 normal, the day of the year, the altitude and the label | Built from the three sources above and IGN altitudes | Licence Ouverte 2.0 |
| `calibration.json` | Leave-one-station-out results: Brier scores, reliability table, band thresholds | Computed by `fountains calibrate` | Apache-2.0, like the code |

The Météo-France daily file of the current year is downloaded at run time
(about 1 MB, refreshed every 6 hours), so the score uses the rain measured up
to the day before yesterday or yesterday.
