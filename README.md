# fountains

> **A score is never a reason to leave with less water.** Carry what you need for the whole ride as if every fountain were dry. This tool says nothing about whether the water is safe to drink.

Will the fountains along your ride be running on the day you ride?

You export your route as a GPX file, run one command, and get every drinking-water point that OpenStreetMap knows within a short detour of the route, each with a band (**likely**, **uncertain** or **unlikely** to be running), the reason, and the date anyone last checked it. The same list goes onto your bike computer as waypoints. The screen part takes a minute; the rest happens outside.

It works in Corsica only, because that is where the model learned.

**Built with PriorLabs-TabPFN.**

## Install

Python 3.10 or newer. Everything runs on a CPU; no GPU, no account, no API key.

```bash
git clone https://github.com/dsiacci/fountains && cd fountains
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # the CPU build, much smaller
pip install .
```

The first run downloads the TabPFN v2 classifier weights (29 MB, from Hugging Face, `Prior-Labs/TabPFN-v2-clf`) and the Météo-France daily file of the current year (about 1 MB, refreshed every 6 hours).

## Use

```bash
fountains score my-ride.gpx --date 2026-10-11 --html
```

- `--date`: the day you ride (default: today). Rain measured up to the last Météo-France report is used, then the Météo-France forecast for the days in between. Forecast days that are not available yet count as dry, so a missing forecast never makes a fountain look wetter.
- `--max-detour 250`: how far off the route you are willing to go, one way, along roads and paths, in metres.
- `--out out/`: where the files go.

It prints the table and writes four files:

| File | For |
|---|---|
| `<track>-<date>.txt` | the table above, to read before leaving |
| `<track>-<date>-waypoints.gpx` | waypoints for a bike computer, named like `Funtana Vechja - likely`, with the reason in the description |
| `<track>-<date>.geojson` | everything, for a map or another tool |
| `<track>-<date>.html` | a map (with `--html`) |

Every point within the detour limit is listed, in riding order. None is hidden, none is ranked. Points that are close as the crow flies but further than the limit by road or path are listed in a second section with their real detour. Points tagged `access=private` or `access=no` are left out and counted; `access=customers` is shown with a note.

Other commands:

- `fountains trim ride.gpx shared.gpx --start 1500 --end 1500` cuts the first and last 1.5 km of a track and drops every timestamp and author field, before you share a track.
- `fountains check-memories memories.csv` compares the bands with what someone remembers seeing (see below).
- `fountains calibrate` and `fountains build-data --all` rebuild the model's measurements and the files in `data/` from the open sources.

## How it works

### 1. The fountains

From OpenStreetMap, through the Overpass API: every `amenity=drinking_water` and `amenity=water_point`, plus springs, taps, wells and decorative fountains tagged `drinking_water=yes` or `conditional`, minus anything tagged `drinking_water=no`. On 6 October 2026 that is 608 points in Corsica. Only 2 of them say whether they are seasonal (`seasonal=*`), and 33 carry a date when someone checked them (`check_date`, `survey:date`). Nobody records whether a fountain runs, which is the gap this tool tries to fill.

Each point is attached to the nearest road or path, and the detour is measured along the network from the route. A fountain 40 m from the road as the crow flies can be 400 m away if the only access is a loop through the village.

### 2. The rain

- **Measured**: Météo-France daily rain gauges in Corsica (103 gauges reported since 2010, 52 of them still active). The rain at a fountain is interpolated from the three nearest gauges that reported that day, weighted by the inverse square of the distance.
- **Normal**: for each gauge, the mean rain of the 30, 90 and 180 days before each day of the year over 1991-2020. That turns "45 mm in 90 days" into "38 % of normal".
- **Forecast**: the Météo-France models (AROME, ARPEGE), at the fountain, through Open-Meteo, for the days between the last gauge report and the ride.

Altitude is not used to correct the interpolation; it is given to the model as its own input (IGN altitude of the point).

### 3. The model: TabPFN, given 3,831 stream observations

Nobody publishes whether fountains run. What France does publish is whether small streams run: every summer since 2012, agents of the Office français de la biodiversité walk to the same points on small streams and record what they see (the ONDE network). In Corsica that is 33 streams and 3,831 usable observations from 2012 to 2026.

The model answers one question: *is flowing water visible?* ONDE records four states; "flowing, normal" and "flowing, weak" count as yes, "water but no visible flow" and "dry" count as no. A fountain fills a bottle only if water flows, and this is also the only split that means the same thing in every year: from 2016 to 2019 and in 2022, Corsican observers recorded only "flowing", without the normal/weak distinction.

Inputs: the day of the year, the altitude, the rain of the last 30, 90 and 180 days [and how it compares with the 1991-2020 normal: *kept or dropped by the validation below*].

[TabPFN](https://github.com/PriorLabs/TabPFN) (Prior Labs) is a tabular foundation model: a transformer pre-trained on millions of synthetic tables, which takes the training rows as context and predicts new rows in one forward pass. There is no training loop, no hyperparameter search, and it returns probabilities that are usually well calibrated, which is what this tool needs. It runs here on a CPU with the open v2 weights.

### 4. How far to trust it

*Leave-one-station-out*: each of the 33 streams is hidden in turn, the model gets the other 32 as context, and predicts the hidden one, which it has never seen. Pooled, these predictions give the Brier score, a comparison with simpler baselines (the rate for the month, a logistic regression on the same inputs) and the reliability curve: among the cases the model scored around *p*, how many were flowing.

The bands come from that curve and from targets fixed before the first run: **likely** where at least 90 % of the hidden cases were flowing, **unlikely** where at most 50 % were, **uncertain** in between. The tool shows bands, not percentages, because a percentage measured on streams would claim more than we know about fountains.

*Results: filled in after the run (`data/calibration.json`, `docs/reliability.svg`).*

### 5. The weak link: a fountain is not a stream

The model assumes that a fountain fed by a spring dries up like a small headwater stream after the same weather. That is plausible for a village fountain on a shallow spring, wrong for a fountain on the town mains (which runs whatever the rain), and unknown for a deep spring that reacts months later. OpenStreetMap almost never says which is which.

This assumption is tested on real fountains in two ways, both small:

- **Memories**: what the author remembers seeing at the fountains of his usual routes in September 2026 (`fountains check-memories`). Memories are not measurements, and the sample is small. The model is never adjusted to agree with them.
- **A ride**: the bands computed before a ride, and what the fountains actually did.

*Results: filled in after the checks.*

## Limits

- **Streams are not fountains** (above). This is the main one.
- **Only what OpenStreetMap knows.** A fountain that is not mapped does not exist for this tool, and positions can be off by tens of metres.
- **Rain is interpolated** between gauges that can be 10 to 20 km away and hundreds of metres lower; mountain rain is underestimated.
- **The forecast** reaches 2 to 4 days ahead; beyond that, the days count as dry.
- **Potability**: never assessed. `drinking_water=yes` in OpenStreetMap is what a mapper wrote, not a water test.
- **Corsica only**: the stream observations and the rain gauges are Corsican; the tool refuses tracks elsewhere.
- **CPU time**: TabPFN reads its 3,831 rows of context at each run; on a small 2-core machine a score takes one to two minutes.

## Privacy

The tool reads a GPX file that you export yourself, from any app. It never connects to Strava or any other account, and your track is never used to train anything. The track itself never leaves your machine: to fetch roads and paths, the tool sends Overpass the positions of the fountains it found near your route (public OpenStreetMap points); Open-Meteo and IGN receive the same positions.

## Data sources and licenses

| What | Source | License | How it is used |
|---|---|---|---|
| Drinking-water points, roads and paths | © OpenStreetMap contributors, via the Overpass API | ODbL 1.0 | `data/fountains-corsica.geojson` (extract of 6 October 2026); roads fetched at run time |
| Stream observations | Office français de la biodiversité, réseau ONDE, via Hub'Eau (`/api/v1/ecoulement`) | Licence Ouverte 2.0 (etalab-2.0) | `data/onde-observations.csv`, `data/training.csv` |
| Daily rain | Météo-France, « Données climatologiques de base - quotidiennes » (data.gouv.fr), department 20 | Licence Ouverte 2.0. Source: Météo-France | normals and training table in `data/`; current-year file downloaded at run time |
| Rain forecast | Météo-France models through Open-Meteo (`/v1/meteofrance`) | CC BY 4.0 | at run time |
| Altitude | IGN, Géoplateforme altimetry service | Licence Ouverte 2.0 | training table; fountains at run time |
| Model weights | Prior Labs, TabPFN v2 classifier (`Prior-Labs/TabPFN-v2-clf`) | Prior Labs License 1.1 (Apache 2.0 with an attribution clause), copy in `licenses/` | downloaded at run time |

The code is under the Apache License 2.0 (`LICENSE`, `NOTICE`). The tool only reads OpenStreetMap; it never edits it.

## Tests

```bash
pip install pytest && pytest
```

The tests cover the geometry, the OpenStreetMap selection rule, the detour along the network, the rain windows and normals, and the rules the tool must never break: every point is shown, in riding order, with the warning, in every output.
