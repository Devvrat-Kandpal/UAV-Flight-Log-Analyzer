# UAV Flight Log Analyzer

A Python-based tool for analyzing PX4 UAV flight telemetry and visualizing flight behavior using recorded log data.

## Overview

UAV flight logs contain valuable information about a vehicle's movement, altitude, speed, and GPS conditions. However, interpreting raw telemetry can be tedious without processing and visualization tools.

**UAV Flight Log Analyzer** processes telemetry exported from PX4 flight logs into readable flight summaries and visual plots. The project focuses on understanding flight behavior from recorded data, with an emphasis on trajectory visualization, flight-event estimation, and GPS quality analysis.

The analyzer was developed using telemetry from a real PX4 flight log recorded with the legacy `sdlog2` logging format.

## Features

- **Flight Summary:** Calculates key metrics such as estimated flight duration, distance traveled, altitude range, and ground speed.
- **Flight Event Estimation:** Heuristically estimates takeoff, the start of final descent, and landing.
- **Ground Track Visualization:** Plots the recorded GPS trajectory to help inspect the vehicle's flight path.
- **Altitude Profile:** Visualizes altitude changes throughout the recording.
- **Ground Speed Profile:** Shows how ground speed changes over time.
- **GPS Quality Analysis:** Examines GPS fix codes, satellite counts, estimated position error (EPH), and duplicate timestamps.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation and telemetry processing |
| Pandas | Loading, cleaning, and analyzing telemetry data |
| NumPy | Numerical calculations and data processing |
| Matplotlib | Generating flight analysis plots |
| PX4 / `sdlog2` | Source flight-log format |

## How It Works

1. **Telemetry Input:** Reads supported telemetry data exported from a PX4 flight log into CSV files.
2. **Data Processing:** Processes position, GPS, and other supported telemetry fields.
3. **Metric Calculation:** Calculates flight statistics and estimates relevant flight events.
4. **Visualization:** Generates plots for the ground track, altitude, speed, and GPS quality.
5. **Output:** Prints a flight summary in the terminal and saves generated plots locally.

## Getting Started

### Requirements

- Python 3.10 or newer
- pandas
- NumPy
- Matplotlib

### Installation

Clone the repository:

```bash
git clone https://github.com/Devvrat-Kandpal/UAV-Flight-Log-Analyzer.git
cd UAV-Flight-Log-Analyzer
```

Create and activate a virtual environment:

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install pandas numpy matplotlib
```

### Running the Analyzer

Place the required telemetry CSV files in the project directory, then run:

```bash
python analyzer.py
```

The flight summary is printed in the terminal, and the generated plots are saved to the `figures/` directory.

## Limitations

- Takeoff, final descent, and landing are heuristic estimates rather than confirmed flight events.
- The GPOS time axis is interpolated uniformly across the GPS recording duration because usable per-row GPOS timestamps are unavailable in the current analysis.
- GPS quality metrics describe the recorded GPS telemetry and should not be interpreted as a complete assessment of navigation accuracy.
- Results depend on the availability and quality of the required telemetry fields.

## Future Scope

- Improve flight-event detection using configurable thresholds and more robust algorithms.
- Improve timestamp alignment across telemetry streams to produce more accurate time-based analysis.
- Add support for additional PX4 log formats and telemetry export schemas.
- Generate downloadable flight reports containing summary statistics and plots.
- Add automated tests and sample datasets to make the analyzer easier to validate and maintain.
- Develop a command-line interface with configurable input paths and output directories.

## Project Structure

```text
UAV-Flight-Log-Analyzer/
├── analyzer.py
├── .gitignore
└── README.md
```

## License

No license has been specified yet.

---

**Author:** [Devvrat Kandpal](https://github.com/Devvrat-Kandpal)
