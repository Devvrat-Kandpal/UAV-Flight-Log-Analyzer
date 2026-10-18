from pathlib import Path
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"

REQUIRED_COLUMNS = {
    "GPOS.csv": ["GPOS_Lat", "GPOS_Lon", "GPOS_Alt", "GPOS_VelN", "GPOS_VelE", "GPOS_VelD"],
    "GPS.csv": ["GPS_GPSTime", "GPS_Fix", "GPS_EPH", "GPS_nSat"],
    "STAT.csv": ["STAT_NavState", "STAT_ArmS", "STAT_Failsafe", "STAT_Landed"],
}


def load_data():
    """Load and validate the Version 1 telemetry files."""
    data = {}
    for filename, required in REQUIRED_COLUMNS.items():
        path = BASE_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {filename} in {BASE_DIR}")

        frame = pd.read_csv(path)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{filename} is missing columns: {', '.join(missing)}")
        if frame.empty:
            raise ValueError(f"{filename} contains no rows")
        data[filename.removesuffix(".csv").lower()] = frame

    return data["gpos"], data["gps"], data["stat"]


def prepare_data(gpos, gps):
    """Create elapsed-time axes and ground speed from the available fields."""
    gpos = gpos.copy()
    gps = gps.copy()

    gps["GPS_GPSTime"] = pd.to_numeric(gps["GPS_GPSTime"], errors="coerce")
    gps = gps.dropna(subset=["GPS_GPSTime"]).sort_values("GPS_GPSTime")
    if len(gps) < 2:
        raise ValueError("GPS.csv needs at least two valid GPS_GPSTime values")

    duration_seconds = (gps["GPS_GPSTime"].iloc[-1] - gps["GPS_GPSTime"].iloc[0]) / 1_000_000
    if duration_seconds <= 0:
        raise ValueError("GPS timestamps do not span a positive duration")

    duration_min = duration_seconds / 60
    gpos["time_min"] = np.linspace(0, duration_min, len(gpos))
    gps["time_min"] = (gps["GPS_GPSTime"] - gps["GPS_GPSTime"].iloc[0]) / 60_000_000

    numeric_columns = ["GPOS_Lat", "GPOS_Lon", "GPOS_Alt", "GPOS_VelN", "GPOS_VelE"]
    for column in numeric_columns:
        gpos[column] = pd.to_numeric(gpos[column], errors="coerce")

    for column in ["GPS_Fix", "GPS_EPH", "GPS_nSat"]:
        gps[column] = pd.to_numeric(gps[column], errors="coerce")

    gpos["ground_speed"] = np.hypot(gpos["GPOS_VelN"], gpos["GPOS_VelE"])
    return gpos, gps, duration_min


def identify_flight_events(gpos):
    """Estimate takeoff, final descent, and landing from sustained telemetry patterns."""
    altitude = gpos["GPOS_Alt"].to_numpy(dtype=float)
    vertical_speed = gpos["GPOS_VelD"].to_numpy(dtype=float)
    ground_speed = gpos["ground_speed"].to_numpy(dtype=float)
    times = gpos["time_min"].to_numpy(dtype=float)
    sample_rate = (len(gpos) - 1) / max(times[-1] * 60, 1e-9)
    smooth_altitude = pd.Series(altitude).rolling(11, center=True, min_periods=1).median().to_numpy()

    def sustained_starts(mask, seconds):
        run_length = max(1, round(sample_rate * seconds))
        values = np.asarray(mask, dtype=bool)
        if len(values) < run_length:
            return np.array([], dtype=int), run_length
        counts = np.convolve(values.astype(int), np.ones(run_length, dtype=int), mode="valid")
        return np.flatnonzero(counts >= run_length), run_length

    # PX4's Down velocity is negative while climbing and positive while descending.
    takeoff_candidates, ascent_rows = sustained_starts(vertical_speed < -0.25, 1.5)
    takeoff_idx = None
    for candidate in takeoff_candidates:
        confirm_idx = min(len(gpos) - 1, int(candidate + ascent_rows))
        if smooth_altitude[confirm_idx] - smooth_altitude[candidate] >= 0.5:
            takeoff_idx = int(candidate)
            break
    if takeoff_idx is None:
        raise ValueError("Could not identify takeoff from sustained vertical climb")

    # Estimate ground altitude from the few seconds immediately before takeoff.
    baseline_start = max(0, takeoff_idx - round(sample_rate * 5))
    baseline = float(np.nanmedian(altitude[baseline_start:takeoff_idx]))
    peak_idx = int(np.nanargmax(altitude))

    # Require a stable post-climb altitude period before looking for the final descent.
    stable_rows = max(2, round(sample_rate * 20))
    altitude_spread = pd.Series(smooth_altitude).rolling(stable_rows, min_periods=stable_rows).std().to_numpy()
    stable_candidates = np.flatnonzero((np.arange(len(gpos)) > peak_idx) & (altitude_spread <= 2.5))
    cruise_idx = int(stable_candidates[0]) if len(stable_candidates) else min(
        len(gpos) - 1, peak_idx + round(sample_rate * 30)
    )

    descent_candidates, descent_rows = sustained_starts(vertical_speed > 1.2, 1.5)
    descent_idx = None
    for candidate in descent_candidates:
        if candidate < cruise_idx:
            continue
        confirm_idx = min(len(gpos) - 1, int(candidate + descent_rows))
        if smooth_altitude[candidate] - smooth_altitude[confirm_idx] >= 2.5:
            descent_idx = int(candidate)
            break
    if descent_idx is None:
        raise ValueError("Could not identify final descent from sustained downward velocity")

    # Landing requires near-ground altitude and low horizontal speed for two seconds.
    landing_candidates, _ = sustained_starts(
        (smooth_altitude <= baseline + 1.5) & (ground_speed < 2.0), 2.0
    )
    valid_landings = landing_candidates[landing_candidates >= descent_idx]
    landing_idx = int(valid_landings[0]) if len(valid_landings) else len(gpos) - 1

    if not takeoff_idx < descent_idx < landing_idx:
        raise ValueError("Detected flight events are not in chronological order")

    return float(times[takeoff_idx]), float(times[descent_idx]), float(times[landing_idx])


def haversine_distances_m(latitude, longitude):
    """Calculate distances between consecutive valid GPS coordinates."""
    lat = np.radians(np.asarray(latitude, dtype=float))
    lon = np.radians(np.asarray(longitude, dtype=float))
    distances = np.zeros(len(lat), dtype=float)

    if len(lat) < 2:
        return distances

    valid = np.isfinite(lat) & np.isfinite(lon)
    for i in range(1, len(lat)):
        if valid[i - 1] and valid[i]:
            dlat = lat[i] - lat[i - 1]
            dlon = lon[i] - lon[i - 1]
            a = math.sin(dlat / 2) ** 2 + math.cos(lat[i - 1]) * math.cos(lat[i]) * math.sin(dlon / 2) ** 2
            distances[i] = 2 * 6_371_000 * math.asin(min(1.0, math.sqrt(a)))

    return distances


def calculate_metrics(gpos, gps, stat, flight_start, flight_end, duration_min):
    flight = gpos[gpos["time_min"].between(flight_start, flight_end)].copy()
    if flight.empty:
        flight = gpos.copy()

    distances = haversine_distances_m(flight["GPOS_Lat"], flight["GPOS_Lon"])
    speeds = flight["ground_speed"].dropna()
    altitudes = flight["GPOS_Alt"].dropna()

    fix_mode = gps["GPS_Fix"].mode()
    return {
        "recording_duration_min": duration_min,
        "estimated_flight_duration_min": max(0.0, flight_end - flight_start),
        "distance_km": float(np.nansum(distances) / 1000),
        "max_altitude_amsl_m": float(altitudes.max()) if not altitudes.empty else float("nan"),
        "min_altitude_amsl_m": float(altitudes.min()) if not altitudes.empty else float("nan"),
        "peak_ground_speed_mps": float(speeds.max()) if not speeds.empty else float("nan"),
        "average_ground_speed_mps": float(speeds.mean()) if not speeds.empty else float("nan"),
        "gps_fix_mode": fix_mode.iloc[0] if not fix_mode.empty else float("nan"),
        "mean_satellites": float(gps["GPS_nSat"].mean()),
        "mean_eph_m": float(gps["GPS_EPH"].mean()),
        "gps_duplicate_timestamps": int(gps["GPS_GPSTime"].duplicated().sum()),
        "status_rows": len(stat),
        "armed_rows": int((pd.to_numeric(stat["STAT_ArmS"], errors="coerce") > 0).sum()),
        "failsafe_rows": int((pd.to_numeric(stat["STAT_Failsafe"], errors="coerce") > 0).sum()),
    }


def save_ground_track(gpos, flight_start, flight_end):
    data = gpos[gpos["time_min"].between(flight_start, flight_end)].dropna(subset=["GPOS_Lat", "GPOS_Lon"])
    if data.empty:
        data = gpos.dropna(subset=["GPOS_Lat", "GPOS_Lon"])

    fig, ax = plt.subplots(figsize=(9, 7))
    points = ax.scatter(data["GPOS_Lon"], data["GPOS_Lat"], c=data["ground_speed"], s=10, cmap="viridis")
    ax.scatter(data["GPOS_Lon"].iloc[0], data["GPOS_Lat"].iloc[0], marker="o", s=90, label="Start")
    ax.scatter(data["GPOS_Lon"].iloc[-1], data["GPOS_Lat"].iloc[-1], marker="X", s=90, label="End")
    fig.colorbar(points, ax=ax, label="Ground speed (m/s)")
    ax.set(title="Estimated Flight Ground Track", xlabel="Longitude (°)", ylabel="Latitude (°)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ground_track.png", dpi=160)
    plt.close(fig)


def save_altitude_profile(gpos, flight_start, descent_start, flight_end):
    data = gpos[gpos["time_min"].between(flight_start, flight_end)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data["time_min"], data["GPOS_Alt"], linewidth=1.2)
    ax.axvline(flight_start, color="green", linestyle="--", label="Estimated takeoff")
    ax.axvline(descent_start, color="orange", linestyle="--", label="Start of final descent")
    ax.axvline(flight_end, color="red", linestyle="--", label="Estimated landing")
    ax.set(title="Altitude Profile", xlabel="Estimated elapsed time (min)", ylabel="Altitude AMSL (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "altitude_profile.png", dpi=160)
    plt.close(fig)


def save_speed_profile(gpos, flight_start, descent_start, flight_end):
    data = gpos[gpos["time_min"].between(flight_start, flight_end)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data["time_min"], data["ground_speed"], linewidth=1.2)
    ax.axvline(flight_start, color="green", linestyle="--", label="Estimated takeoff")
    ax.axvline(descent_start, color="orange", linestyle="--", label="Start of final descent")
    ax.axvline(flight_end, color="red", linestyle="--", label="Estimated landing")
    ax.set(title="Ground-Speed Profile", xlabel="Estimated elapsed time (min)", ylabel="Ground speed (m/s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "speed_profile.png", dpi=160)
    plt.close(fig)


def save_gps_quality(gps):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gps["time_min"], gps["GPS_nSat"], label="Satellites", linewidth=1.2)
    ax.set(xlabel="Elapsed time (min)", ylabel="Satellite count", title="GPS Quality")
    ax.grid(True, alpha=0.3)

    second_axis = ax.twinx()
    second_axis.plot(gps["time_min"], gps["GPS_EPH"], label="EPH", linewidth=1.0, alpha=0.8)
    second_axis.set_ylabel("Horizontal position error estimate (m)")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = second_axis.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gps_quality.png", dpi=160)
    plt.close(fig)


def print_summary(metrics, flight_start, descent_start, flight_end):
    print("\nUAV FLIGHT LOG SUMMARY")
    print("=" * 34)
    print(f"Recording duration:          {metrics['recording_duration_min']:.2f} min")
    print(f"Estimated takeoff:            {flight_start:.2f} min")
    print(f"Start of final descent:       {descent_start:.2f} min")
    print(f"Estimated landing:            {flight_end:.2f} min")
    print(f"Estimated flight duration:   {metrics['estimated_flight_duration_min']:.2f} min")
    print(f"Distance during window:      {metrics['distance_km']:.3f} km")
    print(f"Maximum altitude (AMSL):     {metrics['max_altitude_amsl_m']:.2f} m")
    print(f"Minimum altitude (AMSL):     {metrics['min_altitude_amsl_m']:.2f} m")
    print(f"Peak ground speed:           {metrics['peak_ground_speed_mps']:.2f} m/s")
    print(f"Average ground speed:        {metrics['average_ground_speed_mps']:.2f} m/s")
    print(f"Most common GPS fix code:    {metrics['gps_fix_mode']}")
    print(f"Mean satellite count:        {metrics['mean_satellites']:.2f}")
    print(f"Mean GPS EPH:                {metrics['mean_eph_m']:.2f} m")
    print(f"Duplicate GPS timestamps:    {metrics['gps_duplicate_timestamps']}")
    print(f"STAT status rows:            {metrics['status_rows']}")
    print(f"STAT armed rows:             {metrics['armed_rows']}")
    print(f"STAT failsafe rows:          {metrics['failsafe_rows']}")
    print(f"\nPlots saved to: {FIGURES_DIR}")
    print("\nNote: GPOS timestamps are estimated uniformly across the GPS recording duration.")
    print("Takeoff, final descent, and landing are heuristic estimates; STAT has no row timestamps.")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        gpos, gps, stat = load_data()
        gpos, gps, duration_min = prepare_data(gpos, gps)
        flight_start, descent_start, flight_end = identify_flight_events(gpos)
        metrics = calculate_metrics(gpos, gps, stat, flight_start, flight_end, duration_min)

        save_ground_track(gpos, flight_start, flight_end)
        save_altitude_profile(gpos, flight_start, descent_start, flight_end)
        save_speed_profile(gpos, flight_start, descent_start, flight_end)
        save_gps_quality(gps)
        print_summary(metrics, flight_start, descent_start, flight_end)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
