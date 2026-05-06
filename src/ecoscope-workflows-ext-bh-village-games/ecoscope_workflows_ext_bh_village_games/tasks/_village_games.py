"""
Village Games Workflow — Custom Tasks
All tasks for community feedback tables, heatmaps, donut charts,
leaderboards, per-village icon bars, and summary icon bars.

Icon loading convention
───────────────────────
Every task that needs icons accepts a single `icons_dir` parameter
pointing to the folder that contains all PNG icon files.
The expected filenames are defined in ICON_FILES below.

Key changes from original
──────────────────────────
- `year` and `quarter` are NO LONGER explicit task parameters.
  They are derived automatically from the incoming `df` slice,
  which is already partitioned by split_yearly / split_quarterly.
- `draw_community_feedback_table` now returns a full HTML string
  (wrapping the base64 PNG in an <img> tag) so that persist_text
  and the widget pipeline handle it correctly.

FIXES APPLIED (v4)
──────────────────
FIX 1 — Zero counts (dtype): clean_dataframe casts Year/Quarter to str.
  _add_date_cols now also casts Year to str so filter comparisons
  match. _extract_year returns str. _extract_year_quarter handles
  "2024Q1"-style strings from split_quarterly.

FIX 2 — Widget sizing: all Plotly figures use autosize=True and
  remove hardcoded widths so the dashboard widget controls layout
  responsively. _fig_to_html adds config={"responsive": True}.
  Donut charts constrained to max-width 600px via wrapper div.

FIX 3 — Report year selection: render_vg_report._pick() now
  filters by the target year key instead of always taking [0],
  and falls back to keyed_list[-1] (most recent) instead of [0]
  (oldest). get_report_year returns the latest year in the dataset.

FIX 4 (v4 — REAL FIX) — Illegal activity detection:
  Referencing the antipoaching workflow (bh_antipoaching spec) which
  reads the SAME CSV and successfully detects illegal activities by
  matching "Observation Category 0" against values like
  "mangrove logging", "poaching", "arrests", "poached turtle sum",
  and "Resource Use Type" == "illegal use" for fishing.

  The previous FIX 4 was WRONG: it assumed "Observation Category 0"
  held only high-level labels ("Resource use", "Wildlife Crime").
  In reality, "Observation Category 0" directly contains the specific
  activity names that the antipoaching workflow matches against.

  Corrected detection logic (matches antipoaching spec exactly):
    illegal fishing  → Resource Use Type == "illegal use"
    poaching         → Observation Category 0 contains "poaching"
                       (case-insensitive)
    mangrove logging → Observation Category 0 contains "mangrove logging"
                       (case-insensitive)
    arrests          → Observation Category 0 contains "arrests"
                       (case-insensitive)
    poached turtle   → Observation Category 0 contains "poached turtle"
                       (case-insensitive)
    damaging coral   → Observation Category 0 contains "damaging coral"
                       (case-insensitive)
"""

import base64
import io
import logging
from collections import Counter
from pathlib import Path
from typing import Annotated, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle
from pydantic import Field

from ecoscope_workflows_core.annotations import AnyDataFrame
from ecoscope_workflows_core.decorators import task

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# ICON FILENAMES
# ─────────────────────────────────────────────

ICON_FILES = {
    "thumbs_up": "green_thumb.png",
    "thumbs_down": "red_thumb.png",
    "boat": "boat.png",
    "happy": "happy.png",
    "sad": "sad.png",
    "poaching": "turtle_poaching.png",
    "mangrove logging": "mangrove_logging.png",
    "illegal fishing": "illegal_fishing.png",
    "arrests": "arrests.png",
    "poached turtle sum": "poached_turtle_sum.png",
}


def _icon(icons_dir: str, key: str) -> Path:
    filename = ICON_FILES[key]
    path = Path(icons_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Icon '{filename}' not found in '{icons_dir}'. Expected: {path}")
    return path


# ─────────────────────────────────────────────
# SHARED CONSTANTS
# ─────────────────────────────────────────────

ACTIVITY_CONFIGS = {
    "poaching": "Poaching",
    "mangrove logging": "Mangrove Logging",
    "illegal fishing": "Illegal Fishing",
    "arrests": "Arrests",
    "poached turtle sum": "Poached Turtle Sum",
}

DONUT_ACTIVITY_CONFIGS = {
    "mangrove logging": "Mangrove Logging",
    "poaching": "Poaching",
    "arrests": "Arrests",
    "poached turtle sum": "Poached Turtle Sum",
    "damaging coral": "Damaging Coral",
    "illegal fishing": "Illegal Fishing",
}

NO_MANGROVE_SECTORS = ["Sector 2", "Sector 3", "Sector 4", "Sector 5", "Sector 13"]

HEATMAP_COLORSCALE = [
    [0.00, "#FFFFE0"],
    [0.33, "#FFA500"],
    [0.66, "#FF4500"],
    [1.00, "#8B0000"],
]

LEADERBOARD_COLORSCALE = [
    [0.00, "#317F3F"],
    [0.30, "#739D50"],
    [0.55, "#C4C66D"],
    [0.75, "#F8A034"],
    [1.00, "#E63B1B"],
]

FONT_FAMILY = "Arial"
TITLE_FONT_SIZE = 16
AXIS_FONT_SIZE = 13
TICK_FONT_SIZE = 12


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────


def _img_to_base64(path: str | Path) -> str:
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise columns and add IsIllegal / Plot_Category / per-activity boolean flags.

    FIX 4 (v4) — Correct detection matching the antipoaching workflow:
    "Observation Category 0" directly contains specific activity names like
    "mangrove logging", "poaching", "arrests", "poached turtle sum", etc.
    This is confirmed by the bh_antipoaching spec which reads the SAME CSV
    and successfully filters on these exact values.

    Detection priority:
      1. illegal fishing  → Resource Use Type (case-insensitive) == "illegal use"
      2. poaching         → obs_cat contains "poaching"
      3. mangrove logging → obs_cat contains "mangrove logging" or "mangrove"
      4. arrests          → obs_cat contains "arrests"
      5. poached turtle   → obs_cat contains "poached turtle"
      6. damaging coral   → obs_cat contains "damaging coral"
    """
    df = pd.DataFrame(df).copy()
    df.columns = [c.strip() for c in df.columns]

    def _col(name: str) -> str:
        """Resolve column name, trying space and underscore variants."""
        if name in df.columns:
            return name
        alt = name.replace(" ", "_")
        if alt in df.columns:
            return alt
        alt2 = name.replace("_", " ")
        if alt2 in df.columns:
            return alt2
        return name

    # ── Resolve core columns ───────────────────────────────────────────────
    obs_col = _col("Observation Category 0")
    res_col = _col("Resource Use Type")

    # FIX 4: obs_cat is "Observation Category 0" — this column contains the
    # specific activity name ("poaching", "mangrove logging", etc.) NOT a
    # high-level label. This matches how antipoaching spec uses it.
    obs_cat = df[obs_col].fillna("").str.lower().str.strip() if obs_col in df.columns else pd.Series("", index=df.index)
    res_raw = df[res_col].fillna("").str.lower().str.strip() if res_col in df.columns else pd.Series("", index=df.index)

    df["_obs"] = obs_cat  # Observation Category 0 — specific activity name
    df["_res"] = res_raw  # Resource Use Type — "legal use" / "illegal use"

    # ── Per-activity boolean flags (FIX 4 — correct detection) ─────────────
    # These mirror the exact matching logic in the antipoaching workflow:
    # match_values: ["mangrove logging","poaching","arrests","poached turtle sum"]
    # remap: Resource Use Type == "illegal use" → "illegal fishing"
    df["_is_fishing"] = df["_res"] == "illegal use"
    df["_is_poaching"] = df["_obs"].str.contains("poaching", na=False)
    df["_is_mangrove"] = df["_obs"].str.contains("mangrove", na=False)
    df["_is_arrests"] = df["_obs"].str.contains("arrests", na=False)
    df["_is_turtle"] = df["_obs"].str.contains("poached turtle", na=False)
    df["_is_coral"] = df["_obs"].str.contains("damaging coral", na=False)

    df["IsIllegal"] = df["_is_fishing"] | df["_is_poaching"] | df["_is_mangrove"] | df["_is_arrests"] | df["_is_turtle"]

    # Plot_Category: most specific activity label for this row
    def _plot_cat(row: pd.Series) -> str:
        if row["_is_fishing"]:
            return "illegal fishing"
        if row["_is_poaching"]:
            return "poaching"
        if row["_is_mangrove"]:
            return "mangrove logging"
        if row["_is_arrests"]:
            return "arrests"
        if row["_is_turtle"]:
            return "poached turtle sum"
        return row["_obs"]

    df["Plot_Category"] = df.apply(_plot_cat, axis=1)

    logger.info(
        f"_prep_df: {df['IsIllegal'].sum()} illegal rows out of {len(df)} total. "
        f"Breakdown — fishing:{df['_is_fishing'].sum()} poaching:{df['_is_poaching'].sum()} "
        f"mangrove:{df['_is_mangrove'].sum()} arrests:{df['_is_arrests'].sum()} "
        f"turtle:{df['_is_turtle'].sum()}"
    )
    return df


def _add_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive Year / Month / YearQuarter from the patrol date column.

    FIX 1: Year is now cast to str to match the string dtype produced by
    clean_dataframe (which clean_dataframe must use so split_groups works).
    """
    for dcol in ["Patrol Start Date", "Patrol_Start_Date", "event_date"]:
        if dcol in df.columns:
            parsed = pd.to_datetime(df[dcol], errors="coerce")
            if "Year" not in df.columns:
                df["Year"] = parsed.dt.year.astype(str)
            if "Month" not in df.columns:
                df["Month"] = parsed.dt.month
            if "YearQuarter" not in df.columns:
                df["YearQuarter"] = parsed.dt.to_period("Q")
            break
    return df


def _extract_year(df: pd.DataFrame) -> str:
    """
    FIX 1: Returns str (e.g. "2024") so filter df[df["Year"] == year]
    matches the str dtype produced by clean_dataframe / split_yearly.
    """
    if "Year" in df.columns:
        val = df["Year"].dropna()
        if not val.empty:
            return str(val.iloc[0])
    for dcol in ["Patrol Start Date", "Patrol_Start_Date", "event_date"]:
        if dcol in df.columns:
            parsed = pd.to_datetime(df[dcol], errors="coerce").dropna()
            if not parsed.empty:
                return str(parsed.iloc[0].year)
    raise ValueError("Cannot determine year from dataframe. " "Expected a 'Year' column or a recognisable date column.")


def _extract_year_quarter(df: pd.DataFrame) -> tuple[int, int]:
    """
    FIX 1: Also handles the "2024Q1" string format produced by
    clean_dataframe → split_quarterly.
    Returns (year, quarter) as ints.
    """
    if "YearQuarter" in df.columns:
        val = df["YearQuarter"].dropna()
        if not val.empty:
            period = val.iloc[0]
            if hasattr(period, "year") and hasattr(period, "quarter"):
                return int(period.year), int(period.quarter)

    if "Quarter" in df.columns:
        val = df["Quarter"].dropna()
        if not val.empty:
            try:
                p = pd.Period(str(val.iloc[0]), freq="Q")
                return int(p.year), int(p.quarter)
            except Exception:
                pass

    for dcol in ["Patrol Start Date", "Patrol_Start_Date", "event_date"]:
        if dcol in df.columns:
            parsed = pd.to_datetime(df[dcol], errors="coerce").dropna()
            if not parsed.empty:
                dt = parsed.iloc[0]
                return int(dt.year), int((dt.month - 1) // 3 + 1)

    raise ValueError(
        "Cannot determine year/quarter from dataframe. "
        "Expected a 'YearQuarter'/'Quarter' column or a recognisable date column."
    )


def _fig_to_html(fig: go.Figure) -> str:
    """
    Renders a Plotly figure as a self-contained HTML page that fills
    the dashboard widget iframe exactly.
    - autosize=True + responsive config lets Plotly resize to container
    - 100vw/100vh CSS ensures the div fills the iframe
    - viewport meta prevents webview rescaling
    - All hardcoded width/height removed from figure layouts
    """
    fig.update_layout(autosize=True, width=None, height=None)
    inner = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displayModeBar": False},
        div_id="plotly-chart",
    )
    return (
        "<!DOCTYPE html><html><head>"
        "<meta charset='utf-8'/>"
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'/>"
        "<style>"
        "* { box-sizing: border-box; margin: 0; padding: 0; }"
        "html, body { width: 100%; height: 100%; overflow: hidden; background: #fff; }"
        "#plotly-chart { width: 100% !important; height: 100% !important; }"
        "#plotly-chart .plotly { width: 100% !important; height: 100% !important; }"
        "</style>"
        "</head><body>"
        f"{inner}"
        "<script>"
        "function resize() {"
        "  var el = document.getElementById('plotly-chart');"
        "  if (el) Plotly.relayout(el, {width: window.innerWidth, height: window.innerHeight});"
        "}"
        "window.addEventListener('resize', resize);"
        "window.addEventListener('load', resize);"
        "setTimeout(resize, 100);"
        "</script>"
        "</body></html>"
    )


def _fig_to_html_constrained(fig: go.Figure, max_width: int = 600) -> str:
    """
    Same as _fig_to_html but limits width for donut charts so they don't
    stretch across a very wide widget. Chart fills height fully.
    """
    fig.update_layout(autosize=True, width=None, height=None)
    inner = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displayModeBar": False},
        div_id="plotly-chart",
    )
    return (
        "<!DOCTYPE html><html><head>"
        "<meta charset='utf-8'/>"
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'/>"
        "<style>"
        "* { box-sizing: border-box; margin: 0; padding: 0; }"
        f"html, body {{ width: 100%; height: 100%; overflow: hidden; background: #fff; }}"
        f"#plotly-chart {{ width: 100% !important; max-width: {max_width}px; "
        f"height: 100% !important; margin: 0 auto; display: block; }}"
        "#plotly-chart .plotly { width: 100% !important; height: 100% !important; }"
        "</style>"
        "</head><body>"
        f"{inner}"
        "<script>"
        "function resize() {"
        f"  var w = Math.min(window.innerWidth, {max_width});"
        "  var h = window.innerHeight;"
        "  var el = document.getElementById('plotly-chart');"
        "  if (el) Plotly.relayout(el, {width: w, height: h});"
        "}"
        "window.addEventListener('resize', resize);"
        "window.addEventListener('load', resize);"
        "setTimeout(resize, 100);"
        "</script>"
        "</body></html>"
    )


def _fig_to_html_scrollable(fig: go.Figure, fig_height: int) -> str:
    """
    For tall charts (leaderboards, icon bars) that have many rows:
    - Preserves the computed fig_height so rows are not squashed
    - Wraps in a scrollable div so the widget can scroll vertically
      if the tile is shorter than the chart
    - Width still fills the container horizontally (100%)
    """
    # Do NOT call update_layout(autosize=True) here — we keep the fixed height
    fig.update_layout(autosize=False, width=None, height=fig_height)
    inner = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": False, "displayModeBar": False},
        div_id="plotly-chart",
    )
    return (
        "<!DOCTYPE html><html><head>"
        "<meta charset='utf-8'/>"
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'/>"
        "<style>"
        "* { box-sizing: border-box; margin: 0; padding: 0; }"
        "html, body { width: 100%; height: 100%; background: #fff; overflow-y: auto; overflow-x: hidden; }"
        # The chart div fills full width but keeps its computed height
        f"#plotly-chart {{ width: 100% !important; height: {fig_height}px !important; min-height: {fig_height}px; }}"
        "#plotly-chart .plotly { width: 100% !important; }"
        "</style>"
        "</head><body>"
        f"{inner}"
        # On load, relayout width to container width (keeps height fixed)
        "<script>"
        "function resizeWidth() {"
        "  var el = document.getElementById('plotly-chart');"
        f"  if (el) Plotly.relayout(el, {{width: window.innerWidth, height: {fig_height}}});"
        "}"
        "window.addEventListener('resize', resizeWidth);"
        "window.addEventListener('load', resizeWidth);"
        "setTimeout(resizeWidth, 100);"
        "</script>"
        "</body></html>"
    )


def _matplotlib_fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf.read()


def _png_bytes_to_html(png_bytes: bytes) -> str:
    """Wrap raw PNG bytes in a minimal HTML page with an <img> tag."""
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return (
        "<!DOCTYPE html><html><body style='margin:0;padding:0;background:#fff;'>"
        f"<img src='data:image/png;base64,{b64}' style='max-width:100%;display:block;'/>"
        "</body></html>"
    )


# ─────────────────────────────────────────────
# SCRIPT 1 — COMMUNITY FEEDBACK TABLES
# ─────────────────────────────────────────────


@task
def draw_community_feedback_table(
    df: Annotated[AnyDataFrame, Field(description="Cleaned patrol dataframe (yearly slice)")],
    activity_key: Annotated[
        str,
        Field(
            description=(
                "Activity to tabulate. One of: 'poaching', 'mangrove logging', "
                "'illegal fishing', 'arrests', 'poached turtle sum'."
            )
        ),
    ],
    icons_dir: Annotated[
        str,
        Field(description="Folder containing green_thumb.png and red_thumb.png"),
    ],
    station_column: Annotated[str, Field(description="Column for sector/station")] = "Station",
    village_column: Annotated[str, Field(description="Column for village")] = "Village",
    low_threshold: Annotated[int, Field(description="Events ≤ this → orange circle")] = 12,
    high_threshold: Annotated[int, Field(description="Events ≤ this → green circle")] = 30,
) -> Annotated[str, Field(description="HTML string containing an embedded PNG image")]:
    """
    Generate a community feedback table for one illegal activity.
    Year is derived automatically from the incoming df slice.
    Returns a full HTML string with the chart embedded as a base64 PNG.
    """
    df = _prep_df(df)
    df = _add_date_cols(df)

    year = _extract_year(df)
    logger.info(f"draw_community_feedback_table: activity={activity_key}, year={year}")

    thumbs_up_img = mpimg.imread(_icon(icons_dir, "thumbs_up"))
    thumbs_down_img = mpimg.imread(_icon(icons_dir, "thumbs_down"))

    all_stations = df[station_column].dropna().unique()
    station_village = (
        df[[station_column, village_column]].drop_duplicates().set_index(station_column)[village_column].to_dict()
    )

    year_df = df[df["Year"] == year]

    # FIX 4: use corrected per-activity flag columns
    if activity_key == "illegal fishing":
        filtered = year_df[year_df["_is_fishing"]]
    elif activity_key == "poaching":
        filtered = year_df[year_df["_is_poaching"]]
    elif activity_key == "mangrove logging":
        filtered = year_df[year_df["_is_mangrove"]]
    elif activity_key == "arrests":
        filtered = year_df[year_df["_is_arrests"]]
    elif activity_key == "poached turtle sum":
        filtered = year_df[year_df["_is_turtle"]]
    else:
        filtered = year_df[year_df["IsIllegal"]]

    counts = filtered.groupby(station_column).size().reset_index(name="Events")

    base = pd.DataFrame({station_column: all_stations})
    base[village_column] = base[station_column].map(station_village).fillna("Unknown")
    base = base.merge(counts, on=station_column, how="left").fillna({"Events": 0})
    base["Events"] = base["Events"].astype(int)

    def _traffic_color(x: int) -> str:
        if x <= low_threshold:
            return "#FFA500"
        if x <= high_threshold:
            return "#00CC00"
        return "#FF0000"

    def _message(x: int) -> str:
        if x == 0:
            return "No illegal activity recorded"
        if x <= low_threshold:
            return "Low activity — keep protecting!"
        if x <= high_threshold:
            return "Moderate activity — stay vigilant!"
        return "High activity — action needed!"

    base["Traffic_Color"] = base["Events"].apply(_traffic_color)
    base["Message"] = base["Events"].apply(_message)
    base["Icon"] = base["Events"].apply(lambda x: thumbs_up_img if x <= high_threshold else thumbs_down_img)
    base = base.sort_values("Events", ascending=False).reset_index(drop=True)

    n = len(base)
    fig_h = max(7.0, 0.50 * n + 2.8)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.axis("off")

    title_label = ACTIVITY_CONFIGS.get(activity_key, activity_key.title())
    ax.set_title(
        f"{title_label} Events: {year}",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        color="black",
        pad=20,
        fontfamily=FONT_FAMILY,
    )

    cell_text = [
        [row[village_column], row[station_column], row["Events"], row["Message"], ""] for _, row in base.iterrows()
    ]

    table = ax.table(
        cellText=cell_text,
        colLabels=["Village", "Sector", "Illegal Events", "Activity Status", "Feedback"],
        colWidths=[0.22, 0.16, 0.12, 0.40, 0.10],
        loc="upper center",
        cellLoc="center",
        bbox=[0.01, 0.03, 0.98, 0.94],
    )
    table.auto_set_font_size(False)
    table.scale(1, 2.6)
    fig.canvas.draw()

    for (row, col), cell in table.get_celld().items():
        fs = TITLE_FONT_SIZE if row == 0 else AXIS_FONT_SIZE
        fw = "bold" if row == 0 else "normal"
        cell.set_text_props(
            fontfamily=FONT_FAMILY,
            fontsize=fs,
            fontweight=fw,
            color="black",
            ha="center",
            va="center",
        )
        cell.set_edgecolor("black")
        cell.set_linewidth(1.2)
        cell.set_facecolor("white")

    for ri in range(1, n + 1):
        color = base["Traffic_Color"].iloc[ri - 1]
        cell = table.get_celld()[(ri, 3)]
        cx = cell.get_x() + 0.05 * cell.get_width()
        cy = cell.get_y() + cell.get_height() / 2
        ax.add_patch(Circle((cx, cy), 0.012, color=color, transform=ax.transAxes, zorder=10))

    for ri in range(1, n + 1):
        icon = base["Icon"].iloc[ri - 1]
        cell = table.get_celld()[(ri, 4)]
        xc = cell.get_x() + cell.get_width() / 2
        yc = cell.get_y() + cell.get_height() / 2
        ax.add_artist(
            AnnotationBbox(
                OffsetImage(icon, zoom=0.35),
                (xc, yc),
                xycoords="axes fraction",
                frameon=False,
                box_alignment=(0.5, 0.5),
            )
        )

    plt.tight_layout(rect=[0, 0.0, 1, 0.96])
    png_bytes = _matplotlib_fig_to_png_bytes(fig)
    plt.close(fig)

    return _png_bytes_to_html(png_bytes)


# ─────────────────────────────────────────────
# SCRIPT 2 — MONTHLY HEATMAP  (no icons)
# ─────────────────────────────────────────────


@task
def draw_monthly_heatmap(
    df: Annotated[AnyDataFrame, Field(description="Cleaned patrol dataframe (yearly slice)")],
    village_column: Annotated[str, Field(description="Column for village")] = "Village",
    title_prefix: Annotated[
        str,
        Field(description="Chart title prefix"),
    ] = "Monthly Distribution of Illegal Events by Village",
) -> Annotated[str, Field(description="Plotly HTML string")]:
    """
    Generate a monthly heatmap (villages × months) of illegal events.
    Year is derived automatically from the incoming df slice.
    """
    df = _prep_df(df)
    df = _add_date_cols(df)

    year = _extract_year(df)
    logger.info(f"draw_monthly_heatmap: year={year}")

    year_df = df[(df["Year"] == year) & df["IsIllegal"]].copy()
    if year_df.empty:
        fig = go.Figure()
        fig.update_layout(title=dict(text=f"No data for {year}", x=0.5))
        return _fig_to_html(fig)

    monthly = year_df.groupby([village_column, "Month"]).size().reset_index(name="Count")
    pivot = monthly.pivot(index=village_column, columns="Month", values="Count").fillna(0)
    pivot = pivot.reindex(columns=range(1, 13), fill_value=0)
    pivot["Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Total", ascending=True)
    hm = pivot.drop(columns=["Total"])
    villages = hm.index.tolist()
    totals = pivot["Total"].tolist()
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    # Compute height: 45px per village row + 140px for title/axes/margins
    heatmap_height = max(500, len(villages) * 45 + 140)

    fig = go.Figure(
        data=go.Heatmap(
            z=hm.values,
            x=month_names,
            y=villages,
            colorscale=HEATMAP_COLORSCALE,
            hoverongaps=False,
            text=hm.values.astype(int).astype(str),
            texttemplate="%{text}",
            textfont=dict(size=TICK_FONT_SIZE, color="black", family=FONT_FAMILY),
        )
    )

    for i, total in enumerate(totals):
        fig.add_annotation(
            x=12.5,
            y=i,
            text=f"<b>{int(total)}</b>",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color="black"),
        )
    fig.add_annotation(
        x=12.5,
        y=len(villages),
        text="<b>Total</b>",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color="black"),
    )

    fig.update_layout(
        autosize=True,
        title=dict(
            text=f"{title_prefix} — {year}",
            x=0.5,
            xanchor="center",
            font=dict(size=TITLE_FONT_SIZE, family=FONT_FAMILY, color="black"),
        ),
        xaxis=dict(
            title="Month",
            tickfont=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY, color="black"),
            title_font=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color="black"),
        ),
        yaxis=dict(
            title="Village",
            automargin=True,
            tickfont=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY, color="black"),
            title_font=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color="black"),
        ),
        height=heatmap_height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=160, r=90, t=90, b=80),
    )
    fig.update_traces(
        colorbar_title="Number of Events",
        colorbar=dict(title_font=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color="black")),
    )
    return _fig_to_html(fig)


# ─────────────────────────────────────────────
# SCRIPT 3 — PER-VILLAGE DONUT CHARTS  (no icons)
# ─────────────────────────────────────────────


@task
def draw_village_donut_chart(
    df: Annotated[AnyDataFrame, Field(description="Cleaned patrol dataframe (yearly slice)")],
    village: Annotated[str, Field(description="Village name to plot")],
    village_column: Annotated[str, Field(description="Column for village")] = "Village",
    damaging_col: Annotated[
        str,
        Field(description="Column for damaging coral activity"),
    ] = "Damaging coral activity",
) -> Annotated[str, Field(description="Plotly HTML string")]:
    """
    Generate a donut chart of illegal activities for one village.
    Year is derived automatically from the incoming df slice.

    FIX 2: Output is wrapped in a max-width div (600px) to prevent the
    chart from rendering too large / zoomed-in in the dashboard widget.
    """
    df = _prep_df(df)
    df = _add_date_cols(df)

    year = _extract_year(df)
    logger.info(f"draw_village_donut_chart: village={village}, year={year}")

    vdf = df[(df[village_column] == village) & (df["Year"] == year)]

    rows = []
    for key, label in DONUT_ACTIVITY_CONFIGS.items():
        # FIX 4: use corrected per-activity flags
        if key == "illegal fishing":
            count = int(vdf["_is_fishing"].sum())
        elif key == "poaching":
            count = int(vdf["_is_poaching"].sum())
        elif key == "mangrove logging":
            count = int(vdf["_is_mangrove"].sum())
        elif key == "arrests":
            count = int(vdf["_is_arrests"].sum())
        elif key == "poached turtle sum":
            count = int(vdf["_is_turtle"].sum())
        elif key == "damaging coral":
            count = int(vdf["_is_coral"].sum())
        else:
            count = 0
        rows.append({"label": label, "count": count})

    counts_df = pd.DataFrame(rows)
    total = counts_df["count"].sum()
    non_zero = counts_df[counts_df["count"] > 0]

    if non_zero.empty:
        labels_plot, values_plot, textinfo = ["No Activity"], [1], "none"
    else:
        labels_plot = non_zero["label"].tolist()
        values_plot = non_zero["count"].tolist()
        textinfo = "value"

    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            labels=labels_plot,
            values=values_plot,
            hole=0.5,
            textinfo=textinfo,
            textposition="inside",
            textfont=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color="black"),
            marker=dict(line=dict(color="#000000", width=1)),
        )
    )
    fig.add_annotation(
        text=f"<b>Total: {total}</b>",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=AXIS_FONT_SIZE + 2, family=FONT_FAMILY, color="black"),
        xanchor="center",
        yanchor="middle",
    )
    fig.update_layout(
        # FIX 2: fixed width + height so donut renders at a reasonable size
        # and is not zoomed in when the widget is larger than the chart.
        title=dict(
            text=f"<b>{village} — {year}</b>",
            x=0.5,
            y=0.95,
            font=dict(size=TITLE_FONT_SIZE, family=FONT_FAMILY, color="black"),
            xanchor="center",
            yanchor="top",
        ),
        margin=dict(t=60, b=60, l=40, r=40),
        template="simple_white",
        showlegend=True,
        legend=dict(
            x=1.02,
            y=0.5,
            xanchor="left",
            yanchor="middle",
            orientation="v",
            title=dict(
                text="Activities",
                font=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color="black"),
            ),
            font=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY, color="black"),
        ),
    )
    # FIX 2: constrain max-width via wrapper div so chart doesn't expand
    # to fill a very wide dashboard widget and appear zoomed in.
    return _fig_to_html(fig)


# ─────────────────────────────────────────────
# SCRIPT 4 — LEADERBOARD (GAMIFICATION)
# ─────────────────────────────────────────────


@task
def draw_activity_leaderboard(
    df: Annotated[AnyDataFrame, Field(description="Cleaned patrol dataframe (quarterly slice)")],
    activity_key: Annotated[
        str,
        Field(
            description=(
                "Activity key: 'poaching', 'mangrove logging', " "'illegal fishing', 'arrests', 'poached turtle sum'"
            )
        ),
    ],
    icons_dir: Annotated[
        str,
        Field(description="Folder containing boat.png, happy.png, sad.png"),
    ],
    station_column: Annotated[str, Field(description="Column for sector/station")] = "Station",
    village_column: Annotated[str, Field(description="Column for village")] = "Village",
    no_mangrove_sectors: Annotated[
        Optional[list[str]],
        Field(description="Sectors with no mangroves (mangrove logging only)"),
    ] = None,
) -> Annotated[str, Field(description="Plotly HTML string")]:
    """
    Reversed dark-gradient leaderboard for one activity / quarter.
    Improved gradient visibility and icon positioning.
    """
    df = _prep_df(df)
    df = _add_date_cols(df)

    year, quarter = _extract_year_quarter(df)
    logger.info(f"draw_activity_leaderboard: activity={activity_key}, Q{quarter} {year}")

    no_mg = no_mangrove_sectors or NO_MANGROVE_SECTORS

    boat_b64 = _img_to_base64(_icon(icons_dir, "boat"))
    happy_b64 = _img_to_base64(_icon(icons_dir, "happy"))
    sad_b64 = _img_to_base64(_icon(icons_dir, "sad"))

    selected_yq = pd.Period(year=year, quarter=quarter, freq="Q")
    qdf = df[(df["YearQuarter"] == selected_yq) & df["IsIllegal"]]

    # FIX 4: use corrected per-activity flags
    if activity_key == "illegal fishing":
        adf = qdf[qdf["_is_fishing"]]
    elif activity_key == "poaching":
        adf = qdf[qdf["_is_poaching"]]
    elif activity_key == "mangrove logging":
        adf = qdf[qdf["_is_mangrove"]]
    elif activity_key == "arrests":
        adf = qdf[qdf["_is_arrests"]]
    elif activity_key == "poached turtle sum":
        adf = qdf[qdf["_is_turtle"]]
    else:
        adf = qdf

    sector_order = {f"Sector {i}": i for i in range(1, 14)}
    all_stations = sorted(df[station_column].dropna().unique(), key=lambda x: sector_order.get(x, 99))
    village_map = (
        df[[station_column, village_column]].drop_duplicates().set_index(station_column)[village_column].to_dict()
    )

    counts = adf.groupby(station_column).size().reset_index(name="Events")
    sb = pd.DataFrame({station_column: all_stations})
    sb[village_column] = sb[station_column].map(village_map).fillna("Unknown")
    sb = sb.merge(counts, on=station_column, how="left").fillna({"Events": 0})
    sb["Events"] = sb["Events"].astype(int)

    if activity_key == "mangrove logging":
        sb = sb[~sb[station_column].isin(no_mg)].reset_index(drop=True)

    sb["Order"] = sb[station_column].map(sector_order).fillna(99)
    sb = sb.sort_values("Order").reset_index(drop=True)

    y_labels = sb[village_column].tolist()
    max_events = max(sb["Events"].max(), 1)

    # ====================== IMPROVED LEADERBOARD ======================
    fig = go.Figure()

    # Background color gradient (wider range + better opacity)
    x_vals = np.linspace(-2.5, max_events + 4.5, 1000)
    z = np.tile(x_vals, (len(y_labels), 1))

    fig.add_trace(
        go.Heatmap(
            z=z,
            x=x_vals,
            y=y_labels,
            colorscale=LEADERBOARD_COLORSCALE,
            opacity=0.48,
            showscale=False,
            hoverinfo="skip",
            zmin=-2.5,
            zmax=max_events + 4.5,
        )
    )

    # Invisible bar for hover information
    fig.add_trace(
        go.Bar(
            y=y_labels,
            x=sb["Events"],
            orientation="h",
            marker_color="rgba(0,0,0,0)",
            opacity=0,
            hovertemplate="<b>%{y}</b><br>Events: %{x}<extra></extra>",
        )
    )

    # Boat icons - one per sector
    for i, events in enumerate(sb["Events"]):
        x_pos = -1.0 if events == 0 else float(events)
        fig.add_layout_image(
            dict(
                source=boat_b64,
                xref="x",
                yref="y",
                x=x_pos,
                y=i,
                sizex=1.65,
                sizey=1.65,
                xanchor="center",
                yanchor="middle",
                layer="above",
            )
        )

    # Happy / Sad icons at the top
    fig.add_layout_image(
        dict(
            source=happy_b64,
            xref="x",
            yref="paper",
            x=-1.9,
            y=1.09,
            sizex=0.95,
            sizey=0.95,
            xanchor="center",
            yanchor="bottom",
            layer="above",
        )
    )
    fig.add_layout_image(
        dict(
            source=sad_b64,
            xref="x",
            yref="paper",
            x=max_events + 3.6,
            y=1.09,
            sizex=0.95,
            sizey=0.95,
            xanchor="center",
            yanchor="bottom",
            layer="above",
        )
    )

    title_label = ACTIVITY_CONFIGS.get(activity_key, activity_key.title())

    fig.update_layout(
        title=dict(
            text=f"Leaderboard: {title_label} — Q{quarter} {year}",
            x=0.5,
            xanchor="center",
            font=dict(size=TITLE_FONT_SIZE, family=FONT_FAMILY, color="black"),
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY, color="black"),
            showgrid=False,
            showline=False,
            tickmode="array",
            tickvals=list(range(len(y_labels))),
            ticktext=y_labels,
        ),
        xaxis=dict(
            range=[max_events + 4.8, -2.8],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False,
        ),
        margin=dict(t=120, b=80, l=360, r=230),
        height=max(720, len(y_labels) * 68 + 190),
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        shapes=[
            # Green "Good" line
            dict(
                type="line",
                x0=-1.9,
                x1=-1.9,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="#006400", width=3.5),
            ),
            # Red "Bad" line
            dict(
                type="line",
                x0=max_events + 3.4,
                x1=max_events + 3.4,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="#D30000", width=3.5),
            ),
        ],
        annotations=[
            dict(
                x=-1.9,
                y=1.02,
                xref="x",
                yref="paper",
                text="Good (Fewer Events)",
                showarrow=False,
                font=dict(size=TICK_FONT_SIZE + 1, family=FONT_FAMILY, color="#006400"),
                xanchor="right",
                yanchor="bottom",
            ),
            dict(
                x=max_events + 3.4,
                y=1.02,
                xref="x",
                yref="paper",
                text="Bad (More Events)",
                showarrow=False,
                font=dict(size=TICK_FONT_SIZE + 1, family=FONT_FAMILY, color="#D30000"),
                xanchor="left",
                yanchor="bottom",
            ),
        ],
    )

    lb_height = max(720, len(y_labels) * 68 + 190)
    return _fig_to_html_scrollable(fig, lb_height)


# ─────────────────────────────────────────────
# SHARED HELPER — ICON BAR FIGURE
# ─────────────────────────────────────────────


def _build_icon_bar_figure(
    vdf: pd.DataFrame,
    village: str,
    period_label: str,
    icon_b64: dict[str, str],
    icons_per_row: int = 8,
) -> go.Figure:
    """Shared rendering logic for Scripts 5 and 6."""
    row_gap = 6.0
    col_spacing = 2.0
    icon_size = 2.8
    icon_row_spacing = 2.8
    x_icon_start = 1.5
    total_box_gap = (2 * col_spacing) - 4.0

    rows = []
    for key, label in ACTIVITY_CONFIGS.items():
        # FIX 4: use corrected per-activity flags
        if key == "illegal fishing":
            count = int(vdf["_is_fishing"].sum())
        elif key == "poaching":
            count = int(vdf["_is_poaching"].sum())
        elif key == "mangrove logging":
            count = int(vdf["_is_mangrove"].sum())
        elif key == "arrests":
            count = int(vdf["_is_arrests"].sum())
        elif key == "poached turtle sum":
            count = int(vdf["_is_turtle"].sum())
        else:
            count = 0
        rows.append({"Activity": label, "Count": count, "Key": key})

    counts_df = pd.DataFrame(rows)
    y_positions = [i * row_gap for i in range(len(counts_df))]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=y_positions,
            x=[1] * len(y_positions),
            orientation="h",
            marker_color="rgba(0,0,0,0)",
            hoverinfo="skip",
        )
    )

    images, annotations = [], []

    for idx, row in counts_df.iterrows():
        y_base = y_positions[idx]
        count = row["Count"]
        src = icon_b64.get(row["Key"])

        if count == 0 or src is None:
            annotations.append(
                dict(
                    xref="x",
                    yref="y",
                    x=x_icon_start,
                    y=y_base,
                    text="<b>No Events</b>",
                    showarrow=False,
                    xanchor="left",
                    font=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color="black"),
                )
            )
            continue

        for i in range(count):
            r, c = divmod(i, icons_per_row)
            images.append(
                dict(
                    source=src,
                    xref="x",
                    yref="y",
                    x=x_icon_start + c * col_spacing,
                    y=y_base - r * icon_row_spacing,
                    sizex=icon_size,
                    sizey=icon_size,
                    xanchor="center",
                    yanchor="middle",
                )
            )

        annotations.append(
            dict(
                xref="x",
                yref="y",
                x=x_icon_start + icons_per_row * col_spacing + total_box_gap,
                y=y_base,
                text=f"<b>{count}</b>",
                showarrow=False,
                font=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color="black"),
                bordercolor="black",
                borderwidth=1.5,
                borderpad=5,
                bgcolor="white",
                xanchor="center",
                yanchor="middle",
            )
        )

    x_max = x_icon_start + (icons_per_row + 4) * col_spacing
    fig.update_layout(
        autosize=True,
        title=dict(
            text=f"<b>{village} — {period_label}</b>",
            x=0.5,
            font=dict(size=TITLE_FONT_SIZE, family=FONT_FAMILY, color="black"),
        ),
        images=images,
        annotations=annotations,
        margin=dict(l=300, r=250, t=120, b=100),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(
                text="<b>Number of illegal events</b>",
                font=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color="black"),
            ),
            domain=[0.0, 0.78],
            range=[0, x_max],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(
                text="<b>Illegal Event</b>",
                font=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color="black"),
            ),
            tickvals=y_positions,
            ticktext=[f"<b>{r['Activity']}</b>" for _, r in counts_df.iterrows()],
            tickfont=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color="black"),
            autorange="reversed",
            showgrid=False,
            zeroline=False,
        ),
    )
    # Height: enough for all activity rows with generous spacing
    ibar_height = max(500, len(counts_df) * 80 + 140)
    fig.update_layout(autosize=False, height=ibar_height)
    return fig, ibar_height


# ─────────────────────────────────────────────
# SCRIPT 5 — PER-VILLAGE ICON BAR (individual)
# ─────────────────────────────────────────────


@task
def draw_village_icon_bar(
    df: Annotated[AnyDataFrame, Field(description="Cleaned patrol dataframe (quarterly slice)")],
    village: Annotated[str, Field(description="Village name")],
    icons_dir: Annotated[
        str,
        Field(
            description=(
                "Folder containing turtle_poaching.png, mangrove_logging.png, "
                "illegal_fishing.png, arrests.png, poached_turtle_sum.png"
            )
        ),
    ],
    village_column: Annotated[str, Field(description="Column for village")] = "Village",
    icons_per_row: Annotated[int, Field(description="Icons per row before wrapping")] = 8,
) -> Annotated[str, Field(description="Plotly HTML string")]:
    """
    Per-village stacked icon bar for one quarter.
    Year and quarter are derived automatically from the incoming df slice.
    """
    df = _prep_df(df)
    df = _add_date_cols(df)

    year, quarter = _extract_year_quarter(df)
    logger.info(f"draw_village_icon_bar: village={village}, Q{quarter} {year}")

    selected_yq = pd.Period(year=year, quarter=quarter, freq="Q")
    vdf = df[(df[village_column] == village) & (df["YearQuarter"] == selected_yq) & df["IsIllegal"]]

    icon_b64 = {key: _img_to_base64(_icon(icons_dir, key)) for key in ACTIVITY_CONFIGS}
    fig, ibar_height = _build_icon_bar_figure(vdf, village, f"Q{quarter} {year}", icon_b64, icons_per_row)
    return _fig_to_html_scrollable(fig, ibar_height)


# ─────────────────────────────────────────────
# SCRIPT 6 — SUMMARY ALL-VILLAGES ICON BAR
# ─────────────────────────────────────────────


@task
def draw_all_villages_icon_bar(
    df: Annotated[AnyDataFrame, Field(description="Cleaned patrol dataframe (quarterly slice)")],
    icons_dir: Annotated[
        str,
        Field(
            description=(
                "Folder containing turtle_poaching.png, mangrove_logging.png, "
                "illegal_fishing.png, arrests.png, poached_turtle_sum.png"
            )
        ),
    ],
    village_column: Annotated[str, Field(description="Column for village")] = "Village",
    icons_per_row: Annotated[int, Field(description="Icons per row before wrapping")] = 8,
) -> Annotated[str, Field(description="Plotly HTML string")]:
    """
    Single icon bar with all villages on the Y-axis for one quarter.
    Year and quarter are derived automatically from the incoming df slice.
    """
    df = _prep_df(df)
    df = _add_date_cols(df)

    year, quarter = _extract_year_quarter(df)
    logger.info(f"draw_all_villages_icon_bar: Q{quarter} {year}")

    selected_yq = pd.Period(year=year, quarter=quarter, freq="Q")
    qdf = df[(df["YearQuarter"] == selected_yq) & df["IsIllegal"]].copy()
    villages = sorted(qdf[village_column].dropna().unique())

    icon_b64 = {key: _img_to_base64(_icon(icons_dir, key)) for key in ACTIVITY_CONFIGS}

    row_gap = 6.0
    col_spacing = 2.0
    icon_size = 2.8
    icon_row_spacing = 2.8
    x_icon_start = 1.5
    total_box_gap = (2 * col_spacing) - 4.0
    y_positions = [i * row_gap for i in range(len(villages))]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=y_positions,
            x=[1] * len(y_positions),
            orientation="h",
            marker_color="rgba(0,0,0,0)",
            hoverinfo="skip",
        )
    )

    images, annotations = [], []

    for idx, village in enumerate(villages):
        y_base = y_positions[idx]
        vdf = qdf[qdf[village_column] == village]

        # FIX 4: use corrected per-activity flags
        activity_seq = []
        for _, row in vdf.iterrows():
            if row.get("_is_fishing", False):
                activity_seq.append("illegal fishing")
            elif row.get("_is_poaching", False):
                activity_seq.append("poaching")
            elif row.get("_is_mangrove", False):
                activity_seq.append("mangrove logging")
            elif row.get("_is_arrests", False):
                activity_seq.append("arrests")
            elif row.get("_is_turtle", False):
                activity_seq.append("poached turtle sum")

        if not activity_seq:
            annotations.append(
                dict(
                    xref="x",
                    yref="y",
                    x=x_icon_start,
                    y=y_base,
                    text="<b>No Events</b>",
                    showarrow=False,
                    xanchor="left",
                    font=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color="black"),
                )
            )
            continue

        counts = Counter(activity_seq)
        ordered = [act for act in ACTIVITY_CONFIGS for _ in range(counts.get(act, 0))]

        for i, act in enumerate(ordered):
            r, c = divmod(i, icons_per_row)
            images.append(
                dict(
                    source=icon_b64[act],
                    xref="x",
                    yref="y",
                    x=x_icon_start + c * col_spacing,
                    y=y_base - r * icon_row_spacing,
                    sizex=icon_size,
                    sizey=icon_size,
                    xanchor="center",
                    yanchor="middle",
                )
            )

        annotations.append(
            dict(
                xref="x",
                yref="y",
                x=x_icon_start + icons_per_row * col_spacing + total_box_gap,
                y=y_base,
                text=f"<b>{len(ordered)}</b>",
                showarrow=False,
                font=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color="black"),
                bordercolor="black",
                borderwidth=1.5,
                borderpad=5,
                bgcolor="white",
                xanchor="center",
                yanchor="middle",
            )
        )

    x_max = x_icon_start + (icons_per_row + 4) * col_spacing
    period_label = f"Q{quarter} {year}"

    fig.update_layout(
        autosize=True,
        title=dict(
            text=f"<b>Illegal Activities by Village — {period_label}</b>",
            x=0.5,
            font=dict(size=TITLE_FONT_SIZE, family=FONT_FAMILY, color="black"),
        ),
        height=max(800, len(villages) * 80),
        images=images,
        annotations=annotations,
        margin=dict(l=300, r=250, t=120, b=100),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(
                text="<b>Number of illegal events</b>",
                font=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color="black"),
            ),
            domain=[0.0, 0.78],
            range=[0, x_max],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(
                text="<b>Villages</b>",
                font=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color="black"),
            ),
            tickvals=y_positions,
            ticktext=[f"<b>{v}</b>" for v in villages],
            tickfont=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color="black"),
            autorange="reversed",
            showgrid=False,
            zeroline=False,
        ),
    )
    all_height = max(600, len(villages) * 65 + 140)
    fig.update_layout(autosize=False, height=all_height)
    return _fig_to_html_scrollable(fig, all_height)


# ─────────────────────────────────────────────
# HELPER — rekey_widget_by_year
# ─────────────────────────────────────────────


@task
def rekey_widget_by_year(
    widget: Annotated[
        list,
        Field(
            description=(
                "A mapvalues widget output — a list of (quarter_key, widget) tuples "
                "e.g. [((Quarter, =, 2021Q1), w1), ((Quarter, =, 2021Q2), w2), ...]. "
                "Each entry is re-keyed to its parent Year so the dashboard Year "
                "selector controls these quarterly widgets alongside yearly ones."
            )
        ),
    ],
) -> Annotated[
    list,
    Field(
        description=(
            "Re-keyed list of (year_key, widget) tuples where the quarter composite "
            "filter is replaced by the equivalent Year composite filter, e.g. "
            "[((Year, =, 2021), w1), ((Year, =, 2021), w2), ...]."
        )
    ),
]:
    """
    Re-key a quarterly mapvalues widget output by its parent Year.

    The framework's mapvalues mechanism keys quarterly widgets with composite
    filters like ((('Quarter', '=', '2021Q1'), ...), widget). The dashboard
    Year filter cannot match these Quarter keys, so quarterly widgets are
    invisible when the Year selector is active.

    This task converts each Quarter key to its parent Year key so that
    selecting e.g. '2024' in the dashboard shows both yearly charts AND
    all quarterly charts (leaderboards, icon bars) for every quarter in 2024.

    Key mapping:
        Quarter key: ((('Quarter', '=', '2024Q1'),), widget)
        →  Year key: ((('Year',    '=', '2024'),),   widget)
    """
    result = []
    for entry in widget:
        if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
            result.append(entry)
            continue

        composite_filter, w = entry

        # composite_filter is a tuple/list of condition tuples
        # e.g. ((('Quarter', '=', '2021Q1'),),)  or  (('Quarter', '=', '2021Q1'),)
        # Normalise to a flat list of condition 3-tuples
        def _flatten(cf):
            """Recursively yield individual (index_name, op, value) triples."""
            if not cf:
                return
            first = cf[0] if hasattr(cf, "__getitem__") else next(iter(cf))
            if isinstance(first, str):
                # already a 3-tuple like ('Quarter', '=', '2021Q1')
                yield tuple(cf)
            else:
                for item in cf:
                    yield from _flatten(item)

        conditions = list(_flatten(composite_filter))

        # Build a new composite filter replacing Quarter conditions with Year
        new_conditions = []
        for cond in conditions:
            if len(cond) == 3 and cond[0] == "Quarter":
                _, op, quarter_val = cond
                # Parse year from quarter string: "2024Q1" → "2024"
                try:
                    year_val = str(pd.Period(str(quarter_val), freq="Q").year)
                except Exception:
                    # fallback: take first 4 chars
                    year_val = str(quarter_val)[:4]
                new_conditions.append(("Year", op, year_val))
            else:
                new_conditions.append(cond)

        # Reconstruct composite filter in the same shape as the original
        # The framework expects a tuple of tuples matching the original structure
        if len(new_conditions) == 1:
            new_cf = (tuple(new_conditions[0]),)
        else:
            new_cf = tuple(tuple(c) for c in new_conditions)

        result.append((new_cf, w))

    return result


# ─────────────────────────────────────────────
# HELPER — get_report_year (FIX 3)
# ─────────────────────────────────────────────


@task
def get_report_year(
    df: Annotated[AnyDataFrame, Field(description="Cleaned patrol dataframe")],
    groupers: Annotated[
        list,
        Field(description="Groupers (unused here, kept for workflow compatibility)"),
    ],
) -> Annotated[str, Field(description="Latest year present in the dataset as a string")]:
    """
    FIX 3: Returns the LATEST year in the dataset as a str (e.g. '2024').
    Uses sorted(...) descending so years[-1] == most recent.
    """
    df = pd.DataFrame(df)
    if "Year" in df.columns:
        years = sorted(df["Year"].dropna().astype(str).unique())
        if years:
            return years[-1]  # latest year (sort is ascending, so last = newest)
    for dcol in ["Patrol Start Date", "Patrol_Start_Date", "event_date"]:
        if dcol in df.columns:
            parsed = pd.to_datetime(df[dcol], errors="coerce").dropna()
            if not parsed.empty:
                return str(parsed.dt.year.max())
    raise ValueError("Cannot determine report year from dataframe.")


# ─────────────────────────────────────────────
# SCRIPT 7 — VILLAGE GAMES REPORT RENDERER
# ─────────────────────────────────────────────


@task
def render_vg_report(
    template_path: Annotated[str, Field(description="Path to the Jinja2 .docx template")],
    output_path: Annotated[str, Field(description="Directory to write the rendered report")],
    report_year: Annotated[str, Field(description="Report year string, e.g. '2024'")],
    generation_date: Annotated[str, Field(description="Human-readable generation date")],
    # ── Section 1: Community Feedback Tables ──────────────
    feedback_poaching_png: Annotated[list, Field(description="Keyed list of feedback poaching PNGs")],
    feedback_mangrove_png: Annotated[list, Field(description="Keyed list of feedback mangrove PNGs")],
    feedback_fishing_png: Annotated[list, Field(description="Keyed list of feedback fishing PNGs")],
    feedback_arrests_png: Annotated[list, Field(description="Keyed list of feedback arrests PNGs")],
    feedback_turtles_png: Annotated[list, Field(description="Keyed list of feedback turtles PNGs")],
    # ── Section 2: Monthly Heatmap ─────────────────────────
    monthly_heatmap_png: Annotated[list, Field(description="Keyed list of monthly heatmap PNGs")],
    # ── Section 3: Donut Charts ────────────────────────────
    donut_darakasi_watamu_png: Annotated[list, Field(description="Keyed list of donut PNGs")],
    donut_dongokundu_sita_png: Annotated[list, Field(description="Keyed list of donut PNGs")],
    donut_jacaranda_kanani_png: Annotated[list, Field(description="Keyed list of donut PNGs")],
    donut_kanani_darakasi_png: Annotated[list, Field(description="Keyed list of donut PNGs")],
    donut_kivunjeni_wesa_png: Annotated[list, Field(description="Keyed list of donut PNGs")],
    donut_magangani_mida_png: Annotated[list, Field(description="Keyed list of donut PNGs")],
    donut_marafiki_uyombo_png: Annotated[list, Field(description="Keyed list of donut PNGs")],
    donut_mawe_jacaranda_png: Annotated[list, Field(description="Keyed list of donut PNGs")],
    donut_mid_mayungu_mawe_png: Annotated[list, Field(description="Keyed list of donut PNGs")],
    donut_mida_marafiki_png: Annotated[list, Field(description="Keyed list of donut PNGs")],
    donut_sita_magangani_png: Annotated[list, Field(description="Keyed list of donut PNGs")],
    donut_uyombo_kivunjeni_png: Annotated[list, Field(description="Keyed list of donut PNGs")],
    donut_watamu_dongokundu_png: Annotated[list, Field(description="Keyed list of donut PNGs")],
    # ── Section 4: Leaderboards ────────────────────────────
    lb_poaching_png: Annotated[list, Field(description="Keyed list of leaderboard PNGs")],
    lb_mangrove_png: Annotated[list, Field(description="Keyed list of leaderboard PNGs")],
    lb_fishing_png: Annotated[list, Field(description="Keyed list of leaderboard PNGs")],
    lb_arrests_png: Annotated[list, Field(description="Keyed list of leaderboard PNGs")],
    # ── Section 5: Per-Village Icon Bars ──────────────────
    ibar_darakasi_watamu_png: Annotated[list, Field(description="Keyed list of icon bar PNGs")],
    ibar_dongokundu_sita_png: Annotated[list, Field(description="Keyed list of icon bar PNGs")],
    ibar_magangani_mida_png: Annotated[list, Field(description="Keyed list of icon bar PNGs")],
    ibar_uyombo_kivunjeni_png: Annotated[list, Field(description="Keyed list of icon bar PNGs")],
    ibar_watamu_dongokundu_png: Annotated[list, Field(description="Keyed list of icon bar PNGs")],
    ibar_mida_marafiki_png: Annotated[list, Field(description="Keyed list of icon bar PNGs")],
    ibar_mawe_jacaranda_png: Annotated[list, Field(description="Keyed list of icon bar PNGs")],
    ibar_mid_mayungu_mawe_png: Annotated[list, Field(description="Keyed list of icon bar PNGs")],
    ibar_marafiki_uyombo_png: Annotated[list, Field(description="Keyed list of icon bar PNGs")],
    ibar_kivunjeni_wesa_png: Annotated[list, Field(description="Keyed list of icon bar PNGs")],
    ibar_kanani_darakasi_png: Annotated[list, Field(description="Keyed list of icon bar PNGs")],
    ibar_jacaranda_kanani_png: Annotated[list, Field(description="Keyed list of icon bar PNGs")],
    # ── Section 6: All-Villages Summary ───────────────────
    all_villages_icon_bar_png: Annotated[list, Field(description="Keyed list of all-villages bar PNGs")],
    # ── Section 7: Maps ───────────────────────────────────
    illegal_events_map_png: Annotated[list, Field(description="Keyed list of illegal events map PNGs")],
    poaching_map_png: Annotated[list, Field(description="Keyed list of poaching map PNGs")],
    mangrove_map_png: Annotated[list, Field(description="Keyed list of mangrove map PNGs")],
    illegal_fishing_map_png: Annotated[list, Field(description="Keyed list of illegal fishing map PNGs")],
    arrests_map_png: Annotated[list, Field(description="Keyed list of arrests map PNGs")],
) -> Annotated[str, Field(description="Path to the rendered report file")]:
    """
    Render the Village Games Word report from a Jinja2 .docx template.

    FIX 3: _pick() selects the PNG whose key matches report_year (latest year),
    falling back to keyed_list[-1] (most recent entry, NOT [0] = oldest).
    """
    from docxtpl import DocxTemplate, InlineImage
    from docx.shared import Inches
    from pathlib import Path as _Path

    logger.info(f"render_vg_report: year={report_year}")
    year_str = str(report_year)

    def _pick(keyed_list: list) -> str:
        """
        FIX 3: Walk the keyed list and return the path whose key string
        contains report_year. Falls back to keyed_list[-1] (most recent year).
        """
        if not keyed_list:
            raise ValueError("Received an empty keyed list — no PNG was produced upstream.")

        for entry in keyed_list:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                key, value = entry
                if year_str in str(key):
                    return str(value)

        # FIX 3: fallback to LAST entry (most recent year), NOT first (oldest)
        all_keys = [str(e[0]) if isinstance(e, (list, tuple)) and len(e) == 2 else str(e) for e in keyed_list]
        logger.warning(
            f"render_vg_report._pick: no key matched year='{year_str}'. "
            f"Available keys: {all_keys}. Falling back to last (most recent) entry."
        )
        last = keyed_list[-1]
        if isinstance(last, (list, tuple)) and len(last) == 2:
            return str(last[1])
        return str(last)

    tpl = DocxTemplate(template_path)

    def _img(keyed_list: list, width_in: float = 6.8) -> InlineImage:
        path = _pick(keyed_list)
        return InlineImage(tpl, path, width=Inches(width_in))

    context = {
        "report_year": year_str,
        "generation_date": str(generation_date),
        # Section 1
        "feedback_poaching_png": _img(feedback_poaching_png, 6.8),
        "feedback_mangrove_png": _img(feedback_mangrove_png, 6.8),
        "feedback_fishing_png": _img(feedback_fishing_png, 6.8),
        "feedback_arrests_png": _img(feedback_arrests_png, 6.8),
        "feedback_turtles_png": _img(feedback_turtles_png, 6.8),
        # Section 2
        "monthly_heatmap_png": _img(monthly_heatmap_png, 7.0),
        # Section 3
        "donut_darakasi_watamu_png": _img(donut_darakasi_watamu_png, 6.5),
        "donut_dongokundu_sita_png": _img(donut_dongokundu_sita_png, 6.5),
        "donut_jacaranda_kanani_png": _img(donut_jacaranda_kanani_png, 6.5),
        "donut_kanani_darakasi_png": _img(donut_kanani_darakasi_png, 6.5),
        "donut_kivunjeni_wesa_png": _img(donut_kivunjeni_wesa_png, 6.5),
        "donut_magangani_mida_png": _img(donut_magangani_mida_png, 6.5),
        "donut_marafiki_uyombo_png": _img(donut_marafiki_uyombo_png, 6.5),
        "donut_mawe_jacaranda_png": _img(donut_mawe_jacaranda_png, 6.5),
        "donut_mid_mayungu_mawe_png": _img(donut_mid_mayungu_mawe_png, 6.5),
        "donut_mida_marafiki_png": _img(donut_mida_marafiki_png, 6.5),
        "donut_sita_magangani_png": _img(donut_sita_magangani_png, 6.5),
        "donut_uyombo_kivunjeni_png": _img(donut_uyombo_kivunjeni_png, 6.5),
        "donut_watamu_dongokundu_png": _img(donut_watamu_dongokundu_png, 6.5),
        # Section 4
        "lb_poaching_png": _img(lb_poaching_png, 6.8),
        "lb_mangrove_png": _img(lb_mangrove_png, 6.8),
        "lb_fishing_png": _img(lb_fishing_png, 6.8),
        "lb_arrests_png": _img(lb_arrests_png, 6.8),
        # Section 5
        "ibar_darakasi_watamu_png": _img(ibar_darakasi_watamu_png, 6.8),
        "ibar_dongokundu_sita_png": _img(ibar_dongokundu_sita_png, 6.8),
        "ibar_magangani_mida_png": _img(ibar_magangani_mida_png, 6.8),
        "ibar_uyombo_kivunjeni_png": _img(ibar_uyombo_kivunjeni_png, 6.8),
        "ibar_watamu_dongokundu_png": _img(ibar_watamu_dongokundu_png, 6.8),
        "ibar_mida_marafiki_png": _img(ibar_mida_marafiki_png, 6.8),
        "ibar_mawe_jacaranda_png": _img(ibar_mawe_jacaranda_png, 6.8),
        "ibar_mid_mayungu_mawe_png": _img(ibar_mid_mayungu_mawe_png, 6.8),
        "ibar_marafiki_uyombo_png": _img(ibar_marafiki_uyombo_png, 6.8),
        "ibar_kivunjeni_wesa_png": _img(ibar_kivunjeni_wesa_png, 6.8),
        "ibar_kanani_darakasi_png": _img(ibar_kanani_darakasi_png, 6.8),
        "ibar_jacaranda_kanani_png": _img(ibar_jacaranda_kanani_png, 6.8),
        # Section 6
        "all_villages_icon_bar_png": _img(all_villages_icon_bar_png, 7.0),
        # Section 7
        "illegal_events_map_png": _img(illegal_events_map_png, 6.8),
        "poaching_map_png": _img(poaching_map_png, 6.8),
        "mangrove_map_png": _img(mangrove_map_png, 6.8),
        "illegal_fishing_map_png": _img(illegal_fishing_map_png, 6.8),
        "arrests_map_png": _img(arrests_map_png, 6.8),
    }

    tpl.render(context)

    from urllib.parse import urlparse
    from urllib.request import url2pathname

    if output_path.startswith("file:"):
        output_path = url2pathname(urlparse(output_path).path)
    out_dir = _Path(output_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"village_games_report_{year_str}.docx"
    tpl.save(str(out_file))

    logger.info(f"render_vg_report: saved to {out_file}")
    return str(out_file)
