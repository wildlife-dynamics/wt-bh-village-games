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
# Update these if your icon files have different names.
# ─────────────────────────────────────────────

ICON_FILES = {
    # Feedback tables
    "thumbs_up": "green_thumb.png",
    "thumbs_down": "red_thumb.png",
    # Leaderboards
    "boat": "boat.png",
    "happy": "happy.png",
    "sad": "sad.png",
    # Icon bars (keyed by activity_key)
    "poaching": "turtle_poaching.png",
    "mangrove logging": "mangrove_logging.png",
    "illegal fishing": "illegal_fishing.png",
    "arrests": "arrests.png",
    "poached turtle sum": "poached_turtle_sum.png",
}


def _icon(icons_dir: str, key: str) -> Path:
    """Resolve a full icon path from the icons directory and a key from ICON_FILES."""
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


def _is_illegal_event(obs_cat: str, resource_use: str) -> bool:
    obs = obs_cat.lower().strip()
    res = resource_use.lower().strip()
    if any(k in obs for k in ["mangrove logging", "poaching", "arrests", "poached turtle sum"]):
        return True
    return res == "illegal use"


def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise columns and add IsIllegal / Plot_Category flags."""
    df = pd.DataFrame(df).copy()
    df.columns = [c.strip() for c in df.columns]

    def _col(name: str) -> str:
        if name in df.columns:
            return name
        alt = name.replace(" ", "_")
        if alt in df.columns:
            return alt
        return name.replace("_", " ")

    obs_col = _col("Observation Category 0")
    res_col = _col("Resource Use Type")

    df["_obs"] = df[obs_col].fillna("").str.lower().str.strip()
    df["_res"] = df[res_col].fillna("").str.lower().str.strip()
    df["IsIllegal"] = df.apply(lambda r: _is_illegal_event(r["_obs"], r["_res"]), axis=1)
    df["Plot_Category"] = df.apply(lambda r: "illegal fishing" if r["_res"] == "illegal use" else r["_obs"], axis=1)
    return df


def _add_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Derive Year / Month / YearQuarter from the patrol date column."""
    for dcol in ["Patrol Start Date", "Patrol_Start_Date", "event_date"]:
        if dcol in df.columns:
            parsed = pd.to_datetime(df[dcol], errors="coerce")
            if "Year" not in df.columns:
                df["Year"] = parsed.dt.year
            if "Month" not in df.columns:
                df["Month"] = parsed.dt.month
            if "YearQuarter" not in df.columns:
                df["YearQuarter"] = parsed.dt.to_period("Q")
            break
    return df


def _extract_year(df: pd.DataFrame) -> int:
    """
    Extract the single year value from a df slice produced by split_yearly.
    Falls back to the first non-null value if the 'Year' column already exists.
    """
    if "Year" in df.columns:
        val = df["Year"].dropna()
        if not val.empty:
            return int(val.iloc[0])
    # Try to derive it from the date column
    for dcol in ["Patrol Start Date", "Patrol_Start_Date", "event_date"]:
        if dcol in df.columns:
            parsed = pd.to_datetime(df[dcol], errors="coerce").dropna()
            if not parsed.empty:
                return int(parsed.iloc[0].year)
    raise ValueError("Cannot determine year from dataframe. " "Expected a 'Year' column or a recognisable date column.")


def _extract_year_quarter(df: pd.DataFrame) -> tuple[int, int]:
    """
    Extract (year, quarter) from a df slice produced by split_quarterly.
    Returns a (year, quarter) tuple where quarter is 1–4.
    """
    if "YearQuarter" in df.columns:
        val = df["YearQuarter"].dropna()
        if not val.empty:
            period = val.iloc[0]
            if hasattr(period, "year") and hasattr(period, "quarter"):
                return int(period.year), int(period.quarter)
    # Derive from date column
    for dcol in ["Patrol Start Date", "Patrol_Start_Date", "event_date"]:
        if dcol in df.columns:
            parsed = pd.to_datetime(df[dcol], errors="coerce").dropna()
            if not parsed.empty:
                dt = parsed.iloc[0]
                return int(dt.year), int((dt.month - 1) // 3 + 1)
    raise ValueError(
        "Cannot determine year/quarter from dataframe. "
        "Expected a 'YearQuarter' column or a recognisable date column."
    )


def _fig_to_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=True, include_plotlyjs="cdn")


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

    The year is derived automatically from the incoming df slice
    (produced by split_yearly), so it must NOT be passed explicitly.

    Reads green_thumb.png / red_thumb.png directly from `icons_dir`.
    Returns a full HTML string with the chart embedded as a base64 PNG.
    """
    df = _prep_df(df)
    df = _add_date_cols(df)

    # ── derive year from the slice ──────────────────────────
    year = _extract_year(df)
    logger.info(f"draw_community_feedback_table: activity={activity_key}, year={year}")

    thumbs_up_img = mpimg.imread(_icon(icons_dir, "thumbs_up"))
    thumbs_down_img = mpimg.imread(_icon(icons_dir, "thumbs_down"))

    all_stations = df[station_column].dropna().unique()
    station_village = (
        df[[station_column, village_column]].drop_duplicates().set_index(station_column)[village_column].to_dict()
    )

    year_df = df[df["Year"] == year]

    if activity_key == "illegal fishing":
        filtered = year_df[year_df["IsIllegal"] & (year_df["_res"] == "illegal use")]
    else:
        filtered = year_df[year_df["IsIllegal"] & (year_df["_obs"] == activity_key)]

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

    # ── FIX: return HTML, not raw base64 ───────────────────
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

    The year is derived automatically from the incoming df slice
    (produced by split_yearly), so it must NOT be passed explicitly.
    """
    df = _prep_df(df)
    df = _add_date_cols(df)

    # ── derive year from the slice ──────────────────────────
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
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

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
        height=max(600, len(villages) * 30),
        width=1300,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=200, r=100, t=80, b=80),
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

    The year is derived automatically from the incoming df slice
    (produced by split_yearly), so it must NOT be passed explicitly.
    """
    df = _prep_df(df)
    df = _add_date_cols(df)

    # ── derive year from the slice ──────────────────────────
    year = _extract_year(df)
    logger.info(f"draw_village_donut_chart: village={village}, year={year}")

    vdf = df[(df[village_column] == village) & (df["Year"] == year)]

    rows = []
    for key, label in DONUT_ACTIVITY_CONFIGS.items():
        if key == "illegal fishing":
            count = int((vdf["_res"] == "illegal use").sum())
        elif key == "damaging coral":
            dc_col = next(
                (c for c in [damaging_col, "Damaging_coral_activity", "Damaging coral activity"] if c in vdf.columns),
                None,
            )
            count = int(vdf[dc_col].notna().sum()) if dc_col else 0
        else:
            count = int((vdf["_obs"] == key).sum())
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
        title=dict(
            text=f"<b>{village} — {year}</b>",
            x=0.5,
            y=0.95,
            font=dict(size=TITLE_FONT_SIZE, family=FONT_FAMILY, color="black"),
            xanchor="center",
            yanchor="top",
        ),
        margin=dict(t=80, b=80, l=60, r=60),
        template="simple_white",
        height=600,
        width=700,
        showlegend=True,
        legend=dict(
            x=1.05,
            y=0.5,
            xanchor="left",
            yanchor="middle",
            orientation="v",
            title=dict(text="Activities", font=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color="black")),
            font=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY, color="black"),
        ),
    )
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

    The year and quarter are derived automatically from the incoming df
    slice (produced by split_quarterly), so they must NOT be passed
    explicitly.

    Reads boat.png / happy.png / sad.png directly from `icons_dir`.
    """
    df = _prep_df(df)
    df = _add_date_cols(df)

    # ── derive year + quarter from the slice ────────────────
    year, quarter = _extract_year_quarter(df)
    logger.info(f"draw_activity_leaderboard: activity={activity_key}, Q{quarter} {year}")

    no_mg = no_mangrove_sectors or NO_MANGROVE_SECTORS

    boat_b64 = _img_to_base64(_icon(icons_dir, "boat"))
    happy_b64 = _img_to_base64(_icon(icons_dir, "happy"))
    sad_b64 = _img_to_base64(_icon(icons_dir, "sad"))

    selected_yq = pd.Period(year=year, quarter=quarter, freq="Q")
    qdf = df[(df["YearQuarter"] == selected_yq) & df["IsIllegal"]]
    adf = qdf[qdf["_res"] == "illegal use"] if activity_key == "illegal fishing" else qdf[qdf["_obs"] == activity_key]

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

    x_vals = np.linspace(-1.6, max_events + 2, 500)
    z = np.tile(x_vals, (len(y_labels), 1))

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=x_vals,
            y=y_labels,
            colorscale=LEADERBOARD_COLORSCALE,
            opacity=0.38,
            showscale=False,
            hoverinfo="skip",
        )
    )
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

    images = []
    for i, (events, station) in enumerate(zip(sb["Events"], sb[station_column])):
        no_mg_row = activity_key == "mangrove logging" and station in no_mg
        if not no_mg_row:
            images.append(
                dict(
                    source=boat_b64,
                    xref="x",
                    yref="y",
                    x=(-1 if events == 0 else events),
                    y=i,
                    sizex=1.4,
                    sizey=1.4,
                    xanchor="center",
                    yanchor="middle",
                    layer="above",
                )
            )

    images += [
        dict(
            source=happy_b64,
            xref="x",
            yref="paper",
            x=-1.6,
            y=1.05,
            sizex=0.7,
            sizey=0.7,
            xanchor="center",
            yanchor="bottom",
            layer="above",
        ),
        dict(
            source=sad_b64,
            xref="x",
            yref="paper",
            x=max_events + 2.5,
            y=1.05,
            sizex=0.7,
            sizey=0.7,
            xanchor="center",
            yanchor="bottom",
            layer="above",
        ),
    ]

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
        ),
        xaxis=dict(range=[max_events + 3.0, -2.0], showticklabels=False, showgrid=False, showline=False),
        margin=dict(t=100, b=80, l=350, r=200),
        height=max(600, len(y_labels) * 50),
        width=1400,
        template="plotly_white",
        showlegend=False,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        images=images,
        shapes=[
            dict(
                type="line",
                x0=max_events + 2.5,
                x1=max_events + 2.5,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="#D30000", width=3),
            ),
            dict(type="line", x0=-1.6, x1=-1.6, y0=0, y1=1, yref="paper", line=dict(color="#006400", width=3)),
        ],
        annotations=[
            dict(
                x=-1.6,
                y=1,
                xref="x",
                yref="paper",
                text="Good (Fewer Events)",
                showarrow=False,
                font=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY, color="#006400"),
                xanchor="right",
                yanchor="bottom",
            ),
            dict(
                x=max_events + 2.5,
                y=1,
                xref="x",
                yref="paper",
                text="Bad (More Events)",
                showarrow=False,
                font=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY, color="#D30000"),
                xanchor="left",
                yanchor="bottom",
            ),
        ],
    )
    return _fig_to_html(fig)


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
        count = (
            int((vdf["_res"] == "illegal use").sum()) if key == "illegal fishing" else int((vdf["_obs"] == key).sum())
        )
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
        title=dict(
            text=f"<b>{village} — {period_label}</b>",
            x=0.5,
            font=dict(size=TITLE_FONT_SIZE, family=FONT_FAMILY, color="black"),
        ),
        width=1600,
        height=1200,
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
            title=dict(text="<b>Illegal Event</b>", font=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color="black")),
            tickvals=y_positions,
            ticktext=[f"<b>{r['Activity']}</b>" for _, r in counts_df.iterrows()],
            tickfont=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color="black"),
            autorange="reversed",
            showgrid=False,
            zeroline=False,
        ),
    )
    return fig


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

    The year and quarter are derived automatically from the incoming df
    slice (produced by split_quarterly), so they must NOT be passed
    explicitly.

    Reads activity icons directly from `icons_dir`.
    """
    df = _prep_df(df)
    df = _add_date_cols(df)

    # ── derive year + quarter from the slice ────────────────
    year, quarter = _extract_year_quarter(df)
    logger.info(f"draw_village_icon_bar: village={village}, Q{quarter} {year}")

    selected_yq = pd.Period(year=year, quarter=quarter, freq="Q")
    vdf = df[(df[village_column] == village) & (df["YearQuarter"] == selected_yq) & df["IsIllegal"]]

    icon_b64 = {key: _img_to_base64(_icon(icons_dir, key)) for key in ACTIVITY_CONFIGS}
    fig = _build_icon_bar_figure(vdf, village, f"Q{quarter} {year}", icon_b64, icons_per_row)
    return _fig_to_html(fig)


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

    The year and quarter are derived automatically from the incoming df
    slice (produced by split_quarterly), so they must NOT be passed
    explicitly.

    Reads activity icons directly from `icons_dir`.
    """
    df = _prep_df(df)
    df = _add_date_cols(df)

    # ── derive year + quarter from the slice ────────────────
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

        activity_seq = []
        for _, row in vdf.iterrows():
            if row["_res"] == "illegal use":
                activity_seq.append("illegal fishing")
            else:
                for k in ACTIVITY_CONFIGS:
                    if k in row["_obs"]:
                        activity_seq.append(k)
                        break

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
        title=dict(
            text=f"<b>Illegal Activities by Village — {period_label}</b>",
            x=0.5,
            font=dict(size=TITLE_FONT_SIZE, family=FONT_FAMILY, color="black"),
        ),
        width=1600,
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
            title=dict(text="<b>Villages</b>", font=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color="black")),
            tickvals=y_positions,
            ticktext=[f"<b>{v}</b>" for v in villages],
            tickfont=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color="black"),
            autorange="reversed",
            showgrid=False,
            zeroline=False,
        ),
    )
    return _fig_to_html(fig)


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

    All PNG arguments are keyed lists produced by mapvalues + html_to_png.
    Each list is a list of (key, path) tuples; this task picks the value
    from the first tuple in each list, matching the pattern used throughout
    the ecoscope-workflows ecosystem.

    Requires: docxtpl (pip install docxtpl)
    """
    from docxtpl import DocxTemplate, InlineImage
    from docx.shared import Inches
    from pathlib import Path as _Path

    logger.info("render_vg_report: starting")

    def _pick(keyed_list: list) -> str:
        """
        Extract a single file path from a mapvalues keyed list.
        Each element is a (key_tuple, value) pair — we just want the value
        from the first element, matching how ecoscope-workflows returns results.
        """
        if not keyed_list:
            raise ValueError("Received an empty list — no PNG was produced upstream.")
        first = keyed_list[0]
        # keyed list entries are (key, value) tuples
        if isinstance(first, (list, tuple)) and len(first) == 2:
            return str(first[1])
        # fallback: already a plain path string
        return str(first)

    tpl = DocxTemplate(template_path)

    def _img(keyed_list: list, width_in: float = 6.8) -> InlineImage:
        path = _pick(keyed_list)
        return InlineImage(tpl, path, width=Inches(width_in))

    context = {
        "report_year": str(report_year),
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

    out_dir = _Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"village_games_report_{report_year}.docx"
    tpl.save(str(out_file))

    logger.info(f"render_vg_report: saved to {out_file}")
    return str(out_file)
