# src/ecoscope_workflows_ext_bahari_hai/tasks/__init__.py

from ._example import add_one_thousand
from ._village_games import (
    draw_community_feedback_table,
    draw_monthly_heatmap,
    draw_village_donut_chart,
    draw_activity_leaderboard,
    draw_village_icon_bar,
    draw_all_villages_icon_bar,
)


__all__ = [
    "add_one_thousand",
    "draw_community_feedback_table",
    "draw_monthly_heatmap",
    "draw_village_donut_chart",
    "draw_activity_leaderboard",
    "draw_village_icon_bar",
    "draw_all_villages_icon_bar",
]
