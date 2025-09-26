from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def save_csv(ts: pd.DataFrame, outdir: Path) -> Path:
    out = outdir / "timeseries.csv"
    ts.to_csv(out, index=True)
    return out

def save_plots(ts: pd.DataFrame, outdir: Path) -> Path:
    out = outdir / "plots.png"
    fig, ax = plt.subplots(figsize=(12,5))
    ts[['load_kw','pv_kw','grid_import_kw']].plot(ax=ax)
    ax.set_ylabel("kW")
    ax.set_title("Load / PV / Grid (kW)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

def save_summary_md(summary: dict, outdir: Path) -> Path:
    out = outdir / "summary.md"
    lines = ["# Daily Energy Report",
             "",
             f"**Objective:** {summary.get('objective','-')}",
             f"**Total cost (DKK):** {summary.get('total_cost_dkk',0):.2f}",
             f"**Total CO2 (kg):** {summary.get('total_co2_kg',0):.2f}",
             f"**Peak grid import (kW):** {summary.get('peak_grid_kw',0):.2f}",
             f"**Self-consumption (%):** {summary.get('self_consumption_pct',0):.1f}%",
             f"**Self-sufficiency (%):** {summary.get('self_sufficiency_pct',0):.1f}%",
             ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
