#!/usr/bin/env python3
import argparse, json, os
from datetime import datetime
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

def _load_json(p): 
    return json.load(open(p, "r", encoding="utf-8")) if p and os.path.exists(p) else {}
def _load_csv(p): 
    return pd.read_csv(p) if p and os.path.exists(p) else pd.DataFrame()

def add_title(flow, meta, styles):
    flow += [
        Paragraph(meta.get("title","Daily EMS Report"), styles["Title"]),
        Spacer(1, 0.2*cm),
        Paragraph(f'Day: {meta.get("day","")}', styles["Normal"]),
        Spacer(1, 0.5*cm),
    ]

def add_meta(flow, meta, styles):
    rows = [
        ["Objective", meta.get("objective","-")],
        ["Location",  meta.get("location","-")],
        ["PV size (kWp)", f'{meta.get("pv_kwp","-")}'],
        ["Battery (kWh/kW)", f'{meta.get("battery_kwh","-")} / {meta.get("battery_kw","-")}'],
        ["Generated at", meta.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M"))],
    ]
    t = Table([["Item","Value"]] + rows, colWidths=[6*cm, 10*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#F0F3F6")),
        ("GRID",(0,0),(-1,-1), 0.25, colors.grey),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
    ]))
    flow += [t, Spacer(1, 0.5*cm)]

def add_kpis(flow, kpi, styles):
    if not kpi: return
    hdr = ["Scenario","Cost (DKK)","CO₂ (kg)","Grid (kWh)","Load (kWh)","PV (kWh)","Self-cons.","Self-suff."]
    def row(lbl, d):
        if not d: return None
        f = lambda x, n: f"{float(x):.{n}f}" if x is not None else "-"
        return [lbl, f(d.get("cost_dkk"),2), f(d.get("co2_kg"),2), f(d.get("grid_kwh"),2),
                f(d.get("load_kwh"),2), f(d.get("pv_kwh"),2),
                f(d.get("self_consumption_pct"),1)+"%", f(d.get("self_sufficiency_pct"),1)+"%"]
    rows = list(filter(None, [
        row("No PV, No Battery", kpi.get("base")),
        row("PV only",          kpi.get("pv_only")),
        row("Battery only",     kpi.get("batt_only")),
        row("PV + Battery (EMS)",kpi.get("ems")),
    ]))
    flow += [Paragraph("Summary & Comparisons", styles["Heading2"]), Spacer(1,0.2*cm)]
    t = Table([hdr]+rows)
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#E6EEF8")),
        ("GRID",(0,0),(-1,-1), 0.25, colors.grey),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
    ]))
    flow += [t, Spacer(1, 0.5*cm)]

def add_slots(flow, df, styles):
    if df is None or df.empty: return
    df = df.copy()
    for c in ("start","end","soc_setpoint_pct","grid_charge_allowed"):
        if c not in df.columns: df[c] = ""
    df["grid_charge_allowed"] = df["grid_charge_allowed"].map({1:"Yes",0:"No"}).fillna(df["grid_charge_allowed"])
    data = [["Start","End","SOC Setpoint (%)","Grid charge?"]] + df[["start","end","soc_setpoint_pct","grid_charge_allowed"]].values.tolist()
    t = Table(data, colWidths=[3*cm,3*cm,5*cm,4*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#F4F7FA")),
        ("GRID",(0,0),(-1,-1), 0.25, colors.grey),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
    ]))
    flow += [Paragraph("Selected EMS Time Slots", styles["Heading2"]), Spacer(1,0.2*cm), t, Spacer(1,0.5*cm)]

def add_devices(flow, df, styles):
    if df is None or df.empty: return
    flow += [Paragraph("Schedules chosen today", styles["Heading2"]), Spacer(1,0.2*cm)]
    for _, r in df.iterrows():
        flow += [Paragraph(f'• <b>{r["Device"]}</b> → {r["Summary"]}', styles["Normal"])]
    flow += [Spacer(1,0.5*cm)]

def add_images(flow, paths, title, styles):
    paths = [p for p in paths if p and os.path.exists(p)]
    if not paths: return
    flow += [Paragraph(title, styles["Heading2"]), Spacer(1, 0.2*cm)]
    max_w = 17.5*cm
    for p in paths:
        flow += [Image(p, width=max_w, height=max_w*0.5, kind="proportional"), Spacer(1, 0.3*cm)]

def build_pdf(args):
    meta  = _load_json(args.meta)
    kpi   = _load_json(args.kpi)
    slots = _load_csv(args.slots)
    devs  = _load_csv(args.devices)

    doc = SimpleDocTemplate(args.out, pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=1.2*cm, bottomMargin=1.2*cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SubTitle", textColor=colors.grey))

    flow = []
    add_title(flow, meta, styles)
    add_meta(flow, meta, styles)
    add_kpis(flow, kpi, styles)
    add_slots(flow, slots, styles)
    add_devices(flow, devs, styles)
    add_images(flow, [args.img_loadpv], "Load and PV", styles)
    add_images(flow, [args.img_powersplit], "Power Split (Grid / PV / Battery / Load)", styles)
    add_images(flow, [args.img_soc], "Battery State of Charge", styles)
    doc.build(flow)
    print("✓ PDF written:", args.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--kpi", required=True)
    ap.add_argument("--slots", required=False, default="")
    ap.add_argument("--devices", required=False, default="")
    ap.add_argument("--img-loadpv", dest="img_loadpv", required=False, default="")
    ap.add_argument("--img-powersplit", dest="img_powersplit", required=False, default="")
    ap.add_argument("--img-soc", dest="img_soc", required=False, default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build_pdf(args)
