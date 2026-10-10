#!/usr/bin/env python3
"""
Step 6 - build the on-site screening dashboard.

    python scripts/06_build_dashboard.py

Reads
    outputs/tables/predictions.csv         (step 4)
    outputs/tables/real_scan_features.csv  (step 1)
    outputs/tables/model_summary.csv       (step 3)
    data/raw/<session>/01_<HOLE>/<box>.curation.{jpg,csv}

Writes
    outputs/dashboard.html   one file, no network needed; opens in any browser

The page shows the actual core tray photo with every scan location marked
and coloured by declared grade, the five predicted indices for the selected
location with its spectrum as evidence, the down-core quality log, and the
model card. If the models were trained on synthetic labels a banner says so.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from coalqual.config import (DIAGNOSTIC_FEATURES, MODEL_DIR, OUT_DIR, TABLE_DIR,
                             TARGET_META, TARGETS, WAVELENGTHS)
from coalqual.io_boxscan import read_curation_csv

# Vertical offset between the curation "Track Y cm" and the visual centre of
# the track on this scanner's QAQC layout, measured once on box 0005.
TRACK_Y_OFFSET_CM = 1.3
PHOTO_WIDTH_PX = 1500


def _photo_payload(session_dir: Path, box_id: int, points: pd.DataFrame) -> dict | None:
    """Downsampled tray photo (base64 JPEG) + marker coordinates in photo pixels."""
    tag = f"{int(box_id):04d}"
    jpg = next(session_dir.glob(f"*/*_{tag}.curation.jpg"), None)
    csv = next(session_dir.glob(f"*/*_{tag}.curation.csv"), None)
    if jpg is None or csv is None:
        return None

    cur = read_curation_csv(csv)
    scale = float(cur["Image Scale cm/pix"])
    x0, y0 = float(cur["Image X cm"]), float(cur["Image Y cm"])

    im = Image.open(jpg).convert("RGB")
    W, H = im.size
    # Crop to the tray: from just above track 1 to just below the last track.
    track_ys = [float(v) for k, v in cur.items()
                if k.startswith("Track") and k.endswith(" Y cm")]
    if track_ys:
        top = max(0, int((y0 - max(track_ys) - 5.0) / scale))
        bot = min(H, int((y0 - min(track_ys) + 9.5) / scale))
        im = im.crop((0, top, W, bot))
    else:
        top = 0
    f = PHOTO_WIDTH_PX / W
    im = im.resize((PHOTO_WIDTH_PX, int(im.size[1] * f)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=78, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()

    markers = []
    for _, r in points.iterrows():
        tr = int(r.get("track", 2)) if pd.notna(r.get("track", 2)) else 2
        tx = float(cur.get(f"Track{tr} X cm", cur.get("Track2 X cm", 15.7)))
        ty = float(cur.get(f"Track{tr} Y cm", cur.get("Track2 Y cm", 33.2)))
        px = (tx - x0 + float(r["section_depth_cm"])) / scale * f
        py = ((y0 - ty + TRACK_Y_OFFSET_CM) / scale - top) * f
        markers.append(dict(id=r["sample_id"], x=round(px, 1), y=round(py, 1)))

    return dict(src=f"data:image/jpeg;base64,{b64}", w=PHOTO_WIDTH_PX,
                h=im.size[1], markers=markers,
                box_start=cur.get("Box Start Depth"), box_end=cur.get("Box End Depth"))


def build(pred_path: Path, feat_path: Path, summary_path: Path,
          raw_root: Path, out_path: Path) -> Path:
    pred = pd.read_csv(pred_path)
    feat = pd.read_csv(feat_path)
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()

    df = feat.merge(pred.drop(columns=[c for c in pred.columns
                                       if c in feat.columns and c != "sample_id"]),
                    on="sample_id", how="inner")
    df = df.sort_values(["session", "borehole_depth_m"]).reset_index(drop=True)

    synthetic = any(joblib.load(p).get("trained_on_synthetic", False)
                    for p in Path(MODEL_DIR).glob("model_*.joblib"))

    # ---- per-session payloads ------------------------------------------
    sessions = []
    xrf_show = ["LE", "Si", "Al", "Fe", "S", "Ca", "K", "Ti"]
    for sess, g in df.groupby("session", sort=True):
        box_id = int(g["box_id"].iloc[0]) if g["box_id"].notna().any() else 0
        pts = []
        for _, r in g.iterrows():
            pts.append(dict(
                id=r["sample_id"],
                mid=str(r.get("measurement_id", "")),
                depth=None if pd.isna(r["borehole_depth_m"]) else round(float(r["borehole_depth_m"]), 3),
                pos_cm=None if pd.isna(r["section_depth_cm"]) else float(r["section_depth_cm"]),
                pred={t: (None if pd.isna(r.get(f"pred_{t}")) else round(float(r[f"pred_{t}"]), 2))
                      for t in TARGETS},
                grade=str(r.get("pred_grade", "NA")),
                tier=str(r.get("ad_tier", "in")),
                ad=None if pd.isna(r.get("ad_score")) else round(float(r["ad_score"]), 1),
                fc_closure=None if pd.isna(r.get("pred_FC_closure")) else round(float(r["pred_FC_closure"]), 2),
                L=None if pd.isna(r.get("col_L")) else round(float(r["col_L"]), 1),
                munsell=str(r.get("munsell", "")),
                xrf={el: (None if pd.isna(r.get(f"xrf_{el}")) else round(float(r[f"xrf_{el}"]) / 1e4, 2))
                     for el in xrf_show},
                spectrum=[None if pd.isna(v) else round(float(v), 3)
                          for v in r[[f"R{w}" for w in WAVELENGTHS]].to_numpy()],
            ))
        photo = _photo_payload(raw_root / sess, box_id, g)
        sessions.append(dict(session=sess, box_id=box_id, hole=str(g["hole_id"].iloc[0]),
                             points=pts, photo=photo))

    model_card = []
    if not summary.empty:
        for _, r in summary.iterrows():
            model_card.append(dict(
                target=r["target"], model=r["best_model"], selector=r["best_selector"],
                n_features=int(r["n_features"]),
                cv_r2=round(float(r["cv_R2"]), 3), cv_rmse=round(float(r["cv_RMSE"]), 3),
                pred_r2=round(float(r["pred_R2"]), 3), pred_rmse=round(float(r["pred_RMSE"]), 3),
                rpd=round(float(r["pred_RPD"]), 2), n_cal=int(r["n_cal"]), n_pred=int(r["n_pred"]),
            ))

    payload = dict(
        sessions=sessions, model_card=model_card, synthetic=synthetic,
        wavelengths=WAVELENGTHS.tolist(),
        bands=[dict(lo=lo, hi=hi, nm=c, label=name.split("_")[0])
               for c, (lo, hi, name, _) in DIAGNOSTIC_FEATURES.items() if c >= 1100],
        targets=[dict(key=t, label=TARGET_META[t]["label"], unit=TARGET_META[t]["unit"])
                 for t in TARGETS],
    )

    html = TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    out_path.write_text(html, encoding="utf-8")
    return out_path


# ==========================================================================
TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BoxScan coal quality screening</title>
<style>
:root{
  --paper:#F6F4EE; --ink:#17181A; --graphite:#4A4D52; --rule:#D8D4CA;
  --lime:#B4CF3A; --lime-ink:#5F6E12;
  --g-hi:#1F5F5B; --g-mid:#7FA332; --g-lo:#D9A441; --g-min:#B5492B;
  --sim:#B03A2E;
  font-size:15px;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--paper);color:var(--ink);
  font-family:"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  font-variant-numeric:tabular-nums;line-height:1.4}
h1,h2,h3{margin:0;font-weight:600;letter-spacing:-.01em}
h1{font-size:1.55rem} h2{font-size:1.05rem} h3{font-size:.9rem;color:var(--graphite);font-weight:500}
button{font:inherit}
.wrap{max-width:1280px;margin:0 auto;padding:22px 24px 48px}
header{display:flex;flex-wrap:wrap;gap:12px 28px;align-items:baseline;
  padding-bottom:14px;border-bottom:2px solid var(--ink)}
header .sub{color:var(--graphite)}
.banner{margin:14px 0 0;padding:10px 14px;border-left:4px solid var(--sim);background:#F3E1DE;
  color:#5b1d16;font-size:.92rem}
.banner b{color:var(--sim)}
.boxes{display:flex;gap:8px;margin:18px 0 10px}
.boxes button{padding:6px 14px;border:1px solid var(--ink);background:transparent;
  border-radius:3px;cursor:pointer;color:var(--ink)}
.boxes button[aria-pressed="true"]{background:var(--ink);color:var(--paper)}
.boxes button:focus-visible{outline:3px solid var(--lime);outline-offset:2px}

.hero{position:relative;border:1px solid var(--rule);background:#111;overflow:hidden}
.hero img{display:block;width:100%;height:auto}
.hero svg{position:absolute;inset:0;width:100%;height:100%}
.hero .m{cursor:pointer}
.hero .m circle{stroke:#fff;stroke-width:1.6;transition:r .12s}
.hero .m:hover circle,.hero .m.sel circle{stroke:var(--lime);stroke-width:3}
.hero .m text{font-size:11px;fill:#fff;text-anchor:middle;pointer-events:none;
  paint-order:stroke;stroke:#000;stroke-width:3px}
.hero .m:focus{outline:none}
.hero .m:focus-visible circle{stroke:var(--lime);stroke-width:3}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;margin:8px 0 0;
  color:var(--graphite);font-size:.85rem}
.legend i{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px;vertical-align:-1px}

.grid{display:grid;grid-template-columns:1.25fr .95fr;gap:26px;margin-top:26px}
@media (max-width:900px){.grid{grid-template-columns:1fr}}

.point h2 span{color:var(--graphite);font-weight:400}
.indices{display:grid;grid-template-columns:repeat(5,1fr);gap:0;margin:12px 0 16px;
  border-top:1px solid var(--ink);border-bottom:1px solid var(--rule)}
.idx{padding:10px 8px 10px 0;border-right:1px solid var(--rule)}
.idx:last-child{border-right:0}
.idx .v{font-size:1.65rem;font-weight:600;line-height:1.1}
.idx .u{color:var(--graphite);font-size:.8rem}
.idx .l{font-size:.85rem;margin-top:2px}
@media (max-width:560px){.indices{grid-template-columns:repeat(3,1fr)}}
.grade{display:inline-flex;align-items:baseline;gap:8px;padding:6px 12px;border-radius:3px;
  color:#fff;font-weight:600;font-size:1.1rem}
.grade small{font-weight:400;opacity:.9;font-size:.8rem}
.meta{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;font-size:.88rem;
  color:var(--graphite);margin-top:12px}
.meta b{color:var(--ink);font-weight:500}

.spec svg{width:100%;height:auto;display:block}
.spec .band{fill:var(--lime);opacity:.22}
.spec .band-t{font-size:9px;fill:var(--graphite)}
.spec .ax{stroke:var(--ink);stroke-width:1}
.spec .tick{font-size:10px;fill:var(--graphite)}
.spec .line{fill:none;stroke:var(--ink);stroke-width:1.6}
.spec .ghost{fill:none;stroke:#9a9a9a;stroke-width:.7;opacity:.55}
.xrf{margin-top:14px}
.xrf table{border-collapse:collapse;width:100%;font-size:.86rem}
.xrf td{padding:3px 6px 3px 0;border-bottom:1px solid var(--rule)}
.xrf td.n{text-align:right;width:64px}
.xrf .bar{height:8px;background:var(--lime);display:inline-block;vertical-align:middle}

.log{margin-top:34px}
.log svg{width:100%;height:auto;display:block}
.log .ax{stroke:var(--ink);stroke-width:1}
.log .gl{stroke:var(--rule);stroke-width:1}
.log .tick{font-size:10px;fill:var(--graphite)}
.log .tt{font-size:11px;fill:var(--ink);font-weight:600}
.log .tr{fill:none;stroke:var(--ink);stroke-width:1.4}
.log .pt{stroke:#fff;stroke-width:.8;cursor:pointer}
.log .pt.sel{stroke:var(--lime);stroke-width:2.5}

.card{margin-top:34px}
.card table{border-collapse:collapse;width:100%;font-size:.88rem}
.card th{text-align:left;font-weight:500;color:var(--graphite);padding:6px 10px 6px 0;
  border-bottom:1px solid var(--ink)}
.card td{padding:7px 10px 7px 0;border-bottom:1px solid var(--rule)}
.card td.n,.card th.n{text-align:right}
.card td.rpd b{padding:1px 6px;border-radius:2px;color:#fff}
footer{margin-top:36px;padding-top:12px;border-top:1px solid var(--rule);
  color:var(--graphite);font-size:.8rem}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>Coal quality from the BoxScan tray</h1>
  <div class="sub" id="hdr-sub"></div>
</header>
<div id="banner"></div>

<div class="boxes" id="boxes" role="tablist" aria-label="core boxes"></div>
<div class="hero" id="hero"></div>
<div class="legend" id="legend"></div>

<div class="grid">
  <section class="point" id="point"></section>
  <section class="spec">
    <h3>Reflectance at this location, 350–2500 nm</h3>
    <div id="spec"></div>
    <div class="xrf" id="xrf"></div>
  </section>
</div>

<section class="log">
  <h2>Down-core quality log</h2>
  <div id="log"></div>
</section>

<section class="card">
  <h2>Model card</h2>
  <div id="card"></div>
</section>

<footer id="foot"></footer>
</div>

<script>
const D = __DATA__;
const state = {box:0, sel:0};
const T = D.targets;
const fmt = (v,t)=> v==null ? '–' : (t==='CV' ? Math.round(v).toLocaleString('en-IN') : v.toFixed(1));

function gradeColor(g){
  const n = parseInt(String(g).replace(/\D/g,''));
  if(isNaN(n)) return '#777';
  if(n<=6) return 'var(--g-hi)'; if(n<=10) return 'var(--g-mid)';
  if(n<=14) return 'var(--g-lo)'; return 'var(--g-min)';
}
function gradeHex(g){
  const n = parseInt(String(g).replace(/\D/g,''));
  if(isNaN(n)) return '#777777';
  if(n<=6) return '#1F5F5B'; if(n<=10) return '#7FA332';
  if(n<=14) return '#D9A441'; return '#B5492B';
}
const cur = ()=> D.sessions[state.box];
const pt  = ()=> cur().points[state.sel];

function render(){
  const s = cur();
  document.getElementById('hdr-sub').textContent =
    `Hole ${s.hole}, box ${s.box_id}, ${s.points.length} scan locations on the coal chip track`;
  document.getElementById('banner').innerHTML = D.synthetic ?
    `<div class="banner"><b>Models trained on simulated labels.</b> The spectra, pXRF and photo below are real
     BoxScan measurements; the predicted values demonstrate the software path only and are not measured coal quality.
     Retrain on the laboratory results before quoting any number.</div>` : '';
  renderBoxes(); renderHero(); renderPoint(); renderSpec(); renderXRF(); renderLog(); renderCard();
  document.getElementById('foot').textContent =
    'IIT (ISM) Dhanbad, CIL-IMin hackathon team AICSAQPS-2026-0005. Grade bands follow the CIL GCV schedule; FC closure = 100 − Ash − MC − VM.';
}
function renderBoxes(){
  const el = document.getElementById('boxes'); el.innerHTML='';
  D.sessions.forEach((s,i)=>{
    const b=document.createElement('button'); b.textContent=`Box ${s.box_id}`; b.role='tab';
    b.setAttribute('aria-pressed', i===state.box);
    b.onclick=()=>{state.box=i; state.sel=0; render();}; el.appendChild(b);
  });
}
function renderHero(){
  const s=cur(), h=document.getElementById('hero'), p=s.photo;
  if(!p){ h.innerHTML='<div style="padding:24px;color:#ccc">No tray photo found for this box. Copy the BoxScan session folder under data/raw and rebuild.</div>'; return; }
  const ms = p.markers.map((m,i)=>{
    const g = s.points[i]?.grade ?? 'NA';
    const tier = s.points[i]?.tier ?? 'in';
    const fill = tier==='out' ? '#2b2b2b' : gradeHex(g);
    const dash = tier==='edge' ? 'stroke-dasharray="4 3"' : '';
    const ring = tier==='out' ? `style="stroke:${gradeHex(g)};stroke-width:3"` : '';
    return `<g class="m ${i===state.sel?'sel':''}" data-i="${i}" tabindex="0" role="button" aria-label="scan ${i+1}, grade ${g}, ${tier}">
      <circle cx="${m.x}" cy="${m.y}" r="${i===state.sel?15:12}" fill="${fill}" ${ring} ${dash}/>
      <text x="${m.x}" y="${m.y+4}">${i+1}</text></g>`;}).join('');
  h.innerHTML = `<img src="${p.src}" alt="core tray photo, box ${s.box_id}">
    <svg viewBox="0 0 ${p.w} ${p.h}" preserveAspectRatio="none">${ms}</svg>`;
  h.querySelectorAll('.m').forEach(g=>{
    const go=()=>{state.sel=+g.dataset.i; render();};
    g.onclick=go; g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();go();}};
  });
  document.getElementById('legend').innerHTML =
    `<span>Marker colour is the declared grade:</span>
     <span><i style="background:var(--g-hi)"></i>G1–G6</span>
     <span><i style="background:var(--g-mid)"></i>G7–G10</span>
     <span><i style="background:var(--g-lo)"></i>G11–G14</span>
     <span><i style="background:var(--g-min)"></i>G15–G17</span>
     <span><i style="background:transparent;border:2px dashed #555"></i>extrapolated</span>
     <span><i style="background:#2b2b2b;border:2px solid var(--g-lo)"></i>outside calibration, send to lab</span>
     <span>Click a marker or a point on the log.</span>`;
}
function renderPoint(){
  const p=pt(), s=cur();
  const cells = T.map(t=>`<div class="idx"><div class="v">${fmt(p.pred[t.key],t.key)}</div>
      <div class="u">${t.unit}</div><div class="l">${t.label}</div></div>`).join('');
  document.getElementById('point').innerHTML = `
    <h2>Scan ${state.sel+1} of ${s.points.length} <span>${p.depth!=null?`at ${p.depth.toFixed(3)} m`:''}</span></h2>
    <div class="indices">${cells}</div>
    <span class="grade" style="background:${p.tier==='out'?'#555':gradeColor(p.grade)}">${p.grade}<small>${
      p.tier==='in' ? 'declared grade from predicted GCV' :
      p.tier==='edge' ? `extrapolated: ${p.ad}× beyond the calibration cloud, quote with caution` :
      `outside the calibration domain (${p.ad}×): not declarable, send this chip to the lab`}</small></span>
    <div class="meta">
      <span>Measurement</span><b>${p.mid}</b>
      <span>Position on track</span><b>${p.pos_cm!=null?p.pos_cm.toFixed(1)+' cm':'–'}</b>
      <span>FC by closure</span><b>${p.fc_closure!=null?p.fc_closure.toFixed(1)+' %':'–'} <span style="color:var(--graphite)">(vs model ${fmt(p.pred.FC,'FC')} %)</span></b>
      <span>Lightness L*</span><b>${p.L??'–'} <span style="color:var(--graphite)">${p.munsell}</span></b>
      <span>GCV, MJ/kg</span><b>${p.pred.CV!=null?(p.pred.CV*0.0041868).toFixed(2):'–'}</b>
    </div>`;
}
function renderSpec(){
  const p=pt(), s=cur(), W=560, H=300, m={l:42,r:10,t:16,b:34};
  const wl=D.wavelengths;
  const all = s.points.map(q=>q.spectrum);
  let ymax = 0; all.forEach(a=>a.forEach(v=>{ if(v!=null&&v>ymax) ymax=v; }));
  ymax = Math.ceil(ymax/5)*5 || 20;
  const X=v=>m.l+(v-wl[0])/(wl[wl.length-1]-wl[0])*(W-m.l-m.r);
  const Y=v=>H-m.b-(v/ymax)*(H-m.t-m.b);
  const path=a=>a.map((v,i)=>v==null?'':`${i&&a[i-1]!=null?'L':'M'}${X(wl[i]).toFixed(1)},${Y(v).toFixed(1)}`).join(' ');
  const bands=D.bands.map((b,i)=>`<rect class="band" x="${X(b.lo)}" y="${m.t}" width="${X(b.hi)-X(b.lo)}" height="${H-m.t-m.b}"/>
      <text class="band-t" x="${X(b.nm)}" y="${i%2? m.t-4 : m.t+9}" text-anchor="middle">${b.nm}</text>`).join('');
  const xt=[500,1000,1500,2000,2500].map(v=>`<line class="ax" x1="${X(v)}" x2="${X(v)}" y1="${H-m.b}" y2="${H-m.b+4}"/>
      <text class="tick" x="${X(v)}" y="${H-m.b+16}" text-anchor="middle">${v}</text>`).join('');
  const yt=[];for(let v=0;v<=ymax;v+=5)yt.push(`<line class="gl" x1="${m.l}" x2="${W-m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="#D8D4CA"/>
      <text class="tick" x="${m.l-6}" y="${Y(v)+3}" text-anchor="end">${v}</text>`);
  document.getElementById('spec').innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="reflectance spectrum">
    ${bands}${yt.join('')}
    ${all.map((a,i)=>i===state.sel?'':`<path class="ghost" d="${path(a)}"/>`).join('')}
    <path class="line" d="${path(p.spectrum)}"/>
    <line class="ax" x1="${m.l}" x2="${W-m.r}" y1="${H-m.b}" y2="${H-m.b}"/>${xt}
    <text class="tick" x="${(m.l+W-m.r)/2}" y="${H-4}" text-anchor="middle">wavelength, nm</text>
    <text class="tick" transform="translate(11,${(m.t+H-m.b)/2}) rotate(-90)" text-anchor="middle">reflectance, %</text>
  </svg>
  <div style="font-size:.8rem;color:var(--graphite);margin-top:4px">Shaded bands: OH 1400, CH 1700, H₂O 1900, aromatic C 2150, Al–OH 2200, CH 2300, carbonate 2350 nm. Grey traces are the other locations in this box.</div>`;
}
function renderXRF(){
  const p=pt(); const els=Object.keys(p.xrf);
  const rows=els.map(el=>{const v=p.xrf[el]; const w=v==null?0:Math.min(100, el==='LE'?v:v*4);
    return `<tr><td style="width:44px">${el}</td><td class="n">${v==null?'–':v.toFixed(el==='LE'?1:2)}</td>
      <td><span class="bar" style="width:${w}%"></span></td></tr>`;}).join('');
  document.getElementById('xrf').innerHTML=`<h3>pXRF at this location, wt %</h3><table><tbody>${rows}</tbody></table>
    <div style="font-size:.8rem;color:var(--graphite);margin-top:4px">LE is the light-element channel (C, H, O, N): the organic matrix. Si and Al track clay, Fe and S track pyrite.</div>`;
}
function renderLog(){
  const s=cur(), pts=s.points.filter(q=>q.depth!=null);
  if(!pts.length){document.getElementById('log').innerHTML='<p style="color:var(--graphite)">No depth information.</p>';return;}
  const n=T.length, colW=150, depthW=64, H=330, m={t:30,b:26};
  const W=depthW+n*colW+20;
  const d0=Math.min(...pts.map(q=>q.depth)), d1=Math.max(...pts.map(q=>q.depth));
  const Y=d=>m.t+(d-d0)/((d1-d0)||1)*(H-m.t-m.b);
  let svg=`<line class="ax" x1="${depthW-8}" x2="${depthW-8}" y1="${m.t}" y2="${H-m.b}"/>`;
  let lastY=-99; pts.forEach(q=>{ const y=Y(q.depth); if(y-lastY<13) return; lastY=y;
    svg+=`<text class="tick" x="${depthW-12}" y="${y+3}" text-anchor="end">${q.depth.toFixed(2)}</text>`;});
  svg+=`<text class="tick" x="${depthW-12}" y="${m.t-14}" text-anchor="end">depth, m</text>`;
  T.forEach((t,j)=>{
    const x0=depthW+j*colW+10, x1=x0+colW-22;
    const vals=pts.map(q=>q.pred[t.key]).filter(v=>v!=null);
    if(!vals.length) return;
    let lo=Math.min(...vals), hi=Math.max(...vals); const pad=(hi-lo)*0.15||1; lo-=pad; hi+=pad;
    const X=v=>x0+(v-lo)/(hi-lo)*(x1-x0);
    svg+=`<text class="tt" x="${x0}" y="${m.t-14}">${t.label}, ${t.unit}</text>`;
    svg+=`<line class="gl" x1="${x0}" x2="${x0}" y1="${m.t}" y2="${H-m.b}"/><line class="gl" x1="${x1}" x2="${x1}" y1="${m.t}" y2="${H-m.b}"/>`;
    svg+=`<text class="tick" x="${x0}" y="${H-m.b+14}">${fmt(lo+pad,t.key)}</text><text class="tick" x="${x1}" y="${H-m.b+14}" text-anchor="end">${fmt(hi-pad,t.key)}</text>`;
    const path=pts.map((q,i)=>q.pred[t.key]==null?'':`${i?'L':'M'}${X(q.pred[t.key]).toFixed(1)},${Y(q.depth).toFixed(1)}`).join(' ');
    svg+=`<path class="tr" d="${path}"/>`;
    pts.forEach(q=>{ if(q.pred[t.key]==null) return; const i=s.points.indexOf(q);
      svg+=`<circle class="pt ${i===state.sel?'sel':''}" data-i="${i}" cx="${X(q.pred[t.key])}" cy="${Y(q.depth)}" r="4.5" fill="${gradeHex(q.grade)}"/>`;});
  });
  document.getElementById('log').innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="down-core quality log">${svg}</svg>
    <div style="font-size:.8rem;color:var(--graphite);margin-top:4px">Each column is scaled to its own range; point colour is the declared grade.</div>`;
  document.querySelectorAll('.log .pt').forEach(c=>c.onclick=()=>{state.sel=+c.dataset.i; render();});
}
function renderCard(){
  if(!D.model_card.length){document.getElementById('card').innerHTML='<p style="color:var(--graphite)">Run scripts/03_train.py to populate the model card.</p>';return;}
  const rpdCol=v=>v>=3?'#1F5F5B':v>=2?'#D9A441':'#B5492B';
  const rows=D.model_card.map(r=>`<tr><td>${T.find(t=>t.key===r.target)?.label ?? r.target}</td><td>${r.model}</td><td>${r.selector}</td>
    <td class="n">${r.n_features}</td><td class="n">${r.n_cal}/${r.n_pred}</td><td class="n">${r.cv_r2.toFixed(3)}</td>
    <td class="n">${r.pred_r2.toFixed(3)}</td><td class="n">${r.pred_rmse}</td><td class="n rpd"><b style="background:${rpdCol(r.rpd)}">${r.rpd.toFixed(2)}</b></td></tr>`).join('');
  document.getElementById('card').innerHTML=`<table><thead><tr><th>Index</th><th>Model</th><th>Bands by</th><th class="n">Features</th>
    <th class="n">Cal / pred</th><th class="n">CV R²</th><th class="n">Held-out R²</th><th class="n">RMSE</th><th class="n">RPD</th></tr></thead><tbody>${rows}</tbody></table>
    <div style="font-size:.8rem;color:var(--graphite);margin-top:6px">RPD ≥ 3 quantitative, 2–3 screening, &lt; 2 qualitative. ${D.synthetic?'<b style="color:var(--sim)">These metrics are from simulated labels.</b>':''}</div>`;
}
render();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(Path(TABLE_DIR) / "predictions.csv"))
    ap.add_argument("--features", default=str(Path(TABLE_DIR) / "real_scan_features.csv"))
    ap.add_argument("--summary", default=str(Path(TABLE_DIR) / "model_summary.csv"))
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default=str(Path(OUT_DIR) / "dashboard.html"))
    args = ap.parse_args()
    out = build(Path(args.pred), Path(args.features), Path(args.summary),
                Path(args.raw), Path(args.out))
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} kB)")


if __name__ == "__main__":
    main()
