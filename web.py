import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
import io
import numpy as np
import zipfile
import base64
import serial
import time
import winsound
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

try:
    import tf_keras as keras      # best for Teachable Machine models
except ImportError:
    from tensorflow import keras

# ==========================================
# CAMERA SETTINGS  (edit these two lines)
# ==========================================
CAMERA_INDEX = None       # None = AUTO (first camera that works) | or force one: 0, 1, 2, 3 ...
CAMERA_BACKEND = "DSHOW"  # "DSHOW" | "MSMF" | "ANY"  (try another one if the webcam does not open)

# ==========================================
# ARDUINO SETTINGS
# ==========================================
ARDUINO_PORT = None       # None = AUTO-detect | or force one: "COM3", "COM5" ...

# ==========================================
# 1. PAGE CONFIG + MODERN CSS
# ==========================================
st.set_page_config(page_title="AOI Vision", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

:root{
  --bg:#070b12; --panel:#0d131c; --panel2:#111a26; --line:#1b2636;
  --txt:#e6edf3; --mut:#7d8ba1;
  --acc:#6366f1; --acc2:#8b8cf8;
  --ok:#3ddc84; --warn:#ffab2e; --bad:#ff5c5c;
}

html, body, .stApp, .stMarkdown, .stButton button, .stDownloadButton button,
[data-testid="stSidebar"] label { font-family:'Inter','Segoe UI',sans-serif; }
.stApp{
  background:
    radial-gradient(1100px 520px at 85% -10%, rgba(99,102,241,.12), transparent 60%),
    var(--bg);
  color:var(--txt);
}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{display:none !important;}
[data-testid="stHeader"]{background:transparent;}
.block-container{padding:1.4rem 2rem 2rem !important; max-width:100% !important;}
[data-testid="stVerticalBlock"]{gap:.9rem;}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"]{background:#0a0f17; border-right:1px solid var(--line);}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding-top:1rem;}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.7rem;}
section[data-testid="stSidebar"] label p{color:var(--txt); font-size:13px; font-weight:500;}
.brand{display:flex; align-items:center; gap:12px; padding:4px 2px 10px;}
.brand .logo{width:40px; height:40px; border-radius:11px; display:flex; align-items:center; justify-content:center;
  font-size:20px; background:linear-gradient(135deg,#4f46e5,#7c7ff7); box-shadow:0 6px 18px rgba(99,102,241,.35);}
.brand .bn{font-size:20px; font-weight:700; letter-spacing:.2px;}
.brand .bs{font-size:12px; color:var(--mut);}
.sec{font-size:12px; color:var(--mut); font-weight:600; margin:8px 0 2px; letter-spacing:.04em;}

/* ---------- CARDS ---------- */
.card, [data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--panel); border:1px solid var(--line) !important; border-radius:14px;
}
.card{padding:16px 18px;}
[data-testid="stVerticalBlockBorderWrapper"]{padding:6px 8px;}
.ctitle{display:flex; justify-content:space-between; align-items:center; font-size:14px; font-weight:600; margin-bottom:12px;}
.mut{color:var(--mut); font-weight:500; font-size:12px;}
.empty{border:1px dashed var(--line); border-radius:10px; padding:26px 10px; text-align:center; color:var(--mut); font-size:13px;}

/* ---------- TOP BAR ---------- */
.topbar{display:flex; justify-content:space-between; align-items:flex-start; gap:16px; margin-bottom:4px;}
.ttl{font-size:27px; font-weight:700; line-height:1.15;}
.sub{font-size:13px; color:var(--mut); margin-top:6px;}
.chips{display:flex; gap:10px; flex-wrap:wrap;}
.chip{display:inline-flex; align-items:center; gap:8px; background:var(--panel); border:1px solid var(--line);
  border-radius:10px; padding:8px 14px; font-size:13px; font-weight:500;}
.livetag{display:inline-flex; align-items:center; gap:6px; margin-left:8px; font-size:12px; color:var(--ok); font-weight:500;}
.fps{font-size:12px; color:var(--mut); background:var(--panel2); border:1px solid var(--line); padding:4px 10px; border-radius:8px;}

/* ---------- ALERT ---------- */
.alert{display:flex; align-items:center; gap:10px; padding:12px 16px; border-radius:12px; font-size:14px; font-weight:600;
  background:rgba(255,92,92,.10); border:1px solid rgba(255,92,92,.55); color:var(--bad);}

/* ---------- KPI ---------- */
.kpis{display:flex; align-items:stretch; min-height:96px; padding:14px 6px;}
.kpi{flex:1; padding:2px 20px; border-right:1px solid var(--line); display:flex; flex-direction:column; justify-content:center; gap:6px;}
.kpi:last-child{border-right:none;}
.kpi .v{font-size:32px; font-weight:700; line-height:1.05;}
.kpi .v small{font-size:13px; font-weight:500; color:var(--mut);}
.kpi .l{font-size:12px; color:var(--mut); font-weight:500;}

/* ---------- BARS ---------- */
.bar{height:6px; background:#1a2433; border-radius:99px; overflow:hidden; margin:6px 0 14px;}
.bar i{display:block; height:100%; border-radius:99px; transition:width .4s ease;}
.kpi .bar{margin:2px 0 0;}
.prow{display:flex; justify-content:space-between; font-size:13px; color:var(--txt);}
.prow b{font-weight:700;}

/* ---------- STATUS ---------- */
@keyframes pulse{0%{box-shadow:0 0 14px rgba(99,102,241,.35);}50%{box-shadow:0 0 4px rgba(99,102,241,.08);}100%{box-shadow:0 0 14px rgba(99,102,241,.35);}}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:.35;}}
.status{margin:10px 0 12px; padding:14px; border-radius:10px; text-align:center; font-weight:700; font-size:16px;}
.status.wait{background:rgba(99,102,241,.10); border:1px solid rgba(99,102,241,.55); color:var(--acc2); animation:pulse 2s infinite;}
.status.ok{background:rgba(61,220,132,.12); border:1px solid var(--ok); color:var(--ok); box-shadow:0 0 18px rgba(61,220,132,.18);}
.status.bad{background:rgba(255,92,92,.12); border:1px solid var(--bad); color:var(--bad); box-shadow:0 0 18px rgba(255,92,92,.18);}

/* ---------- THUMBNAILS ---------- */
.thumbs{display:grid; grid-template-columns:repeat(6,1fr); gap:8px;}
.th{border:1px solid var(--line); border-radius:9px; overflow:hidden; background:#0a0f17;}
.th img{width:100%; display:block; aspect-ratio:4/3; object-fit:cover;}
.th span{display:block; font-size:11px; font-weight:600; padding:4px 8px;}
.th.ok{border-color:rgba(61,220,132,.55);} .th.ok span{color:var(--ok); background:rgba(61,220,132,.08);}
.th.bad{border-color:rgba(255,92,92,.6);} .th.bad span{color:var(--bad); background:rgba(255,92,92,.08);}
.th.none{border-style:dashed; min-height:70px; display:flex; align-items:center; justify-content:center; color:var(--line);}

/* ---------- RECENT LIST ---------- */
.rec{display:flex; align-items:center; gap:12px; padding:10px; margin-bottom:8px; background:var(--panel2); border:1px solid var(--line); border-radius:10px;}
.rec img{width:60px; height:46px; object-fit:cover; border-radius:7px;}
.rec .info{flex:1; min-width:0;}
.rec .pn{font-size:13px; font-weight:600;}
.rec .pm{font-size:12px; color:var(--mut); margin-top:2px;}
.pill{font-size:11px; font-weight:700; padding:5px 10px; border-radius:7px;}
.pill.ok{color:var(--ok); background:rgba(61,220,132,.12);}
.pill.bad{color:var(--bad); background:rgba(255,92,92,.12);}

/* ---------- SYSTEM HEALTH ---------- */
.hrow{display:flex; justify-content:space-between; align-items:center; background:#0d131c; border:1px solid var(--line);
  border-radius:10px; padding:10px 12px; margin-bottom:8px; font-size:13px; font-weight:500;}
.hs{display:flex; align-items:center; gap:8px; font-size:11px; font-weight:700;}
.stApp .hs{font-family:'JetBrains Mono',monospace;}
.hs.ok{color:var(--ok);} .hs.bad{color:var(--bad);} .hs.idle{color:var(--warn);}
.dot{width:8px; height:8px; border-radius:50%; display:inline-block; background:currentColor; box-shadow:0 0 8px currentColor;}
.dot.ok{color:var(--ok);} .dot.idle{color:var(--warn);} .dot.bad{color:var(--bad); animation:blink 1s infinite;}
.hgrid-title{font-size:14px; font-weight:600; margin:2px 0 10px;}

/* ---------- SNAPSHOT ---------- */
.snap img{width:100%; display:block; border-radius:10px;}
[data-testid="stImage"] img{border-radius:10px;}

/* ---------- BUTTONS ---------- */
.stButton > button, .stDownloadButton > button{
  background:var(--panel2); border:1px solid var(--line); color:var(--txt); border-radius:10px; font-weight:600;}
.stButton > button:hover, .stDownloadButton > button:hover{border-color:var(--acc); color:#fff;}
.stDownloadButton > button:disabled{opacity:.35;}

@media (max-width:900px){ .thumbs{grid-template-columns:repeat(3,1fr);} .topbar{flex-direction:column;} }
</style>
""", unsafe_allow_html=True)


_v = tuple(int(x) for x in st.__version__.split(".")[:2])
FIT = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}   # works on old + new Streamlit


def H(s):
    """Collapse multi-line HTML into one line (avoids Markdown code-block parsing)."""
    return "".join(line.strip() for line in s.strip().splitlines())


# ==========================================
# 2. TRANSLATION
# ==========================================
LANG = {
    "EN": {
        "title": "Smart Automated Optical Inspection",
        "sub": "Real-time AI inspection catches defects instantly.",
        "panel": "Control Panel",
        "sens": "Light Sensitivity",
        "size": "Defect Size",
        "conf_th": "Pass Threshold (%)",
        "reset": "Reset Data",
        "tot": "Total Inspected",
        "pass": "Passed",
        "def": "Defects",
        "yield": "Yield Rate",
        "live": "Live Inspection",
        "chart": "Yield Trend",
        "pie": "Defect Distribution",
        "health": "System Health",
        "wait": "SYSTEM READY - Waiting for sensor...",
        "conf": "Model Confidence",
        "frame": "Analyzed Frame",
        "hist": "Recent Inspections",
        "hw_err": "Hardware Error: Arduino disconnected.",
        "cam_err": "Hardware Error: Camera not found.",
        "pass_msg": "PASSED - No defects",
        "fail_msg": "REJECTED - Found defects",
        "rate": "Inspection Rate",
        "unit": "parts/min",
        "perf": "Performance",
        "def_rate": "Defect Rate",
        "avg_def": "Avg. defects per rejected part",
        "no_data": "No inspections yet",
        "no_chart": "Collecting data...",
        "session": "Session",
        "started": "Started at",
        "estop": "🚨 EMERGENCY STOP",
        "estop_msg": "SYSTEM HALTED: EMERGENCY STOP ACTIVATED! Please disengage to resume production.",
        "target": "Yield Target (%)",
        "sound": "Sound Alerts",
        "alert": "Yield {y:.1f}% is below the {t}% target",
        "per_min": "Production per Minute",
        "reports": "Reports & Export",
        "r_all": "All results (CSV)",
        "r_pass": "Passed only (CSV)",
        "r_fail": "Failed only (CSV)",
        "r_charts": "Charts report (HTML)",
        "r_full": "Full report (HTML)",
        "r_zip": "Failed frames (ZIP)",
        "gallery": "Rejected Parts",
        "log": "Inspection Log",
        "generated": "Generated",
        "col_time": "Time", "col_part": "Part", "col_status": "Status", "col_def": "Defects",
        "cam": "Camera 01", "ard": "Arduino", "eng": "Vision Engine", "las": "Laser Scanner", "line": "Production Line",
        "online": "ONLINE", "offline": "OFFLINE", "ready": "READY", "active": "ACTIVE",
        "running": "RUNNING", "stopped": "STOPPED", "idle": "IDLE",
    },
    "ZH": {
        "title": "智能自动光学检测系统",
        "sub": "实时 AI 检测，即时发现缺陷。",
        "panel": "控制面板",
        "sens": "光照灵敏度",
        "size": "最小缺陷尺寸",
        "conf_th": "合格判定阈值 (%)",
        "reset": "重置数据",
        "tot": "检测总数",
        "pass": "合格数量",
        "def": "缺陷数量",
        "yield": "当前良品率",
        "live": "实时检测",
        "chart": "良品率趋势",
        "pie": "缺陷分布",
        "health": "系统状态监控",
        "wait": "系统就绪 - 等待传感器触发...",
        "conf": "模型置信度",
        "frame": "最新分析画面",
        "hist": "最近记录",
        "hw_err": "硬件错误：Arduino 未连接。",
        "cam_err": "硬件错误：未找到摄像头。",
        "pass_msg": "合格 - 未发现缺陷",
        "fail_msg": "拒收 - 发现缺陷",
        "rate": "检测速率",
        "unit": "件/分钟",
        "perf": "性能指标",
        "def_rate": "缺陷率",
        "avg_def": "每个不合格品平均缺陷数",
        "no_data": "暂无检测记录",
        "no_chart": "数据收集中...",
        "session": "本次运行",
        "started": "开始时间",
        "estop": "🚨 紧急停止",
        "estop_msg": "系统已停止：紧急停止已激活！请解除后恢复生产。",
        "target": "良品率目标 (%)",
        "sound": "声音提示",
        "alert": "良品率 {y:.1f}% 低于目标 {t}%",
        "per_min": "每分钟产量",
        "reports": "报告与导出",
        "r_all": "全部结果 (CSV)",
        "r_pass": "仅合格 (CSV)",
        "r_fail": "仅不合格 (CSV)",
        "r_charts": "图表报告 (HTML)",
        "r_full": "完整报告 (HTML)",
        "r_zip": "缺陷画面 (ZIP)",
        "gallery": "拒收零件",
        "log": "检测记录",
        "generated": "生成时间",
        "col_time": "时间", "col_part": "零件", "col_status": "状态", "col_def": "缺陷",
        "cam": "摄像头 01", "ard": "Arduino", "eng": "视觉引擎", "las": "激光扫描仪", "line": "生产线",
        "online": "在线", "offline": "离线", "ready": "就绪", "active": "运行中",
        "running": "运行中", "stopped": "已停止", "idle": "空闲",
    },
}

# ==========================================
# 3. SESSION STATE
# ==========================================
S = st.session_state
_defaults = {
    "total": 0, "ok": 0, "defect": 0,
    "history": [], "yield_data": [], "recent": [], "times": [], "fails": [],
    "conf": 100, "last_snap": None,
    "started": datetime.now().strftime("%H:%M:%S"),
    "started_ts": time.time(),
}
for _k, _v in _defaults.items():
    if _k not in S:
        S[_k] = _v

# ==========================================
# 4. SIDEBAR  (System Health lives on the LEFT)
# ==========================================

with st.sidebar:
    st.markdown(H("""
        <div class="brand"><div class="logo">🛡️</div>
        <div><div class="bn">AOI Vision</div><div class="bs">Quality Inspection</div></div></div>
    """), unsafe_allow_html=True)

    is_zh = st.toggle("🌐 中文 / English", value=False)
    l = "ZH" if is_zh else "EN"
    T = LANG[l]

    health_ui = st.empty()          # filled after hardware init

    st.markdown(f'<div class="sec">{T["panel"]}</div>', unsafe_allow_html=True)
    e_stop = st.toggle(T["estop"], value=False)
    sound_on = st.toggle(f"🔔 {T['sound']}", value=True)
    conf_th = st.slider(f"🔍 {T['conf_th']}", 10, 99, 80)
    target = st.slider(f"🎯 {T['target']}", 50, 100, 95)

    if st.button(f"🗑️ {T['reset']}", **FIT):
        for k in ("history", "yield_data", "recent", "times", "fails"):
            S[k] = []
        S.total = S.ok = S.defect = 0
        S.conf = 100
        S.last_snap = None
        st.rerun()

    st.markdown(H(f"""
        <div class="card" style="margin-top:10px;padding:14px 16px;">
        <div class="mut">{T["session"]}</div>
        <div style="font-size:15px;font-weight:600;margin-top:4px;">{T["started"]} {S.started}</div></div>
    """), unsafe_allow_html=True)


def render_health(cam_ok, ard_ok, halted=False):
    run = cam_ok and ard_ok and not halted
    stop = ("bad", T["stopped"])
    if halted:
        cam_s = ard_s = eng_s = las_s = line_s = stop
    else:
        cam_s = ("ok", T["online"]) if cam_ok else ("bad", T["offline"])
        ard_s = ("ok", T["online"]) if ard_ok else ("bad", T["offline"])
        eng_s = ("ok", T["ready"])
        las_s = ("ok", T["active"]) if run else ("idle", T["idle"])
        line_s = ("ok", T["running"]) if run else stop

    def row(icon, label, s):
        return (f'<div class="hrow"><span>{icon} {label}</span>'
                f'<span class="hs {s[0]}"><i class="dot {s[0]}"></i>{s[1]}</span></div>')

    rows = (row("📷", T["cam"], cam_s) + row("🔌", T["ard"], ard_s) + row("🔬", T["eng"], eng_s)
            + row("⚡", T["las"], las_s) + row("🏭", T["line"], line_s))
    health_ui.markdown(f'<div class="hgrid-title">🛠️ {T["health"]}</div>{rows}', unsafe_allow_html=True)


# ==========================================
# 5. HEADER + EMERGENCY STOP
# ==========================================
st.markdown(H(f"""
    <div class="topbar">
    <div><div class="ttl">🏭 {T["title"]}</div><div class="sub">{T["sub"]}</div></div>
    <div class="chips"><span class="chip"><i class="dot ok"></i>Line 1 - Station A</span>
    <span class="chip">Arduino · 9600</span></div></div>
"""), unsafe_allow_html=True)

if e_stop:
    render_health(False, False, halted=True)
    st.error(T["estop_msg"], icon="🚨")
    st.stop()

alert_ui = st.empty()

# ==========================================
# 6. LAYOUT
# ==========================================
kpi_l, kpi_r = st.columns([1.7, 1], gap="medium")
kpi_left = kpi_l.empty()
kpi_right = kpi_r.empty()

col_left, col_right = st.columns([1.9, 1], gap="medium")

with col_left:
    with st.container(border=True):
        head_ui = st.empty()
        live_cam = st.empty()
        status_ui = st.empty()
        thumbs_ui = st.empty()
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown(f'<div class="ctitle">🥧 {T["pie"]}</div>', unsafe_allow_html=True)
            pie_ui = st.empty()
    with c2:
        with st.container(border=True):
            st.markdown(f'<div class="ctitle">📈 {T["chart"]}</div>', unsafe_allow_html=True)
            chart_ui = st.empty()
    with st.container(border=True):
        st.markdown(f'<div class="ctitle">📊 {T["per_min"]}</div>', unsafe_allow_html=True)
        bar_ui = st.empty()
    with st.container(border=True):
        st.markdown(f'<div class="ctitle">📥 {T["reports"]}</div>', unsafe_allow_html=True)
        reports_ui = st.empty()

with col_right:
    with st.container(border=True):
        st.markdown(f'<div class="ctitle">🖼️ {T["frame"]}</div>', unsafe_allow_html=True)
        snap_ui = st.empty()
    recent_ui = st.empty()
    with st.container(border=True):
        st.markdown(f'<div class="ctitle">🎯 {T["perf"]}</div>', unsafe_allow_html=True)
        gauge_ui = st.empty()
        perf_ui = st.empty()


# ==========================================
# 7. HELPERS (images, figures, reports)
# ==========================================
def to_jpg(img, w=None, q=80):
    if w:
        h = max(1, int(img.shape[0] * w / img.shape[1]))
        img = cv2.resize(img, (w, h))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()


def to_b64(img, w=200):
    return base64.b64encode(to_jpg(img, w, 72)).decode()


def bar(pct, color):
    pct = max(0, min(100, pct))
    return f'<div class="bar"><i style="width:{pct:.1f}%;background:{color}"></i></div>'


def stats():
    tot, ok, de = S.total, S.ok, S.defect
    y = (ok / tot * 100) if tot > 0 else 100.0
    dr = (de / tot * 100) if tot > 0 else 0.0
    avg = (sum(h["Defects"] for h in S.history) / de) if de > 0 else 0.0
    return tot, ok, de, y, dr, avg


def per_minute():
    d = {}
    for h in S.history:
        m = h["Time"][:5]
        d.setdefault(m, [0, 0])[0 if "PASS" in h["Status"] else 1] += 1
    keys = sorted(d)[-30:]
    return keys, [d[k][0] for k in keys], [d[k][1] for k in keys]


def dark_layout(fig, h=215):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#9aa7bb", size=11), height=h,
                      margin=dict(t=4, b=4, l=4, r=4))


def pie_fig(h=215):
    if S.total <= 0:
        return None
    fig = go.Figure(go.Pie(
        labels=[T["pass"], T["def"]], values=[S.ok, S.defect], hole=0.7, sort=False, textinfo="none",
        marker=dict(colors=["#3ddc84", "#ff5c5c"], line=dict(color="#0d131c", width=3)),
        domain=dict(x=[0, 1], y=[0.16, 1])))
    dark_layout(fig, h)
    fig.update_layout(
        legend=dict(orientation="h", x=0.5, xanchor="center", y=0, yanchor="bottom", font=dict(color="#c9d1d9", size=12)),
        annotations=[
            dict(text=f"<b>{S.total}</b>", x=0.5, y=0.63, showarrow=False, font=dict(size=28, color="#e6edf3")),
            dict(text=T["tot"], x=0.5, y=0.52, showarrow=False, font=dict(size=11, color="#7d8ba1")),
        ])
    return fig


def line_fig(h=215):
    if len(S.yield_data) < 2:
        return None
    y = S.yield_data
    fig = go.Figure(go.Scatter(
        x=list(range(1, len(y) + 1)), y=y, mode="lines",
        line=dict(color="#6366f1", width=2.6, shape="spline"),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.13)"))
    fig.add_hline(y=target, line=dict(color="#ffab2e", width=1.5, dash="dot"))
    dark_layout(fig, h)
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, color="#7d8ba1"),
        yaxis=dict(gridcolor="#1b2636", zeroline=False, color="#7d8ba1",
                   range=[max(0, min(min(y), target) - 10), 101], ticksuffix="%"))
    return fig


def bar_fig(h=215):
    keys, p, f = per_minute()
    if not keys:
        return None
    fig = go.Figure([
        go.Bar(x=keys, y=p, name=T["pass"], marker_color="#3ddc84"),
        go.Bar(x=keys, y=f, name=T["def"], marker_color="#ff5c5c"),
    ])
    dark_layout(fig, h)
    fig.update_layout(
        barmode="stack", bargap=0.35, margin=dict(t=28, b=4, l=4, r=4),
        legend=dict(orientation="h", x=1, xanchor="right", y=1.18, font=dict(color="#c9d1d9", size=12)),
        xaxis=dict(showgrid=False, color="#7d8ba1"),
        yaxis=dict(gridcolor="#1b2636", color="#7d8ba1"))
    return fig


def gauge_fig(h=175):
    _, _, _, y, _, _ = stats()
    good = y >= target
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=y,
        number=dict(suffix="%", valueformat=".1f", font=dict(size=34, color="#e6edf3")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#7d8ba1", tickfont=dict(size=10, color="#7d8ba1")),
            bar=dict(color="#3ddc84" if good else "#ff5c5c", thickness=0.28),
            bgcolor="#111a26", borderwidth=0,
            steps=[dict(range=[0, target], color="rgba(255,92,92,.14)"),
                   dict(range=[target, 100], color="rgba(61,220,132,.12)")],
            threshold=dict(line=dict(color="#ffab2e", width=3), thickness=0.85, value=target))))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#9aa7bb"), height=h,
                      margin=dict(t=18, b=0, l=18, r=18))
    return fig


REPORT_CSS = (
    "*{box-sizing:border-box}body{margin:0;background:#070b12;color:#e6edf3;font-family:Inter,'Segoe UI',Arial,sans-serif}"
    ".wrap{max-width:1100px;margin:0 auto;padding:32px 24px}"
    "header{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:1px solid #1b2636;padding-bottom:16px;margin-bottom:22px}"
    "h1{margin:0;font-size:26px}.sub{color:#7d8ba1;font-size:13px;margin-top:6px}"
    ".kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:22px}"
    ".k{background:#0d131c;border:1px solid #1b2636;border-radius:12px;padding:16px}"
    ".kv{font-size:30px;font-weight:700}.kl{color:#7d8ba1;font-size:12px;margin-top:4px}"
    ".grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}"
    ".card{background:#0d131c;border:1px solid #1b2636;border-radius:12px;padding:16px;margin-bottom:14px}"
    ".card h2{font-size:14px;margin:0 0 10px}"
    "table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #1b2636}"
    "th{color:#7d8ba1;font-weight:600}td.p{color:#3ddc84}td.f{color:#ff5c5c}"
    ".gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}"
    "figure{margin:0;background:#111a26;border:1px solid rgba(255,92,92,.5);border-radius:10px;overflow:hidden}"
    "figure img{width:100%;display:block}figcaption{padding:8px 10px;font-size:12px;display:flex;flex-direction:column;gap:2px}"
    "figcaption span{color:#7d8ba1}"
    ".empty{color:#7d8ba1;padding:20px;text-align:center;border:1px dashed #1b2636;border-radius:10px}"
    "@media print{body{-webkit-print-color-adjust:exact;print-color-adjust:exact}.card,figure{break-inside:avoid}}"
    "@media(max-width:700px){.grid{grid-template-columns:1fr}}"
)


def build_report_html(full=True):
    tot, ok, de, y, dr, avg = stats()

    def fh(fig):
        if fig is None:
            return f'<div class="empty">{T["no_chart"]}</div>'
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})

    kp = "".join(
        f'<div class="k"><div class="kv" style="color:{c}">{v}</div><div class="kl">{lbl}</div></div>'
        for v, lbl, c in [
            (tot, T["tot"], "#e6edf3"), (ok, T["pass"], "#3ddc84"), (de, T["def"], "#ff5c5c"),
            (f"{y:.1f}%", T["yield"], "#8b8cf8"), (f"{dr:.1f}%", T["def_rate"], "#ffab2e"),
        ])

    body = (
        f'<header><div><h1>🏭 {T["title"]}</h1><div class="sub">AOI Vision · {T["reports"]}</div></div>'
        f'<div class="sub">{T["generated"]}: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div></header>'
        f'<div class="kpis">{kp}</div>'
        f'<div class="grid"><div class="card"><h2>🥧 {T["pie"]}</h2>{fh(pie_fig(240))}</div>'
        f'<div class="card"><h2>📈 {T["chart"]}</h2>{fh(line_fig(240))}</div></div>'
        f'<div class="card"><h2>📊 {T["per_min"]}</h2>{fh(bar_fig(260))}</div>'
    )

    if full:
        rows = "".join(
            f'<tr><td>{h["Time"]}</td><td>{h.get("Part", "-")}</td>'
            f'<td class="{"p" if "PASS" in h["Status"] else "f"}">{h["Status"]}</td><td>{h["Defects"]}</td></tr>'
            for h in S.history)
        body += (f'<div class="card"><h2>📋 {T["log"]}</h2><table><thead><tr><th>{T["col_time"]}</th><th>{T["col_part"]}</th>'
                 f'<th>{T["col_status"]}</th><th>{T["col_def"]}</th></tr></thead><tbody>{rows}</tbody></table></div>')
        if S.fails:
            cards = "".join(
                f'<figure><img src="data:image/jpeg;base64,{base64.b64encode(fr["jpg"]).decode()}"/>'
                f'<figcaption><b>{fr["part"]}</b><span>{fr["time"]} · {fr["defects"]} {T["def"]}</span></figcaption></figure>'
                for fr in S.fails[:30])
            body += f'<div class="card"><h2>❌ {T["gallery"]}</h2><div class="gal">{cards}</div></div>'

    return ('<!doctype html><html><head><meta charset="utf-8"><title>AOI Report</title>'
            '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
            f'<style>{REPORT_CSS}</style></head><body><div class="wrap">{body}</div></body></html>')


def fails_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fr in S.fails:
            z.writestr(f'{fr["part"]}_{fr["time"].replace(":", "-")}_{fr["defects"]}defects.jpg', fr["jpg"])
    return buf.getvalue()


def dl(label, data, name, mime, key, disabled=False):
    kw = dict(file_name=name, mime=mime, key=key, disabled=disabled, **FIT)
    try:
        st.download_button(label, data, on_click="ignore", **kw)   # newer Streamlit: no rerun (loop keeps running)
    except TypeError:
        st.download_button(label, data, **kw)


def render_reports():
    cols = ["Time", "Part", "Status", "Defects"]
    has = len(S.history) > 0
    has_fail = len(S.fails) > 0
    df = pd.DataFrame(S.history, columns=cols)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    k = S.total

    csv_all = df.to_csv(index=False).encode("utf-8-sig")
    csv_pass = df[df["Status"].str.contains("PASS")].to_csv(index=False).encode("utf-8-sig")
    csv_fail = df[df["Status"].str.contains("FAIL")].to_csv(index=False).encode("utf-8-sig")
    html_charts = build_report_html(full=False).encode("utf-8") if has else b""
    html_full = build_report_html(full=True).encode("utf-8") if has else b""
    zip_bytes = fails_zip() if has_fail else b""

    with reports_ui.container():
        a = st.columns(3)
        with a[0]:
            dl(f"📄 {T['r_all']}", csv_all, f"aoi_all_{stamp}.csv", "text/csv", f"dl_all_{k}", not has)
        with a[1]:
            dl(f"✅ {T['r_pass']}", csv_pass, f"aoi_passed_{stamp}.csv", "text/csv", f"dl_pass_{k}", S.ok == 0)
        with a[2]:
            dl(f"❌ {T['r_fail']}", csv_fail, f"aoi_failed_{stamp}.csv", "text/csv", f"dl_fail_{k}", S.defect == 0)
        b = st.columns(3)
        with b[0]:
            dl(f"📈 {T['r_charts']}", html_charts, f"aoi_charts_{stamp}.html", "text/html", f"dl_charts_{k}", not has)
        with b[1]:
            dl(f"📑 {T['r_full']}", html_full, f"aoi_report_{stamp}.html", "text/html", f"dl_full_{k}", not has)
        with b[2]:
            dl(f"🖼️ {T['r_zip']}", zip_bytes, f"aoi_failed_frames_{stamp}.zip", "application/zip", f"dl_zip_{k}", not has_fail)


def beep(freq, dur):
    if sound_on:
        winsound.Beep(freq, dur)


def toast(msg, icon):
    try:
        st.toast(msg, icon=icon)
    except Exception:
        pass


# ==========================================
# 8. RENDER FUNCTIONS
# ==========================================
def live_head(fps=None):
    up = int(time.time() - S.started_ts)
    uptime = f"{up // 3600:02d}:{up % 3600 // 60:02d}:{up % 60:02d}"
    chips = f'<span class="fps">⏱ {uptime}</span>'
    if fps is not None:
        chips = f'<span class="fps">⚡ {fps} FPS</span>' + chips
    head_ui.markdown(
        f'<div class="ctitle"><span>📷 {T["live"]}<span class="livetag"><i class="dot ok"></i>Live</span></span>'
        f'<span style="display:flex;gap:8px">{chips}</span></div>',
        unsafe_allow_html=True)


def set_status(kind, text):
    status_ui.markdown(f'<div class="status {kind}">{text}</div>', unsafe_allow_html=True)


def render_thumbs():
    items = S.recent[:6]
    cells = ""
    for it in items:
        cls = "ok" if it["ok"] else "bad"
        label = "PASS" if it["ok"] else "FAIL"
        cells += f'<div class="th {cls}"><img src="data:image/jpeg;base64,{it["img"]}"/><span>{label}</span></div>'
    for _ in range(6 - len(items)):
        cells += '<div class="th none">—</div>'
    thumbs_ui.markdown(f'<div class="thumbs">{cells}</div>', unsafe_allow_html=True)


def render_snap():
    if S.last_snap:
        snap_ui.markdown(f'<div class="snap"><img src="data:image/jpeg;base64,{S.last_snap}"/></div>', unsafe_allow_html=True)
    else:
        snap_ui.markdown(f'<div class="empty">{T["no_data"]}</div>', unsafe_allow_html=True)


def render_recent():
    rows = ""
    for it in S.recent[:5]:
        cls = "ok" if it["ok"] else "bad"
        label = "PASS" if it["ok"] else "FAIL"
        rows += (f'<div class="rec"><img src="data:image/jpeg;base64,{it["img"]}"/>'
                 f'<div class="info"><div class="pn">{it["part"]}</div>'
                 f'<div class="pm">{it["time"]} · {it["defects"]} {T["def"]}</div></div>'
                 f'<span class="pill {cls}">{label}</span></div>')
    if not rows:
        rows = f'<div class="empty">{T["no_data"]}</div>'
    recent_ui.markdown(
        f'<div class="card"><div class="ctitle"><span>📋 {T["hist"]}</span><span class="mut">{len(S.history)}</span></div>{rows}</div>',
        unsafe_allow_html=True)


def update_dashboard():
    tot, ok, de, y, dr, avg = stats()

    now = time.time()
    S.times = [t for t in S.times if now - t <= 60]
    rate = len(S.times)

    if tot >= 5 and y < target:
        alert_ui.markdown(f'<div class="alert">⚠️ {T["alert"].format(y=y, t=target)}</div>', unsafe_allow_html=True)
    else:
        alert_ui.empty()

    kpi_left.markdown(H(f"""
        <div class="card kpis">
        <div class="kpi"><div class="v">{tot}</div><div class="l">📦 {T["tot"]}</div></div>
        <div class="kpi"><div class="v" style="color:var(--ok)">{ok}</div><div class="l">✅ {T["pass"]}</div></div>
        <div class="kpi"><div class="v" style="color:var(--bad)">{de}</div><div class="l">❌ {T["def"]}</div></div>
        <div class="kpi"><div class="v" style="color:var(--acc2)">{y:.1f}%</div><div class="l">📊 {T["yield"]}</div></div>
        </div>
    """), unsafe_allow_html=True)

    kpi_right.markdown(H(f"""
        <div class="card kpis">
        <div class="kpi"><div class="l">{T["rate"]}</div><div class="v">{rate}<small> {T["unit"]}</small></div></div>
        <div class="kpi"><div class="l">🤖 {T["conf"]}</div><div class="v">{S.conf:.0f}%</div>{bar(S.conf, "#6366f1")}</div>
        </div>
    """), unsafe_allow_html=True)

    perf_ui.markdown(H(f"""
        <div class="prow"><span>{T["yield"]}</span><b>{y:.1f}%</b></div>{bar(y, "#3ddc84")}
        <div class="prow"><span>{T["def_rate"]}</span><b>{dr:.1f}%</b></div>{bar(dr, "#ff5c5c")}
        <div class="prow"><span>{T["avg_def"]}</span><b>{avg:.1f}</b></div>
    """), unsafe_allow_html=True)


def update_charts():
    cfg = {"displayModeBar": False}
    k = S.total

    fig = pie_fig()
    if fig is not None:
        pie_ui.plotly_chart(fig, config=cfg, **FIT, key=f"pie_{k}")
    else:
        pie_ui.markdown(f'<div class="empty">{T["no_data"]}</div>', unsafe_allow_html=True)

    fig = line_fig()
    if fig is not None:
        chart_ui.plotly_chart(fig, config=cfg, **FIT, key=f"line_{k}")
    else:
        chart_ui.markdown(f'<div class="empty">{T["no_chart"]}</div>', unsafe_allow_html=True)

    fig = bar_fig()
    if fig is not None:
        bar_ui.plotly_chart(fig, config=cfg, **FIT, key=f"bar_{k}")
    else:
        bar_ui.markdown(f'<div class="empty">{T["no_chart"]}</div>', unsafe_allow_html=True)

    gauge_ui.plotly_chart(gauge_fig(), config=cfg, **FIT, key=f"gauge_{k}")


# ---- initial paint ----
live_head()
set_status("wait", f"⏳ {T['wait']}")
render_thumbs()
render_snap()
render_recent()
update_dashboard()
update_charts()
render_reports()

# ==========================================
# 9. HARDWARE INIT & AI LOGIC
# ==========================================
def find_arduino_port():
    if ARDUINO_PORT:
        return ARDUINO_PORT
    from serial.tools import list_ports
    keys = ("arduino", "ch340", "ch341", "usb serial", "usb-serial", "cp210", "ftdi")
    for p in list_ports.comports():
        info = f"{p.description} {p.manufacturer}".lower()
        if any(k in info for k in keys):
            return p.device
    return "COM4"


@st.cache_resource
def _open_serial(port):
    ser = serial.Serial(port, 9600, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    return ser


def init_serial():
    try:
        return _open_serial(find_arduino_port())
    except Exception:
        return None


arduino = init_serial()
def open_camera():
    be = {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF, "ANY": cv2.CAP_ANY}[CAMERA_BACKEND]
    order = [CAMERA_INDEX] if CAMERA_INDEX is not None else list(range(6))
    for i in order:
        c = cv2.VideoCapture(i, be)
        if c.isOpened() and c.read()[0]:
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return c, i
        c.release()
    return cv2.VideoCapture(), -1


cap, cam_found = open_camera()
st.sidebar.caption(f"📷 Camera index: {cam_found}" if cam_found >= 0 else "📷 Camera: not found")
cam_ok = cap.isOpened()
st.sidebar.caption(f"🔌 Arduino port: {find_arduino_port()}" + ("" if arduino else "  (not connected)"))
sensor_ui = st.sidebar.empty()
sensor_ui.caption("📡 Sensor: waiting...")
render_health(cam_ok, arduino is not None)


MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_ai():
    class FixedDepthwiseConv2D(keras.layers.DepthwiseConv2D):
        def __init__(self, **kwargs):
            kwargs.pop("groups", None)
            super().__init__(**kwargs)

    mdl = keras.models.load_model(
        os.path.join(MODEL_DIR, "keras_model.h5"), compile=False,
        custom_objects={"DepthwiseConv2D": FixedDepthwiseConv2D})
    names = []
    with open(os.path.join(MODEL_DIR, "labels.txt"), encoding="utf-8-sig") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                names.append(ln.split(" ", 1)[1].strip() if " " in ln else ln)
    sz = mdl.input_shape[1] or 224
    mdl.predict(np.zeros((1, sz, sz, 3), np.float32), verbose=0)
    return mdl, names


try:
    model, class_names = load_ai()
    model_err = None
except Exception as e:
    model, class_names, model_err = None, [], str(e)


def preprocess(frame, size):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    m = min(h, w)
    y0, x0 = (h - m) // 2, (w - m) // 2
    rgb = cv2.resize(rgb[y0:y0 + m, x0:x0 + m], (size, size), interpolation=cv2.INTER_AREA)
    return np.expand_dims(rgb.astype(np.float32) / 127.5 - 1.0, axis=0)


def inspect_image(frame, thr):
    size = model.input_shape[1] or 224
    probs = model.predict(preprocess(frame, size), verbose=0)[0]
    pi = next((i for i, n in enumerate(class_names) if "pass" in n.lower()), 0)
    p_prob = float(probs[pi]) * 100
    is_def = p_prob < thr                      # PASS only if the model is sure it is a pass
    conf = (100.0 - p_prob) if is_def else p_prob

    processed = frame.copy()
    h, w = processed.shape[:2]
    color = (0, 0, 255) if is_def else (0, 200, 0)
    cv2.rectangle(processed, (0, 0), (w - 1, h - 1), color, 8)
    cv2.putText(processed, f"{'DEFECT' if is_def else 'PASS'} {conf:.1f}%", (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    return ("DEFECT" if is_def else "OK"), processed, (1 if is_def else 0), conf


# ==========================================
# 10. CONTINUOUS LOOP
# ==========================================
if model is None:
    live_cam.error(f"⚠️ AI model error: {model_err}")
elif not arduino:
    live_cam.error(f"⚠️ {T['hw_err']}")
elif not cam_ok:
    live_cam.error(f"⚠️ {T['cam_err']}")

scan_line_y = 0
prev_time = time.time()
fps_s = 0.0
frame_i = 0

if cam_ok and arduino and model is not None:
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # FPS (smoothed)
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps_s = (0.9 * fps_s + 0.1 / dt) if fps_s else 1 / dt
            frame_i += 1
            if frame_i % 10 == 0:
                live_head(int(fps_s))

            # Laser scanner overlay
            h, w, _ = frame.shape
            scan_line_y = (scan_line_y + 15) % h
            scan_frame = frame.copy()
            cv2.line(scan_frame, (0, scan_line_y), (w, scan_line_y), (0, 255, 0), 2)
            live_cam.image(cv2.cvtColor(scan_frame, cv2.COLOR_BGR2RGB), channels="RGB",
                               output_format="JPEG", **FIT)

            if arduino.in_waiting > 0:
                line = arduino.readline().decode('utf-8', errors='ignore').strip()
                sensor_ui.caption(f"📡 Sensor: {line or '...'}")

                if "DETECTED" in line.upper():
                    res, img, d_count, conf = inspect_image(frame, conf_th)
                    S.total += 1
                    now_time = datetime.now().strftime("%H:%M:%S")
                    part = f"PRT-{S.total:05d}"
                    passed = (res == "OK")

                    if passed:
                        S.ok += 1
                        S.conf = conf
                        S.history.insert(0, {"Time": now_time, "Part": part, "Status": "✅ PASS", "Defects": 0})
                        set_status("ok", f"✅ {T['pass_msg']}")
                    else:
                        S.defect += 1
                        S.conf = conf
                        S.history.insert(0, {"Time": now_time, "Part": part, "Status": "❌ FAIL", "Defects": d_count})
                        S.fails.insert(0, {"part": part, "time": now_time, "defects": d_count, "jpg": to_jpg(img, 800, 85)})
                        S.fails = S.fails[:100]
                        set_status("bad", f"❌ {T['fail_msg']} ({d_count})")

                    S.recent.insert(0, {"part": part, "time": now_time, "ok": passed,
                                        "defects": d_count, "img": to_b64(img, 200)})
                    S.recent = S.recent[:12]
                    S.last_snap = to_b64(img, 640)
                    S.times.append(time.time())
                    S.yield_data.append(S.ok / S.total * 100)

                    update_dashboard()
                    update_charts()
                    render_thumbs()
                    render_snap()
                    render_recent()
                    render_reports()

                    if passed:
                        toast(f"{part} · {T['pass_msg']}", "✅")
                        beep(2000, 150)
                    else:
                        toast(f"{part} · {T['fail_msg']} ({d_count})", "❌")
                        beep(400, 400)

                    arduino.reset_input_buffer()
                    time.sleep(1.5)

                    S.conf = 100
                    set_status("wait", f"⏳ {T['wait']}")
                    update_dashboard()
    finally:
        cap.release()

    # camera stream ended
    render_health(False, True)