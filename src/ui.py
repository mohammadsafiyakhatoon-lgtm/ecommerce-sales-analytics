"""Visual layer: CSS theme and small HTML components for the Streamlit app.

Only presentation lives here - no data processing or ML logic.
"""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

ACCENT = "#3B5BDB"


def _parse_version(v: str) -> tuple:
    parts = []
    for p in v.split(".")[:2]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


_NEW_WIDTH_API = _parse_version(getattr(st, "__version__", "0.0")) >= (1, 50)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{--bg:#FAFAF8;--card:#FFFFFF;--ink:#1F2328;--muted:#6B7280;--line:#E8E8E4;--accent:#3B5BDB;
--accent-soft:#EEF1FD;--shadow:0 1px 2px rgba(16,24,40,.04),0 4px 14px rgba(16,24,40,.04);}
html,body,[class*="css"],.stApp,button,input,textarea,select{font-family:'Inter',-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif!important;}
.stApp{background:var(--bg);color:var(--ink);}
#MainMenu,footer{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:1240px;padding:2.2rem 2.4rem 4rem 2.4rem;}
h1,h2,h3,h4{color:var(--ink);letter-spacing:-.01em;}
p,li{color:var(--ink);}

/* Sidebar */
[data-testid="stSidebar"]{background:#FFFFFF;border-right:1px solid var(--line);}
[data-testid="stSidebar"] .block-container{padding:1.6rem 1rem;}
.brand{display:flex;align-items:center;gap:.7rem;margin:.2rem 0 1.6rem .3rem;}
.brand-mark{width:34px;height:34px;border-radius:10px;background:var(--accent);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.85rem;letter-spacing:.02em;}
.brand-name{font-weight:650;font-size:.98rem;line-height:1.15;}
.brand-sub{color:var(--muted);font-size:.74rem;font-weight:500;}
.nav-label{color:var(--muted);font-size:.68rem;font-weight:600;letter-spacing:.09em;text-transform:uppercase;margin:0 0 .5rem .45rem;}
[data-testid="stSidebar"] [role="radiogroup"]{gap:.15rem;}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:.55rem .8rem;border-radius:10px;width:100%;cursor:pointer;transition:background .15s;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:#F4F5F7;}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child{display:none;}
[data-testid="stSidebar"] [role="radiogroup"] label p{font-size:.92rem;font-weight:500;color:#374151;margin:0;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:var(--accent-soft);}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p{color:var(--accent);font-weight:600;}
.user-chip{display:flex;align-items:center;gap:.6rem;padding:.7rem .8rem;border:1px solid var(--line);border-radius:12px;margin-bottom:.6rem;}
.user-dot{width:28px;height:28px;border-radius:50%;background:var(--accent-soft);color:var(--accent);display:flex;align-items:center;justify-content:center;font-weight:600;font-size:.78rem;}
.user-name{font-size:.84rem;font-weight:600;line-height:1.1;}
.user-role{font-size:.72rem;color:var(--muted);}

/* Page header */
.eyebrow{color:var(--accent);font-size:.74rem;font-weight:600;letter-spacing:.09em;text-transform:uppercase;margin-bottom:.35rem;}
.page-title{font-size:1.85rem;font-weight:700;letter-spacing:-.02em;margin:0 0 .3rem 0;line-height:1.2;}
.page-sub{color:var(--muted);font-size:.98rem;margin:0 0 1.6rem 0;max-width:760px;line-height:1.55;}
.section{margin:2.1rem 0 .9rem 0;}
.section-title{font-size:1.08rem;font-weight:650;margin:0;}
.section-cap{color:var(--muted);font-size:.86rem;margin:.2rem 0 0 0;}

/* KPI cards */
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(172px,1fr));gap:.9rem;margin-bottom:.4rem;}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:1.05rem 1.15rem;box-shadow:var(--shadow);}
.kpi-label{color:var(--muted);font-size:.76rem;font-weight:550;letter-spacing:.02em;margin-bottom:.45rem;}
.kpi-value{font-size:1.65rem;font-weight:700;letter-spacing:-.02em;line-height:1.1;}
.kpi-sub{color:var(--muted);font-size:.74rem;margin-top:.4rem;}

/* Generic cards */
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:1.3rem 1.4rem;box-shadow:var(--shadow);}
.card h4{margin:0 0 .45rem 0;font-size:.98rem;font-weight:650;}
.card p{color:var(--muted);font-size:.9rem;line-height:1.6;margin:0;}
.hero{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:2.4rem 2.4rem;box-shadow:var(--shadow);margin-bottom:1.4rem;}
.hero .page-title{font-size:2.2rem;margin-bottom:.6rem;}
.hero .page-sub{margin-bottom:0;}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.8rem;}
.step{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:.95rem 1rem;position:relative;}
.step-n{color:var(--accent);font-size:.72rem;font-weight:700;letter-spacing:.08em;}
.step-t{font-size:.9rem;font-weight:650;margin:.15rem 0 .2rem 0;}
.step-d{color:var(--muted);font-size:.78rem;line-height:1.45;}
.insight{display:flex;gap:.8rem;padding:.8rem 0;border-bottom:1px solid var(--line);font-size:.92rem;line-height:1.55;}
.insight:last-child{border-bottom:none;}
.insight-dot{flex:0 0 7px;height:7px;border-radius:50%;background:var(--accent);margin-top:.55rem;}
.pill{display:inline-block;padding:.22rem .65rem;border-radius:999px;font-size:.74rem;font-weight:600;}
.pill-good{background:#E7F5EC;color:#1B6B3A;}
.pill-warn{background:#FDF1E3;color:#9A5B13;}
.pill-neutral{background:#F1F2F4;color:#4B5563;}
.pill-accent{background:var(--accent-soft);color:var(--accent);}
.result{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:1.5rem 1.6rem;box-shadow:var(--shadow);}
.result-label{color:var(--muted);font-size:.78rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;}
.result-value{font-size:2.6rem;font-weight:700;letter-spacing:-.03em;margin:.3rem 0 .6rem 0;line-height:1.1;}
.empty{border:1.5px dashed var(--line);border-radius:16px;padding:3.2rem 1.6rem;text-align:center;color:var(--muted);background:rgba(255,255,255,.6);}
.empty b{display:block;color:var(--ink);font-size:1rem;margin-bottom:.35rem;}

/* Tables */
.tbl-wrap{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:var(--shadow);overflow-x:auto;}
table.tbl{width:100%;border-collapse:collapse;font-size:.88rem;}
table.tbl th{background:#F7F7F5;color:var(--muted);font-weight:600;font-size:.74rem;letter-spacing:.04em;text-transform:uppercase;text-align:left;padding:.7rem 1rem;border-bottom:1px solid var(--line);}
table.tbl td{padding:.75rem 1rem;border-bottom:1px solid #F0F0EC;}
table.tbl tr:last-child td{border-bottom:none;}
table.tbl tr.best td{background:var(--accent-soft);font-weight:600;}

/* Streamlit widgets */
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:14px;}
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] .filter-marker){background:var(--card);box-shadow:var(--shadow);}
.stButton>button,[data-testid="stFormSubmitButton"]>button{width:100%;border-radius:10px;border:1px solid var(--line);background:#fff;color:var(--ink);font-weight:550;padding:.55rem 1rem;transition:all .15s;}
.stButton>button:hover,[data-testid="stFormSubmitButton"]>button:hover{border-color:var(--accent);color:var(--accent);}
.stButton>button[kind="primary"],[data-testid="stFormSubmitButton"]>button[kind="primaryFormSubmit"],[data-testid="stFormSubmitButton"]>button[kind="primary"]{background:var(--accent);border-color:var(--accent);color:#fff;}
.stButton>button[kind="primary"]:hover,[data-testid="stFormSubmitButton"]>button[kind="primaryFormSubmit"]:hover,[data-testid="stFormSubmitButton"]>button[kind="primary"]:hover{background:#2F4AC0;border-color:#2F4AC0;color:#fff;}
div[data-baseweb="select"]>div,div[data-baseweb="input"],div[data-baseweb="base-input"],.stNumberInput input,.stTextInput input,.stDateInput input{border-radius:10px!important;}
label[data-testid="stWidgetLabel"] p{font-size:.82rem;font-weight:550;color:#374151;}
.stTabs [data-baseweb="tab-list"]{gap:.4rem;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{padding:.55rem .9rem;font-weight:550;color:var(--muted);}
.stTabs [aria-selected="true"]{color:var(--accent);}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden;}
[data-testid="stPlotlyChart"]{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:.4rem .5rem;box-shadow:var(--shadow);}
hr{border-color:var(--line);}

/* Login */
.login-brand{padding:2.2rem 1rem 0 0;}
.login-brand h1{font-size:2.5rem;font-weight:700;letter-spacing:-.03em;line-height:1.15;margin:1.4rem 0 .9rem 0;}
.login-brand p{color:var(--muted);font-size:1.02rem;line-height:1.6;max-width:430px;}
.login-stats{display:flex;gap:2rem;margin-top:2.2rem;}
.login-stat b{display:block;font-size:1.35rem;font-weight:700;letter-spacing:-.02em;}
.login-stat span{color:var(--muted);font-size:.78rem;}
.login-hint{color:var(--muted);font-size:.8rem;margin-top:.9rem;}

@media (max-width:768px){.block-container{padding:1.2rem 1rem 3rem 1rem;}.page-title{font-size:1.5rem;}.hero{padding:1.5rem;}.hero .page-title{font-size:1.7rem;}.login-brand h1{font-size:1.9rem;}}
</style>
"""

HIDE_SIDEBAR_CSS = """
<style>
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"]{display:none!important;}
</style>
"""


def inject_css(show_sidebar: bool = True) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    if not show_sidebar:
        st.markdown(HIDE_SIDEBAR_CSS, unsafe_allow_html=True)


def _e(x) -> str:
    return html.escape(str(x))


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
def brand() -> None:
    st.markdown(
        '<div class="brand"><div class="brand-mark">EA</div><div>'
        '<div class="brand-name">E-Commerce Analytics</div>'
        '<div class="brand-sub">Sales &amp; profit intelligence</div></div></div>',
        unsafe_allow_html=True)


def nav_label(text: str) -> None:
    st.markdown(f'<div class="nav-label">{_e(text)}</div>', unsafe_allow_html=True)


def user_chip(name: str) -> None:
    initial = (name[:1] or "U").upper()
    st.markdown(
        f'<div class="user-chip"><div class="user-dot">{_e(initial)}</div><div>'
        f'<div class="user-name">{_e(name)}</div><div class="user-role">Signed in</div></div></div>',
        unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    parts = []
    if eyebrow:
        parts.append(f'<div class="eyebrow">{_e(eyebrow)}</div>')
    parts.append(f'<h1 class="page-title">{_e(title)}</h1>')
    if subtitle:
        parts.append(f'<p class="page-sub">{_e(subtitle)}</p>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def section(title: str, caption: str = "") -> None:
    cap = f'<p class="section-cap">{_e(caption)}</p>' if caption else ""
    st.markdown(f'<div class="section"><h3 class="section-title">{_e(title)}</h3>{cap}</div>',
                unsafe_allow_html=True)


def kpi_grid(items: list[tuple]) -> None:
    """items: (label, value, optional sub-text)."""
    cards = []
    for it in items:
        label, value = it[0], it[1]
        sub = it[2] if len(it) > 2 and it[2] else ""
        sub_html = f'<div class="kpi-sub">{_e(sub)}</div>' if sub else ""
        cards.append(f'<div class="kpi"><div class="kpi-label">{_e(label)}</div>'
                     f'<div class="kpi-value">{_e(value)}</div>{sub_html}</div>')
    st.markdown('<div class="kpi-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def info_card(title: str, body: str) -> None:
    st.markdown(f'<div class="card"><h4>{_e(title)}</h4><p>{_e(body)}</p></div>', unsafe_allow_html=True)


def hero(title: str, subtitle: str, eyebrow: str = "") -> None:
    eb = f'<div class="eyebrow">{_e(eyebrow)}</div>' if eyebrow else ""
    st.markdown(f'<div class="hero">{eb}<h1 class="page-title">{_e(title)}</h1>'
                f'<p class="page-sub">{_e(subtitle)}</p></div>', unsafe_allow_html=True)


def steps(items: list[tuple]) -> None:
    """items: (title, description)."""
    cells = []
    for i, (t, d) in enumerate(items, 1):
        cells.append(f'<div class="step"><div class="step-n">STEP {i:02d}</div>'
                     f'<div class="step-t">{_e(t)}</div><div class="step-d">{_e(d)}</div></div>')
    st.markdown('<div class="steps">' + "".join(cells) + "</div>", unsafe_allow_html=True)


def insight_list(items: list[str]) -> None:
    rows = "".join(f'<div class="insight"><div class="insight-dot"></div><div>{_e(t)}</div></div>'
                   for t in items)
    st.markdown(f'<div class="card" style="padding:.6rem 1.4rem">{rows}</div>', unsafe_allow_html=True)


def pill(text: str, tone: str = "neutral") -> str:
    return f'<span class="pill pill-{tone}">{_e(text)}</span>'


def html_table(df: pd.DataFrame, highlight_col: str | None = None, highlight_value=None,
               formats: dict | None = None) -> None:
    formats = formats or {}
    head = "".join(f"<th>{_e(c)}</th>" for c in df.columns)
    body = []
    for _, row in df.iterrows():
        cls = ' class="best"' if highlight_col and row.get(highlight_col) == highlight_value else ""
        cells = []
        for c in df.columns:
            v = row[c]
            cells.append(f"<td>{_e(formats[c](v) if c in formats else v)}</td>")
        body.append(f"<tr{cls}>" + "".join(cells) + "</tr>")
    st.markdown(f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead>'
                f'<tbody>{"".join(body)}</tbody></table></div>', unsafe_allow_html=True)


def empty_state(title: str, body: str) -> None:
    st.markdown(f'<div class="empty"><b>{_e(title)}</b>{_e(body)}</div>', unsafe_allow_html=True)


def show(fig, key: str | None = None) -> None:
    """Render a Plotly figure full-width (compatible with old and new Streamlit APIs)."""
    config = {"displayModeBar": False}
    if _NEW_WIDTH_API:
        st.plotly_chart(fig, width="stretch", key=key, config=config)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key, config=config)


def show_df(df: pd.DataFrame, height: int | None = None) -> None:
    kwargs = {"hide_index": True}
    if height:
        kwargs["height"] = height
    if _NEW_WIDTH_API:
        st.dataframe(df, width="stretch", **kwargs)
    else:
        st.dataframe(df, use_container_width=True, **kwargs)
