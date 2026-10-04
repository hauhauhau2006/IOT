from pathlib import Path

import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))

import html, io, time

import numpy as np

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

import streamlit as st

from check_env import environment_status, environment_issues

from engine import (MODEL_FILES, LABELS, load_artifact, feature_names, prepare_frame, infer,

                    evaluate, demo_session, attack_mask, pdf_report)



ROOT=Path(__file__).parent

st.set_page_config(page_title='IoT Sentinel · Phát hiện tấn công IoT',page_icon='🛡️',layout='wide',initial_sidebar_state='expanded')

COLORS=['#0F766E','#C7475D','#7C3AED','#B45309','#047857','#0369A1','#BE185D']

CSS='''

<style>

:root{--cyan:#0F766E;--muted:#52627A;--panel:#fff;--line:#e2e8f0}

.stApp{background:#F3F6FC;color:#172B4D}

[data-testid="stHeader"]{background:#F3F6FCcc}

.block-container{padding-top:2rem;padding-bottom:2.5rem;max-width:1560px}

h1,h2,h3{color:#142747!important;letter-spacing:-.035em}h1{font-size:2rem!important;font-weight:750!important}h2{font-size:1.4rem!important}h3{font-size:1.05rem!important}

[data-testid="stCaptionContainer"]{color:#52627A}

[data-testid="stVerticalBlockBorderWrapper"]>div{background:#fff;border:1px solid #e3eaf3!important;border-radius:18px!important;box-shadow:0 4px 20px #142c4805}

[data-testid="stMetric"]{padding:22px;border:1px solid #e3eaf3;border-radius:16px;background:#fff;min-height:126px;box-shadow:0 5px 20px #142c4805;border-top:3px solid #0F766E}

[data-testid="stMetricLabel"]{color:#52627A;font-size:.86rem}[data-testid="stMetricValue"]{color:#142747;font-size:2rem;font-weight:750}

.stButton button,.stDownloadButton button{border-radius:10px;border:1px solid #d7e2ee;min-height:44px;font-weight:600;background:#fff;color:#263f5c}

.stButton button[kind="primary"],.stDownloadButton button[kind="primary"]{background:#0F766E;color:#fff;border:0;box-shadow:0 4px 12px #0F766E22}

.stButton button:hover,.stDownloadButton button:hover{border-color:#0F766E;color:#0F766E}.stButton button[kind="primary"]:hover{background:#115E59;color:#fff}

[data-testid="stSidebar"]{background:linear-gradient(165deg,#122b47,#0b192d);border-right:0}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"], [data-testid="stSidebar"] [data-testid="stCaptionContainer"], [data-testid="stSidebar"] label{color:#c2d1e3!important}

[data-testid="stSidebar"] .stRadio label{padding:11px 13px!important;border-radius:10px;cursor:pointer;margin:3px 0}

[data-testid="stSidebar"] .stRadio label:has(input:checked){background:#214560;border-left:3px solid #46d2dd;color:#fff!important}

[data-testid="stSidebar"] .stButton button{background:#1c3651;border-color:#36516c;color:#dce8f5}

.brand{display:flex;gap:12px;align-items:center;padding:4px 0 28px}.shield{width:42px;height:48px;color:#54d4df}.brand b{font-size:1.32rem;letter-spacing:.025em;color:white}.brand b span{color:#54d4df}.brand small{display:block;color:#BCCCE3;font-size:.78rem;margin-top:6px}

.badge{display:inline-block;border:1px solid #c2e8e9;background:#edfafa;color:#087f87;border-radius:24px;padding:6px 13px;font-size:.76rem;font-weight:650;letter-spacing:.035em}.badge.demo{color:#9a6a17;background:#fff8e8;border-color:#f2dfb3}

.model-card{padding:8px 3px}.model-card .name{font-size:1.05rem;font-weight:650;color:#163450}.model-card .desc{color:#52627A;font-size:.86rem;margin-top:8px}

.pill{display:inline-block;border-radius:20px;padding:5px 11px;font-size:.76rem;margin-top:12px;background:#e7f8ef;color:#20764e}.pill.wait{background:#edf1f7;color:#52627A}

.step{display:flex;align-items:center;gap:12px;padding:15px;border:1px solid #e0e8f2;border-radius:12px;background:white;color:#49627e;font-size:.88rem}.step b{background:#e5f5f8;color:#07839c;border-radius:9px;width:30px;height:30px;display:flex;align-items:center;justify-content:center;flex-shrink:0}

.footer{border-top:1px solid #dde6f0;padding-top:18px;margin-top:32px;display:flex;justify-content:space-between;gap:12px;color:#52627A;font-size:.76rem}

[data-testid="stDataFrame"]{border:1px solid #e2e8f0;border-radius:12px;overflow:hidden}

[data-testid="stFileUploader"] section{background:#f8fbff;border:1px dashed #b8cadc;border-radius:12px}

[data-testid="stAlert"]{border-radius:12px;font-size:.9rem}

[data-testid="stExpander"]{border:1px solid #e0e8f2;border-radius:12px;background:#fff}

[data-baseweb="tab-list"]{gap:18px;border-bottom:1px solid #dce6f0}[data-baseweb="tab"]{font-weight:600}

.sidebar-session{background:#18344f;border:1px solid #2b4865;border-radius:12px;padding:14px;margin:8px 0 14px}.sidebar-session strong{display:block;color:#fff;font-size:.88rem;overflow-wrap:anywhere}.sidebar-session small{display:block;color:#BCCCE3;margin-top:6px}

@media(max-width:700px){.block-container{padding:1rem}h1{font-size:1.5rem!important}.footer{display:block}[data-testid="stMetric"]{padding:14px;min-height:108px}}

@media print{[data-testid="stSidebar"],[data-testid="stHeader"]{display:none!important}.block-container{padding:0!important}.stApp{background:#fff!important}}


/* Consistent light controls and readable text, including dark OS preferences. */
.stApp{color-scheme:light;font-family:"Segoe UI",Arial,sans-serif}
[data-testid="stMain"]{color:#172B4D;font-size:16px;line-height:1.6}
[data-testid="stMain"] [data-testid="stWidgetLabel"] p{color:#263B59;font-weight:600;font-size:.95rem}
[data-testid="stCaptionContainer"] p{font-size:.88rem;line-height:1.55}
[data-testid="stMain"] input,[data-testid="stMain"] textarea{background:#fff!important;color:#172B4D!important;caret-color:#0F766E}
input::placeholder,textarea::placeholder{color:#607089!important;opacity:1}
[data-baseweb="input"],[data-baseweb="base-input"],[data-baseweb="textarea"],
[data-baseweb="select"]>div{background:#fff!important;color:#172B4D!important;border-color:#B8C6DC!important;border-radius:10px}
[data-baseweb="select"] svg{fill:#425674}
[data-baseweb="popover"],[data-baseweb="menu"], [role="listbox"]{background:#fff!important;color:#172B4D!important}
[role="option"]{background:#fff!important;color:#172B4D!important}
[role="option"]:hover,[role="option"][aria-selected="true"]{background:#E6F3EF!important;color:#115E59!important}
[data-baseweb="tab"]{color:#52627A!important;padding:12px 4px}
[data-baseweb="tab"][aria-selected="true"]{color:#0F766E!important}
[data-baseweb="tab-highlight"]{background:#0F766E!important}
[data-testid="stFileUploaderDropzone"]{background:#F7F9FE!important;color:#263B59!important}
[data-testid="stFileUploaderDropzone"] small{color:#52627A!important}
[data-testid="stExpander"] summary{color:#263B59!important}
.stButton button:hover,.stDownloadButton button:hover{background:#EDF7F3;border-color:#0F766E;color:#0F766E}
.stButton button[kind="primary"],.stDownloadButton button[kind="primary"]{background:#0F766E;color:white}
.stButton button[kind="primary"]:hover,.stDownloadButton button[kind="primary"]:hover{background:#115E59;color:white}
button:focus-visible,input:focus-visible,textarea:focus-visible{outline:3px solid #38B2A0!important;outline-offset:3px}
.stButton button:disabled,.stDownloadButton button:disabled{background:#E8EDF5!important;color:#62718A!important;border-color:#CBD5E1!important;opacity:1}
[data-testid="stSidebar"]{background:linear-gradient(165deg,#172D50 0%,#101D35 100%)}
[data-testid="stSidebar"] .stRadio label{color:#DCE6F5!important}
[data-testid="stSidebar"] .stRadio label:has(input:checked){background:#244574;border-left:3px solid #7DD3C0}
[data-testid="stSidebar"] .stRadio label:has(input:checked) p{color:#fff!important;font-weight:650}
[data-testid="stSidebar"] .stButton button:hover{background:#294A76;color:white;border-color:#7DD3C0}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#F1F5FF!important}
[data-testid="stSidebar"] hr{border-color:#3B506F}
.brand b span,.shield{color:#7DD3C0}
.badge{color:#115E59;background:#E6F3EF;border-color:#B9DCD1}
.badge.demo{color:#85520A;background:#FFF6DF;border-color:#E9D39A}
.step b{background:#E6F3EF;color:#0F766E}
.step{color:#354B69}
.pill{color:#16643D;background:#E7F6EE}.pill.wait{color:#4E6079;background:#EDF1F7}
[data-testid="stAlert"]{color:#172B4D}


/* Main content: explicit foregrounds override inherited Streamlit dark colors. */
[data-testid="stAppViewContainer"],.stApp{background:#F4F6F5!important}
[data-testid="stHeader"]{background:#F4F6F5!important}
[data-testid="stMain"],section.main{background:#F4F6F5!important;color:#000000!important}
:is([data-testid="stMain"],section.main) [data-testid="stMarkdownContainer"]{color:#172B4D!important}
:is([data-testid="stMain"],section.main) [data-testid="stCaptionContainer"],
:is([data-testid="stMain"],section.main) [data-testid="stCaptionContainer"] p{color:#465872!important;opacity:1!important}
:is([data-testid="stMain"],section.main) [data-testid="stWidgetLabel"],
:is([data-testid="stMain"],section.main) [data-testid="stWidgetLabel"] p{color:#243B5A!important}
:is([data-testid="stMain"],section.main) [data-testid="stMetricLabel"] p{color:#465872!important}
:is([data-testid="stMain"],section.main) [data-testid="stMetricValue"]{color:#142747!important}
:is([data-testid="stMain"],section.main) [data-testid="stFileUploaderDropzone"]{background:#FFFFFF!important;border:1.5px dashed #8297B5!important;border-radius:12px!important;color:#243B5A!important}
:is([data-testid="stMain"],section.main) [data-testid="stFileUploaderDropzoneInstructions"] :is(span,p,small),
:is([data-testid="stMain"],section.main) [data-testid="stFileUploaderDropzone"] :is(span,p,small){color:#354B69!important;opacity:1!important;-webkit-text-fill-color:#354B69!important}
:is([data-testid="stMain"],section.main) [data-testid="stFileUploaderDropzone"] svg{color:#0F766E!important;fill:currentColor!important}
:is([data-testid="stMain"],section.main) [data-testid="stFileUploader"] button{background:#0F766E!important;border:1px solid #0F766E!important;color:#FFFFFF!important;border-radius:9px!important;font-weight:600!important;min-height:44px}
:is([data-testid="stMain"],section.main) [data-testid="stFileUploader"] button :is(span,p,div){color:#FFFFFF!important;-webkit-text-fill-color:#FFFFFF!important}
:is([data-testid="stMain"],section.main) [data-testid="stFileUploader"] button:hover{background:#115E59!important;border-color:#115E59!important}
:is([data-testid="stMain"],section.main) [data-testid="stFileUploader"] button svg{color:#FFFFFF!important}
:is([data-testid="stMain"],section.main) [data-testid="stFileUploaderFileName"]{color:#172B4D!important}
:is([data-testid="stMain"],section.main) button [data-testid="stMarkdownContainer"]{color:inherit!important}
:is([data-testid="stMain"],section.main) [data-testid="stExpander"] summary{background:#FFFFFF!important;color:#243B5A!important}

:is([data-testid="stMain"],section.main) [data-testid="stAlert"] :is(p,li,span){color:#243B5A!important}
:is([data-testid="stMain"],section.main) :is(.stRadio,.stCheckbox,.stToggle) label p{color:#243B5A!important}
@media print{[data-testid="stMain"],section.main,.stApp{background:#FFFFFF!important}}


/* Sentinel editorial dashboard finish. */
.block-container{max-width:1480px;padding:3.2rem 2.6rem 3rem}
h1,h2,h3{font-family:"Segoe UI",Arial,sans-serif;letter-spacing:-.025em;line-height:1.25}
h1{font-size:2.25rem!important;font-weight:750!important}
h2{font-size:1.5rem!important}h3{font-size:1.1rem!important;font-weight:650!important}
[data-testid="stMain"] [data-testid="stVerticalBlock"]{gap:1.15rem}
[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"]>div,
[data-testid="stMain"] [data-testid="stVerticalBlock"][data-border="true"],
[data-testid="stMain"] [data-testid="stLayoutWrapper"]:has(>.stVerticalBlock[data-border="true"]){border-color:#DDE5E2!important;border-radius:18px!important}
[data-testid="stMetric"]{background:linear-gradient(135deg,#FFFFFF,#FAFCFB);border:1px solid #DDE5E2;border-top:3px solid #0F766E;border-radius:17px;padding:22px 24px;box-shadow:0 5px 18px #18372D06}
[data-testid="stMetricValue"]{font-variant-numeric:tabular-nums;letter-spacing:-.045em;font-size:2.25rem!important}
[data-testid="stMetricLabel"] p{font-weight:600!important;font-size:.88rem!important}
.stButton button,.stDownloadButton button{min-height:46px;border-radius:11px;transition:background .15s,border-color .15s,box-shadow .15s}
.stButton button[kind="primary"],.stDownloadButton button[kind="primary"]{box-shadow:0 4px 12px #0F766E20}
[data-testid="stMain"] [data-testid="stFileUploaderDropzone"]{padding:20px!important;border-color:#98B9AE!important;background:#FCFEFD!important}
[data-testid="stMain"] [data-testid="stFileUploaderDropzone"]:hover{border-color:#0F766E!important;background:#F1F9F5!important}
[data-testid="stMain"] [data-testid="stFileUploaderDropzone"] small{font-size:.8rem!important}
[data-testid="stMain"] [data-testid="stExpander"]{border-color:#DDE5E2;background:white;border-radius:12px;overflow:hidden}
[data-testid="stSidebar"]{background:linear-gradient(175deg,#142B39,#11212D 70%,#12362F)!important;border-right:1px solid #213D47}
[data-testid="stSidebarUserContent"]{padding-top:2.1rem}
[data-testid="stSidebar"] .stRadio label{padding:12px 14px!important;border:1px solid transparent;border-left:3px solid transparent;transition:background .15s}
[data-testid="stSidebar"] .stRadio label:hover{background:#213D49}
[data-testid="stSidebar"] .stRadio label:has(input:checked){background:#244B4B!important;border-color:#325E5A;border-left:3px solid #7DD3C0;box-shadow:0 4px 12px #00000012}
[data-testid="stSidebar"] .stButton button{background:#1C3741;border-color:#36545B;color:#E5F2ED}
[data-testid="stSidebar"] .stButton button:hover{background:#285047;border-color:#7DD3C0}
.brand{padding-bottom:26px}.brand b{font-size:1.25rem;letter-spacing:.055em}.brand small{font-size:.8rem;color:#BACFD2}
.brand b span,.shield{color:#7DD3C0}
.sidebar-session{background:#1C3942;border-color:#34535C;border-radius:13px;padding:16px}
.sidebar-session small{color:#BACFD2}
.badge{font-size:.73rem;letter-spacing:.045em;background:#E6F3EF;color:#115E59;border-color:#B9DCD1;padding:7px 12px}
.badge.demo{background:#FFF5E5;color:#875412;border-color:#EAD4AA}
.model-card{padding:10px 6px}.model-card .name{font-size:1.08rem;color:#183C47}.model-card .desc{color:#4B6268;line-height:1.5}
.pill{font-size:.78rem;padding:6px 12px}.pill.wait{color:#4B6268;background:#EEF2F1}
.step{padding:17px;background:#FFFFFF;border-color:#DDE5E2;border-radius:13px;color:#344E57;line-height:1.5}
.step b{width:32px;height:32px;background:#E6F3EF;color:#115E59;font-size:.86rem;border-radius:10px}
.footer{margin-top:40px;padding-top:22px;border-color:#D9E2DE;color:#52676B;font-size:.78rem;line-height:1.6}
.page-intro{border-left:4px solid #0F766E;padding:2px 0 4px 20px;margin-bottom:5px}
.page-intro .eyebrow{font-size:.69rem;font-weight:750;letter-spacing:.15em;color:#0F766E;margin-bottom:9px}
.page-intro h1{margin:0!important;padding:0!important;color:#142D3D!important}
.page-intro p{margin:10px 0 0!important;color:#4A606B!important;font-size:.95rem;line-height:1.65;max-width:850px}
[data-baseweb="tab-list"]{gap:22px;border-color:#DDE5E2}
[data-baseweb="tab"]{padding:12px 2px;font-size:.93rem}
[data-testid="stMain"] [data-testid="stAlert"]{border-radius:12px}
[data-testid="stMain"] [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]){background:#E6F4EC!important;border:1px solid #B5DCC6}
[data-testid="stMain"] [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]){background:#FFF5DE!important;border:1px solid #E9D49E}
[data-testid="stMain"] [data-testid="stAlert"]:has([data-testid="stAlertContentError"]){background:#FDECEF!important;border:1px solid #ECC1C9}
[data-testid="stMain"] [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]){background:#EAF2F8!important;border:1px solid #C2D6E5}
@media(max-width:800px){.block-container{padding:2rem 1rem}.page-intro{padding-left:14px}.page-intro h1{font-size:1.75rem!important}.page-intro p{font-size:.88rem}[data-testid="stMetric"]{padding:16px;min-height:108px}.footer{display:block}}
@media(prefers-reduced-motion:reduce){button,[data-testid="stSidebar"] .stRadio label{transition:none!important}}


/* Keep selected model labels fully visible and remove the inner caret outline. */
[data-testid="stMultiSelect"] [data-baseweb="select"]>div{padding-left:10px!important;padding-right:8px!important;overflow:visible!important}
[data-testid="stMultiSelect"] [data-baseweb="select"]>div>div:first-child{display:flex!important;flex-wrap:wrap!important;gap:6px!important;padding:6px 0!important;overflow:visible!important;min-width:0!important}
[data-testid="stMultiSelect"] [data-baseweb="tag"]{flex-shrink:0!important;max-width:100%!important;margin:0!important;padding:7px 10px!important;height:auto!important;border-radius:8px!important;background:#E6F3EF!important;color:#115E59!important;overflow:visible!important}
[data-testid="stMultiSelect"] [data-baseweb="tag"] span{max-width:none!important;text-overflow:clip!important;overflow:visible!important;white-space:normal!important;color:#115E59!important;-webkit-text-fill-color:#115E59!important}
[data-testid="stMultiSelect"] [data-baseweb="tag"] svg{color:#115E59!important}
[data-testid="stMain"] [data-baseweb="select"] input{caret-color:transparent!important}
[data-testid="stMain"] [data-baseweb="select"] input:focus-visible{outline:none!important;box-shadow:none!important}
[data-testid="stMultiSelect"] [data-baseweb="select"]:focus-within>div{border-color:#0F766E!important;box-shadow:0 0 0 2px #0F766E20!important}

</style>'''

st.markdown(CSS,unsafe_allow_html=True)



def init():

    defaults={'session':None,'models':{},'artifact_warnings':{},'scaler':None,'encoders':None,

        'confirmed':False,'encoded':False,'unknown':'block','scaled':{n:True for n in MODEL_FILES},

        'replay_pos':0,'replay_play':False,'replay_key':None,'manifest':None,'page':'Tổng quan','staged_files':{},'loaded_csv':None,'load_notices':{}}

    for k,v in defaults.items():

        if k not in st.session_state:st.session_state[k]=v

    if not st.session_state.get('autoload_done'):

        for name,filename in MODEL_FILES.items():

            p=ROOT/'models'/filename

            if p.exists():

                try:

                    obj,notes=load_artifact(p);st.session_state.models[name]=obj;st.session_state.artifact_warnings[name]=notes

                except Exception as e:st.session_state.artifact_warnings[name]=[str(e)]

        for key,filename in [('scaler','scaler.pkl'),('encoders','label_encoders.pkl')]:

            p=ROOT/'models'/filename

            if p.exists():

                try:obj,notes=load_artifact(p);st.session_state[key]=obj;st.session_state.artifact_warnings[key]=notes

                except Exception as e:st.session_state.artifact_warnings[key]=[str(e)]

        st.session_state.autoload_done=True

init()



# Detach ordinary widget values from Streamlit's page cleanup.

# Uploaders/buttons cannot be assigned; their bytes are kept separately.

UPLOAD_KEYS={'traffic_csv','upload_scaler','upload_encoders',*MODEL_FILES.values()}

for _key in list(st.session_state):

    if _key not in UPLOAD_KEYS and _key != 'nav_page' and not _key.startswith(('load_', 'FormSubmitter:')):

        st.session_state[_key]=st.session_state[_key]





def staged_upload(label, types, key):

    uploaded=st.file_uploader(label,type=types,key=key)

    if uploaded is not None:

        st.session_state.staged_files[key]={'name':uploaded.name,'bytes':uploaded.getvalue()}

    saved=st.session_state.staged_files.get(key)

    if saved:

        st.caption('Đã chọn: '+saved['name'])

        source=io.BytesIO(saved['bytes']);source.name=saved['name']

        return source

    return None





def load_notice(key):

    if key in st.session_state.load_notices:

        st.success(st.session_state.load_notices[key])





def session_notes(s):

    if s.get('notes'):

        with st.expander('Chi tiết xử lý dữ liệu'):

            for note in s['notes']:st.caption(note)



def go_page(page):st.session_state.page=page;st.rerun()



def nav_changed():st.session_state.page=st.session_state.nav_page



def plot(fig,height=300):

    fig.update_layout(template='plotly_white',paper_bgcolor='#ffffff',plot_bgcolor='#ffffff',

        font=dict(family='Segoe UI, Arial, sans-serif',size=14,color='#263B59'),margin=dict(l=15,r=15,t=20,b=20),height=height,

        legend=dict(orientation='h',y=-.18,x=0),hoverlabel=dict(bgcolor='#ffffff',font_color='#172B4D',bordercolor='#B8C6DC'),colorway=COLORS)

    fig.update_xaxes(gridcolor='#E4EAF3',zeroline=True);fig.update_yaxes(gridcolor='#E4EAF3',zeroline=True)

    fig.update_xaxes(
    tickfont=dict(size=15, color='#172B45'),
    title_font=dict(size=16, color='#172B45')
)

    fig.update_yaxes(
    tickfont=dict(size=14, color='#172B45'),
    title_font=dict(size=16, color='#172B45')
)

    fig.update_layout(
    legend=dict(
        font=dict(size=15, color='#172B45'),
        title_font=dict(size=15, color='#172B45')
    )
)

    st.plotly_chart(fig,width='stretch',config={'displaylogo':False,'modeBarButtonsToRemove':['lasso2d','select2d']})



def banner():

    s=st.session_state.session

    if s and s['demo']:

        st.markdown('<span class="badge demo">DỮ LIỆU MINH HỌA · Kết quả tổng hợp</span>',unsafe_allow_html=True)

    elif s:

        st.markdown('<span class="badge">KẾT QUẢ TỪ CSV · '+html.escape(s['filename'])+'</span>',unsafe_allow_html=True)

    else:

        st.markdown('<span class="badge">SẴN SÀNG PHÂN TÍCH CSV</span>',unsafe_allow_html=True)



def header(title,subtitle):

    left,right=st.columns([4,1])

    with left:
        st.markdown('<div class="page-intro"><div class="eyebrow">IOT SENTINEL / PHÂN TÍCH AN NINH MẠNG</div><h1>'+html.escape(title)+'</h1><p>'+html.escape(subtitle)+'</p></div>',unsafe_allow_html=True)

    with right:

        if st.button('Xuất báo cáo',width='stretch',disabled=st.session_state.session is None):go_page('Báo cáo')

    banner();st.write('')



def selected_model(s,key):

    if st.session_state.get(key) not in s['results']:st.session_state[key]=next(iter(s['results']))

    return st.selectbox('Mô hình đang xem',list(s['results']),key=key)



def result_df(s,name):

    r=s['results'][name];src=s['df']

    d=pd.DataFrame({'Bản ghi':np.arange(1,len(src)+1)})

    for c,label in [('ip.src_host','IP nguồn'),('ip.dst_host','IP đích'),('tcp.dstport','Cổng đích')]:

        if c in src:d[label]=src[c].astype(str).to_numpy()

    d['Dự đoán']=r['pred'].astype(str)

    d['Trạng thái']=np.where(attack_mask(r['pred']),'Nghi tấn công','Bình thường')

    if r['score'] is not None:d['Xác suất nhãn dự đoán']=r['score']

    if 'Attack_type' in src:d['Nhãn thật']=src['Attack_type'].astype(str).to_numpy()

    d['Mô hình']=name

    return d



def table(s,name,key='table'):

    d=result_df(s,name)

    a,b,c=st.columns([2,1,1])

    search=a.text_input('Tìm bản ghi / IP / nhãn',placeholder='Tìm kiếm...',key=key+'search')

    state=b.selectbox('Trạng thái',['Tất cả','Nghi tấn công','Bình thường'],key=key+'state')

    if st.session_state.get(key+'label','Tất cả') not in ['Tất cả']+sorted(d['Dự đoán'].unique()):st.session_state[key+'label']='Tất cả'

    label=c.selectbox('Nhãn dự đoán',['Tất cả']+sorted(d['Dự đoán'].unique()),key=key+'label')

    if search:

        keep=d.astype(str).apply(lambda x:x.str.contains(search,case=False,regex=False)).any(axis=1);d=d[keep]

    if state!='Tất cả':d=d[d['Trạng thái']==state]

    if label!='Tất cả':d=d[d['Dự đoán']==label]

    n_pages=max(1,(len(d)+49)//50)

    if key+'page' in st.session_state:st.session_state[key+'page']=min(n_pages,max(1,st.session_state[key+'page']))

    p=st.number_input('Trang · 50 bản ghi/trang',min_value=1,max_value=n_pages,value=1,key=key+'page')

    shown=d.iloc[(p-1)*50:p*50]

    styled=shown.style.map(lambda v: 'color:#20764e;background-color:#e7f8ef' if v=='Bình thường' else 'color:#A51F37;background-color:#FFF0F3',subset=['Trạng thái'])

    st.dataframe(styled,hide_index=True,width='stretch',column_config={'Xác suất nhãn dự đoán':st.column_config.NumberColumn(format='%.3f')})

    st.caption(f'{len(d):,} bản ghi phù hợp · Trang {p}/{n_pages}')

    if len(shown):

        with st.expander('Xem chi tiết một bản ghi'):

            if st.session_state.get(key+'detail') not in shown['Bản ghi'].tolist():st.session_state[key+'detail']=int(shown['Bản ghi'].iloc[0])

            idx=st.selectbox('Mã bản ghi',shown['Bản ghi'].tolist(),key=key+'detail')-1

            st.write('**Đầu vào gốc**');st.dataframe(s['df'].iloc[[idx]].astype(str).T.rename(columns={s['df'].index[idx]:'Giá trị'}),width='stretch')

            st.write('**Dự đoán của từng mô hình**')

            st.dataframe(pd.DataFrame([{'Mô hình':n,'Dự đoán':str(r['pred'][idx]),'Xác suất':float(r['score'][idx]) if r['score'] is not None else None} for n,r in s['results'].items()]),hide_index=True,width='stretch')

            st.caption('Xác suất là đầu ra của model, chưa phải độ tin cậy đã hiệu chỉnh. Nhãn dự đoán không tự giải thích nguyên nhân tấn công.')



def status_cards():

    for n in MODEL_FILES:

        obj=st.session_state.models.get(n)

        with st.container(border=True):

            good=obj is not None and hasattr(obj,'predict')

            text='Đã tải · '+str(getattr(obj,'n_features_in_','?'))+' đặc trưng' if good else 'Chưa tải model'

            st.markdown(f'<div class="model-card"><div class="name">{html.escape(n)}</div><div class="desc">{html.escape(text)}</div><span class="pill {"" if good else "wait"}">{("Chờ kiểm tra dữ liệu" if st.session_state.confirmed else "Chờ xác nhận pipeline") if good else "Chưa sẵn sàng"}</span></div>',unsafe_allow_html=True)

    if st.button('Cấu hình mô hình',width='stretch'):go_page('Cấu hình')



def overview():

    header('Tổng quan phân tích','Hệ thống phát hiện tấn công IoT · Random Forest / Decision Tree / SVM')

    s=st.session_state.session

    if not s:

        cols=st.columns(4)

        for col,title in zip(cols,['Bản ghi đã phân tích','Bình thường','Nghi tấn công','Thời gian xử lý']):col.metric(title,'—')

        left,right=st.columns([2.5,1])

        with left:

            with st.container(border=True):

                st.subheader('Bắt đầu phiên phân tích')

                st.write('Tải CSV, kiểm tra đặc trưng và chọn mô hình để xem kết quả phát hiện tấn công.')

                a,b=st.columns(2)

                if a.button('Phân tích CSV',type='primary',width='stretch'):go_page('Phân tích dữ liệu')

                if b.button('Xem giao diện với dữ liệu minh họa',width='stretch'):

                    st.session_state.session=demo_session();st.rerun()

                st.caption('Dữ liệu minh họa dùng để trải nghiệm dashboard.')

            with st.container(border=True):

                st.subheader('Quy trình xử lý')

                for i,text in enumerate(['Kiểm tra tên và thứ tự đặc trưng','Dùng lại encoder và scaler đã huấn luyện','Dự đoán bằng mô hình được chọn','Tổng hợp, đánh giá và xuất báo cáo'],1):

                    st.markdown(f'<div class="step"><b>{i}</b><span>{text}</span></div>',unsafe_allow_html=True);st.write('')

        with right:st.subheader('Trạng thái mô hình');status_cards()

        return

    name=selected_model(s,'overview_model');r=s['results'][name];bad=int(attack_mask(r['pred']).sum());n=len(s['df'])

    cols=st.columns(4)

    for c,title,v in zip(cols,['Bản ghi đã phân tích','Bình thường','Nghi tấn công','Thời gian dự đoán'],[f'{n:,}',f'{n-bad:,}',f'{bad:,}',f'{r["seconds"]:.2f} s']):c.metric(title,v)

    st.write('');left,right=st.columns([2.2,1])

    with left:

        with st.container(border=True):

            st.subheader('Phân bố kết quả theo lô dữ liệu')

            batch=max(1,int(np.ceil(n/30)));mask=attack_mask(r['pred']).to_numpy()

            groups=[(f'{i+1}–{min(i+batch,n)}',int((~mask[i:i+batch]).sum()),int(mask[i:i+batch].sum())) for i in range(0,n,batch)]

            f=go.Figure()

            for label,index,color in [('Bình thường',1,'#0F766E'),('Nghi tấn công',2,'#C7475D')]:f.add_trace(go.Scatter(x=[g[0] for g in groups],y=[g[index] for g in groups],name=label,mode='lines',stackgroup='one',line=dict(color=color,width=2)))

            f.update_layout(xaxis_title='Khoảng bản ghi',yaxis_title='Số bản ghi');plot(f)

    with right:

        with st.container(border=True):

            st.subheader('Kết quả phân loại')

            f=go.Figure(go.Pie(labels=['Bình thường','Nghi tấn công'],values=[n-bad,bad],hole=.72,marker_colors=['#0F766E','#C7475D'],textinfo='percent'))

            f.add_annotation(text=f'{bad/n:.1%}<br>nghi tấn công',x=.5,y=.5,showarrow=False,font_size=19);plot(f)

    left,right=st.columns([2.2,1])

    with left:

        with st.container(border=True):st.subheader('Kết quả chi tiết');table(s,name,'overview')

    with right:

        with st.container(border=True):

            st.subheader('Phân bố nhãn')

            counts=pd.Series(r['pred']).value_counts().rename_axis('Nhãn').reset_index(name='Số bản ghi')

            plot(px.bar(counts,x='Số bản ghi',y='Nhãn',orientation='h',color='Nhãn',color_discrete_sequence=COLORS).update_layout(showlegend=False),320)

        if st.button('Phân tích CSV mới',type='primary',width='stretch'):go_page('Phân tích dữ liệu')

        if st.button('Phát lại phiên này',width='stretch'):go_page('Phát lại dữ liệu')



def analysis():

    header('Phân tích dữ liệu','Tải CSV → Kiểm tra đầu vào → Chọn mô hình → Xem kết quả')

    for c,i,t in zip(st.columns(4),range(1,5),['Tải dữ liệu','Kiểm tra schema','Chọn mô hình','Phân tích']):

        c.markdown(f'<div class="step"><b>{i}</b><span>{t}</span></div>',unsafe_allow_html=True)

    st.write('')

    with st.container(border=True):

        st.subheader('Nguồn dữ liệu')

        pending=staged_upload('CSV lưu lượng mạng',['csv'],'traffic_csv')

        if st.button('Nạp CSV',type='primary',disabled=pending is None,key='load_csv'):

            try:

                sample=pd.read_csv(io.BytesIO(pending.getvalue()),nrows=1)

                if sample.empty:raise ValueError('CSV không có bản ghi.')

                st.session_state.loaded_csv={'name':pending.name,'bytes':pending.getvalue()}

                st.session_state.load_notices['csv']='Nạp CSV thành công · '+pending.name

            except Exception as e:st.error('Không nạp được CSV: '+str(e))

        load_notice('csv')

    saved=st.session_state.loaded_csv

    file=None

    if saved:

        file=io.BytesIO(saved['bytes']);file.name=saved['name']

    read_mode=st.radio('Phạm vi phân tích',['Giới hạn số dòng','Toàn bộ CSV'],horizontal=True,key='read_mode')

    if read_mode=='Giới hạn số dòng':

        limit=st.number_input('Số dòng đọc tối đa',min_value=1,value=5000,step=100,key='row_limit')

        st.caption('Mặc định 5.000 dòng; có thể điều chỉnh theo dung lượng dữ liệu.')

    else:

        limit=None

        st.caption('Phân tích toàn bộ CSV, dự đoán theo lô. File tối đa 200 MB.')

    if file is None:

        st.caption('Chọn file CSV và nhấn Nạp CSV để bắt đầu.')

        if st.button('Mở cấu hình model'):go_page('Cấu hình')

        return

    try:

        # Read one more row to reliably report truncation, without loading the whole file.

        raw=pd.read_csv(io.BytesIO(file.getvalue()),nrows=int(limit)+1 if limit is not None else None,low_memory=False)

        truncated=limit is not None and len(raw)>limit

        df=(raw.iloc[:int(limit)] if limit is not None else raw).copy().reset_index(drop=True)

        if not len(df):raise ValueError('CSV không có bản ghi.')

    except Exception as e:st.error(f'Không đọc được CSV: {e}');return

    for c,title,v in zip(st.columns(4),['Số dòng đọc','Số cột','Cột nhãn','Giá trị trống'],[len(df),len(df.columns),sum(c in df for c in LABELS),int(df.isna().sum().sum())]):c.metric(title,f'{v:,}')

    if truncated:st.caption(f'Chỉ đọc {int(limit):,} dòng đầu. File còn dữ liệu phía sau.')

    with st.expander('Xem trước CSV và kiểu dữ liệu',expanded=True):

        st.dataframe(df.head(8),hide_index=True,width='stretch')

        st.caption('Hai cột Attack_label và Attack_type chỉ dùng đánh giá, không đưa vào dự đoán.')

    scaler,encoders=st.session_state.scaler,st.session_state.encoders

    if scaler is None or encoders is None:

        st.info('Nạp scaler và encoders trong Cấu hình để tiếp tục.')

        if st.button('Tải các file còn thiếu',type='primary'):go_page('Cấu hình')

        return

    try:

        names=feature_names(scaler,st.session_state.manifest)

        missing=[c for c in names if c not in df];extra=[c for c in df if c not in names+LABELS+list(getattr(scaler,'raw_ignored_columns_',[]))]

        with st.container(border=True):

            st.subheader('Kiểm tra tương thích')

            st.write(f'**Schema:** {len(names)} đặc trưng · **CSV:** {len([c for c in df if c not in LABELS])} cột đầu vào')

            if missing or extra:st.error(f'Thiếu: {missing or "không"} · Thừa: {extra or "không"}')

            else:

                st.success('Tên cột khớp schema. Dữ liệu được xếp theo thứ tự đặc trưng đã huấn luyện.')

                ignored_cols=list(getattr(scaler,'raw_ignored_columns_',[]))

                if ignored_cols:st.caption(f'Model sử dụng {len(names)} đặc trưng số. Bỏ qua {len(ignored_cols)} cột chữ theo cấu hình train; các cột này vẫn được giữ trong CSV.')

            with st.expander('Xem thứ tự đặc trưng'):st.dataframe(pd.DataFrame({'Thứ tự':range(1,len(names)+1),'Đặc trưng':names}),hide_index=True,width='stretch')

    except Exception as e:st.error(str(e));return

    available=[n for n,obj in st.session_state.models.items() if hasattr(obj,'predict')]

    if 'analysis_selected' in st.session_state:st.session_state.analysis_selected=[n for n in st.session_state.analysis_selected if n in available]

    selected=st.multiselect('Mô hình cần chạy',available,default=available[:1],key='analysis_selected')

    encoded=st.checkbox('CSV đã mã hóa đặc trưng nhưng CHƯA chuẩn hóa',value=st.session_state.encoded,key='analysis_encoded')

    st.caption('Không chọn mục trên nếu CSV còn chứa chuỗi gốc mà encoder cần transform. CSV đã chuẩn hóa không thuộc hai chế độ này.')

    if not st.session_state.confirmed:

        st.caption('Xác nhận pipeline trong Cấu hình trước khi phân tích.')

        if st.button('Kiểm tra và xác nhận cấu hình'):go_page('Cấu hình')

    env_bad=environment_issues()

    version_notes=[w for k in selected+['scaler','encoders'] for w in st.session_state.artifact_warnings.get(k,[]) if 'unpickle' in w.lower()]

    if env_bad:st.error('Môi trường chưa đúng. Mở Cấu hình → Kiểm tra môi trường, hoặc dừng web rồi chạy start_windows.bat trong bản mới.')

    if version_notes:st.error('PKL được lưu bằng phiên bản sklearn khác môi trường hiện tại. Cần sửa môi trường hoặc dùng lại bộ PKL tương thích trước khi dự đoán.')

    run=st.button('Bắt đầu phân tích',type='primary',disabled=not selected or not st.session_state.confirmed or bool(missing or extra) or bool(env_bad or version_notes),width='stretch')

    if run:

        try:

            bar=st.progress(0,text='Kiểm tra và biến đổi dữ liệu…')

            x,notes=prepare_frame(df,scaler,encoders,encoded=encoded,unknown=st.session_state.unknown,manifest=st.session_state.manifest)

            results={};failures=[]

            for index,name in enumerate(selected):

                try:

                    pred,score,seconds=infer(st.session_state.models[name],x,scaler,st.session_state.scaled[name],progress=lambda p:bar.progress((index+p)/len(selected),text=f'Đang chạy {name}…'))

                    metric,note=evaluate(df,pred,getattr(st.session_state.models[name],'classes_',None))

                    results[name]={'pred':pred,'score':score,'seconds':seconds,'metrics':metric,'evaluation_note':note,'scaled':st.session_state.scaled[name]}

                except Exception as e:failures.append(f'{name}: {e}')

            if not results:raise ValueError('Không model nào chạy thành công. '+'; '.join(failures))

            pipeline=('CSV đã mã hóa' if encoded else 'CSV gốc → encoders.transform')+' → chuẩn hóa theo cấu hình từng model'

            notes+=failures

            if truncated:notes.append(f'Chỉ phân tích {int(limit):,} dòng đầu của CSV.')

            if limit is None:notes.append('Đã đọc toàn bộ CSV tải lên.')

            st.session_state.session={'df':df,'results':results,'filename':file.name,'demo':False,'notes':notes,'run_time':pd.Timestamp.now().strftime('%d/%m/%Y %H:%M'),'pipeline':pipeline}

            st.session_state.replay_pos=0;st.session_state.replay_play=False

            bar.progress(1,text='Phân tích hoàn tất')

            st.success(f'Đã phân tích {len(df):,} dòng bằng {len(results)} mô hình.')

        except Exception as e:st.error(f'Dừng phân tích: {e}')

    s=st.session_state.session

    if s:

        session_notes(s)

        tabs=st.tabs(['Kết quả','Phân bố','Đánh giá'])

        name=selected_model(s,'analysis_model')

        with tabs[0]:table(s,name,'analysis')

        with tabs[1]:

            counts=pd.Series(s['results'][name]['pred']).value_counts().rename_axis('Nhãn').reset_index(name='Số bản ghi')

            plot(px.bar(counts,x='Nhãn',y='Số bản ghi',color='Nhãn',color_discrete_sequence=COLORS).update_layout(showlegend=False),350)

        with tabs[2]:metrics(s,name)



def metrics(s,name):

    r=s['results'][name];m=r['metrics']

    if not m:st.info(r['evaluation_note']);return

    if not s['demo']:st.caption('Chỉ số trên CSV đã tải lên; chưa xác nhận đây là tập kiểm thử độc lập. Precision, Recall và F1 dùng trung bình macro, zero_division=0.')

    else:st.caption('Chỉ số tính trên dữ liệu tổng hợp minh họa; không đại diện hiệu năng model thật.')

    for c,k in zip(st.columns(4),['Accuracy','Precision','Recall','F1']):c.metric(k,f'{m[k]:.2%}')

    with st.container(border=True):

        st.subheader('Confusion matrix')

        f=px.imshow(m['matrix'],x=m['labels'],y=m['labels'],text_auto=True,color_continuous_scale=['#102139','#0F766E'],aspect='auto',labels=dict(x='Dự đoán',y='Nhãn thật',color='Số bản ghi'))

        plot(f,max(330,len(m['labels'])*28));st.caption('Hàng: nhãn thật · Cột: dự đoán.')



def comparison():

    header('So sánh mô hình','Đối chiếu kết quả trên cùng các bản ghi trong phiên phân tích')

    s=st.session_state.session

    if not s:st.info('Chạy phân tích CSV bằng một hoặc nhiều model trước.');return

    rows=[]

    for name,r in s['results'].items():

        row={'Mô hình':name,'Số bản ghi':len(r['pred']),'Thời gian (s)':round(r['seconds'],3),'Nghi tấn công':int(attack_mask(r['pred']).sum())}

        if r['metrics']:row.update({k:r['metrics'][k] for k in ['Accuracy','Precision','Recall','F1']})

        rows.append(row)

    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch',column_config={k:st.column_config.NumberColumn(format='%.4f') for k in ['Accuracy','Precision','Recall','F1']})

    st.caption('Thời gian gồm dự đoán và tính xác suất nếu model hỗ trợ; không gồm đọc CSV và mã hóa. Chênh lệch chức năng đầu ra có thể ảnh hưởng thời gian.')

    left,right=st.columns([2,1])

    with left:

        with st.container(border=True):
            st.subheader('Các chỉ số đánh giá')

            valid=[{'Mô hình':n,'Chỉ số':k,'Giá trị':r['metrics'][k]} for n,r in s['results'].items() if r['metrics'] for k in ['Accuracy','Precision','Recall','F1']]

            if valid:plot(px.bar(pd.DataFrame(valid),x='Chỉ số',y='Giá trị',color='Mô hình',barmode='group',color_discrete_sequence=COLORS).update_yaxes(range=[0,1],tickformat='.0%'),320)

            else:st.info('Chưa có nhãn thật phù hợp để tính chỉ số.')
    with right:

        with st.container(border=True):

            st.subheader('Thời gian dự đoán')

            plot(px.bar(pd.DataFrame(rows),x='Mô hình',y='Thời gian (s)',color='Mô hình',color_discrete_sequence=COLORS).update_layout(showlegend=False),320)

    name=selected_model(s,'comparison_model');metrics(s,name)

    with st.container(border=True):

        st.subheader('Bản ghi có dự đoán khác nhau')

        if len(s['results'])<2:st.info('Chọn ít nhất hai model khi phân tích để xem mức độ đồng thuận.');return

        d=pd.DataFrame({n:r['pred'].astype(str) for n,r in s['results'].items()});d.insert(0,'Bản ghi',np.arange(1,len(d)+1))

        disagree=d.drop(columns='Bản ghi').nunique(axis=1)>1

        st.metric('Tỷ lệ đồng thuận hoàn toàn',f'{(~disagree).mean():.1%}')

        st.dataframe(d[disagree].head(200),hide_index=True,width='stretch')

        st.caption(f'{int(disagree.sum()):,} bản ghi khác nhau · Hiển thị tối đa 200 dòng. Đồng thuận không chứng minh dự đoán đúng.')



@st.fragment(run_every=1)

def replay_surface(s,name,batch,speed):

    key=(s['filename'],s['run_time'],name,id(s))

    if st.session_state.replay_key!=key:

        st.session_state.replay_key=key;st.session_state.replay_pos=0;st.session_state.replay_play=False

    a,b,c=st.columns(3)

    if a.button('Bắt đầu / Tiếp tục',type='primary',width='stretch',disabled=st.session_state.replay_pos>=len(s['df'])):st.session_state.replay_play=True;st.session_state.replay_tick=0

    if b.button('Tạm dừng',width='stretch'):st.session_state.replay_play=False

    if c.button('Đặt lại',width='stretch'):st.session_state.replay_play=False;st.session_state.replay_pos=0

    if st.session_state.replay_play:

        now=time.monotonic()

        if now-st.session_state.get('replay_tick',0)>=speed:

            st.session_state.replay_pos=min(len(s['df']),st.session_state.replay_pos+batch);st.session_state.replay_tick=now

        if st.session_state.replay_pos>=len(s['df']):st.session_state.replay_play=False

    pos=st.session_state.replay_pos;r=s['results'][name];mask=attack_mask(r['pred'][:pos]);bad=int(mask.sum())

    st.progress(pos/len(s['df']),text=f'Đã phát {pos:,}/{len(s["df"]):,} bản ghi')

    for col,title,value in zip(st.columns(3),['Đã phát','Bình thường','Nghi tấn công'],[pos,pos-bad,bad]):col.metric(title,f'{value:,}')

    d=result_df(s,name).iloc[:pos]

    left,right=st.columns([1.3,1])

    with left:

        with st.container(border=True):

            st.subheader('Kết quả gần đây')

            st.dataframe(d.tail(12),hide_index=True,width='stretch')

    with right:

        with st.container(border=True):

            st.subheader('Cảnh báo từ dữ liệu phát lại')

            flagged=d[d['Trạng thái']=='Nghi tấn công'].tail(8)

            if not len(flagged):st.caption('Chưa có cảnh báo trong các bản ghi đã phát.')

            for _,row in flagged.iloc[::-1].iterrows():

                st.warning(f'Bản ghi #{row["Bản ghi"]} · {row["Dự đoán"]}'+(f' · {row["IP nguồn"]}' if 'IP nguồn' in row else ''))

    if pos==len(s['df']):st.success('Đã phát hết dữ liệu trong phiên.')



def replay():

    header('Phát lại dữ liệu','Mô phỏng trình diễn từ CSV · Không phải thu thập lưu lượng mạng trực tiếp')

    s=st.session_state.session

    if not s:st.info('Cần một phiên phân tích hoặc dữ liệu minh họa để phát lại.');return

    name=selected_model(s,'replay_model')

    a,b=st.columns(2);batch=a.select_slider('Số bản ghi mỗi lô',options=[1,5,10,25,50,100],value=25,key='replay_batch')

    speed=b.select_slider('Khoảng cách giữa hai lô (giây)',options=[1,2,3,5],value=1,key='replay_speed')

    st.caption('Phát lại các dự đoán đã tính trong phiên. Đổi model hoặc phiên dữ liệu sẽ đặt lại tiến trình.')

    replay_surface(s,name,batch,speed)



def reports():

    header('Báo cáo & xuất dữ liệu','Lưu kết quả phân tích để đưa vào báo cáo và slide thuyết trình')

    s=st.session_state.session

    if not s:st.info('Chưa có phiên phân tích để xuất.');return

    with st.container(border=True):

        st.subheader('Thông tin phiên')

        st.write(f'**Dữ liệu:** {s["filename"]} · **Thời điểm:** {s["run_time"]}')

        st.write(f'**Số mẫu:** {len(s["df"]):,} · **Mô hình:** {", ".join(s["results"])}')

        st.caption(s['pipeline'])

        session_notes(s)

    name=selected_model(s,'report_model')

    a,b=st.tabs(['CSV kết quả','Báo cáo PDF'])

    with a:

        with st.container(border=True):

            st.subheader('Kết quả CSV');st.caption('Bản ghi, IP nếu có, nhãn dự đoán và model.')

            st.download_button('Tải CSV kết quả',result_df(s,name).to_csv(index=False).encode('utf-8-sig'),f'ket_qua_{name.replace(" ","_")}.csv','text/csv',width='stretch')

    with b:

        with st.container(border=True):

            st.subheader('Báo cáo PDF');st.caption('Tóm tắt, phân bố nhãn, chỉ số và confusion matrix.')

            try:st.download_button('Tải báo cáo PDF',pdf_report(s),'IoT_Sentinel_Bao_cao.pdf','application/pdf',width='stretch')

            except ModuleNotFoundError:

                st.warning('Thiếu reportlab trong Python đang chạy. Dừng web, mở start_windows.bat để cài đúng môi trường rồi thử lại.')

            except Exception as e:st.error('Chưa tạo được PDF: '+str(e))

    st.subheader('Biểu đồ cho slide')

    counts=pd.Series(s['results'][name]['pred']).value_counts().rename_axis('Nhãn').reset_index(name='Số bản ghi')

    f=px.bar(counts,x='Số bản ghi',y='Nhãn',orientation='h',color='Nhãn',color_discrete_sequence=COLORS).update_layout(showlegend=False,yaxis={'categoryorder':'total ascending'})

    plot(f,max(360,len(counts)*32));st.caption('Dùng nút máy ảnh trên thanh công cụ của biểu đồ để tải ảnh PNG.')

    st.download_button('Tải biểu đồ tương tác HTML',f.to_html(include_plotlyjs=True).encode(),'bieu_do_phan_bo.html','text/html')



def configuration():

    header('Cấu hình mô hình','Quản lý model, encoder, scaler và xác nhận pipeline dùng khi huấn luyện')

    with st.expander('Thông tin môi trường',expanded=False):

        st.subheader('Kiểm tra môi trường')

        st.dataframe(pd.DataFrame(environment_status()),hide_index=True,width='stretch')

        st.caption('Python đang chạy: '+sys.executable)

        if environment_issues():

            st.error('Dừng web bằng Ctrl+C tại cửa sổ chạy, sau đó mở start_windows.bat. File khởi động sẽ kiểm tra và cài lại đúng thư viện trong .venv, gồm sklearn 1.6.1 và reportlab.')

            with st.expander('Nếu bạn muốn sửa bằng Terminal'):

                st.code(f'"{sys.executable}" -m pip install --upgrade -r requirements.txt')

                st.caption('Chạy lệnh trong thư mục iot_sentinel rồi khởi động lại web. Không dùng lệnh pip riêng nếu chưa chắc đang ở đúng Python.')

        else:st.success('Đúng bộ thư viện đã kiểm tra, có bộ xuất PDF.')

    st.caption('Chỉ tải PKL do nhóm của bạn tạo hoặc nguồn bạn tin tưởng: định dạng pickle có thể chạy mã khi nạp.')

    top=st.columns(3)

    for col,(name,filename) in zip(top,MODEL_FILES.items()):

        with col:

            with st.container(border=True):

                st.subheader(name)

                obj=st.session_state.models.get(name)

                if obj is not None:

                    st.caption(type(obj).__name__);st.metric('Đặc trưng đầu vào',getattr(obj,'n_features_in_','—'))

                    with st.expander('Các lớp đầu ra'):st.write(list(map(str,getattr(obj,'classes_',[]))))

                else:st.info('Chưa tải model')

                up=staged_upload('Chọn model PKL',['pkl','joblib'],filename)

                load_notice(name)

                if up and st.button('Nạp model',key='load_'+filename,width='stretch'):

                    try:

                        obj,notes=load_artifact(io.BytesIO(up.getvalue()))

                        if not hasattr(obj,'predict'):raise ValueError('File không phải model có predict().')

                        st.session_state.models[name]=obj;st.session_state.artifact_warnings[name]=notes;st.session_state.load_notices[name]='Nạp '+name+' thành công · '+up.name;st.session_state.confirmed=False;st.session_state.confirm_pipeline=False;st.rerun()

                    except Exception as e:st.error(str(e))

    cols=st.columns(2)

    for col,key,title in zip(cols,['scaler','encoders'],['Scaler chuẩn hóa','Label encoders']):

        with col:

            with st.container(border=True):

                st.subheader(title);obj=st.session_state[key]

                st.caption('Đã tải: '+type(obj).__name__ if obj is not None else 'Chưa tải')

                up=staged_upload(title+' PKL',['pkl','joblib'],'upload_'+key)

                load_notice(key)

                if up and st.button('Nạp '+title,key='load_'+key,width='stretch'):

                    try:

                        obj,notes=load_artifact(io.BytesIO(up.getvalue()))

                        if key=='scaler' and not hasattr(obj,'transform'):raise ValueError('Scaler cần transform().')

                        if key=='encoders' and not isinstance(obj,dict):raise ValueError('Encoders cần dictionary.')

                        st.session_state[key]=obj;st.session_state.artifact_warnings[key]=notes;st.session_state.load_notices[key]='Nạp '+title+' thành công · '+up.name;st.session_state.confirmed=False;st.session_state.confirm_pipeline=False;st.rerun()

                    except Exception as e:st.error(str(e))

    if st.session_state.scaler is not None:

        with st.expander('Schema và encoder đang dùng'):

            try:

                names=feature_names(st.session_state.scaler,st.session_state.manifest)

                enc=st.session_state.encoders or {};enc=enc.get('categorical_encoders',enc)

                st.dataframe(pd.DataFrame({'Thứ tự':range(1,len(names)+1),'Đặc trưng':names,'Encoder':[c in enc for c in names]}),hide_index=True,width='stretch')

            except Exception as e:st.warning(str(e))

    for name,notes in st.session_state.artifact_warnings.items():

        if notes:

            with st.expander('Cảnh báo khi tải '+name):

                for note in notes:st.warning(note)

    with st.container(border=True):

        st.subheader('Pipeline dự đoán')

        st.caption('Chọn cách chuẩn hóa theo notebook huấn luyện của từng mô hình.')

        new_scaled={}

        for col,name in zip(st.columns(3),MODEL_FILES):

            new_scaled[name]=col.checkbox(name+' dùng scaler',value=st.session_state.scaled[name],key='scale_'+name)

        encoded=st.checkbox('Mặc định CSV đã mã hóa nhưng chưa chuẩn hóa',value=st.session_state.encoded,key='encoded_default')

        mode=st.radio('Khi gặp giá trị chưa có trong encoder',['Dừng và báo cột lỗi','Thay bằng lớp 0.0/0 đã tồn tại trong encoder'],index=0 if st.session_state.unknown=='block' else 1,key='unknown_mode')

        unknown='block' if mode.startswith('Dừng') else 'fallback'

        if unknown=='fallback':st.caption('Chỉ dùng nếu nhóm chấp nhận quy tắc này. Thay giá trị lạ có thể làm sai lệch dự đoán; ứng dụng sẽ ghi số lượng thay thế trong báo cáo.')

        if new_scaled!=st.session_state.scaled or encoded!=st.session_state.encoded or unknown!=st.session_state.unknown:

            st.session_state.scaled=new_scaled;st.session_state.encoded=encoded;st.session_state.unknown=unknown;st.session_state.confirmed=False;st.session_state.confirm_pipeline=False

        confirm=st.checkbox('Tôi đã đối chiếu notebook train: đúng bộ encoder/scaler, thứ tự đặc trưng và lựa chọn chuẩn hóa cho từng model.',value=st.session_state.confirmed,key='confirm_pipeline')

        if st.button('Lưu cấu hình phiên',type='primary'):

            st.session_state.confirmed=confirm;st.success('Đã lưu cấu hình cho phiên làm việc hiện tại.' if confirm else 'Đã lưu. Pipeline chưa được xác nhận để dự đoán.')

    st.caption('Các file tải qua giao diện và kết quả chỉ giữ trong phiên. Muốn dùng lại khi khởi động, đặt PKL trong thư mục models theo hướng dẫn.')



with st.sidebar:

    st.markdown('<div class="brand"><svg class="shield" viewBox="0 0 48 54" fill="none"><path d="M24 3L43 10V26C43 38 31 47 24 51C17 47 5 38 5 26V10L24 3Z" stroke="currentColor" stroke-width="3"/><path d="M24 10L36 15V26C36 34 29 41 24 44C19 41 12 34 12 26V15L24 10Z" fill="#168adb" opacity=".65"/></svg><div><b>IoT <span>SENTINEL</span></b><small>Phát hiện tấn công IoT</small></div></div>',unsafe_allow_html=True)

    pages=['Tổng quan','Phân tích dữ liệu','So sánh mô hình','Phát lại dữ liệu','Báo cáo','Cấu hình']

    st.session_state.nav_page=st.session_state.page

    st.radio('Điều hướng',pages,key='nav_page',label_visibility='collapsed',on_change=nav_changed)

    st.divider();st.caption('PHIÊN PHÂN TÍCH')

    s=st.session_state.session

    if s:st.markdown(f'<div class="sidebar-session"><strong>{html.escape(s["filename"])}</strong><small>{len(s["df"]):,} bản ghi · {len(s["results"])} mô hình</small></div>',unsafe_allow_html=True)

    else:st.caption('Chưa có dữ liệu phân tích')

    if st.button('Dùng dữ liệu minh họa',width='stretch'):

        st.session_state.session=demo_session();st.session_state.replay_pos=0;st.session_state.replay_play=False;st.rerun()

    if s and st.button('Xóa kết quả phiên',width='stretch'):

        st.session_state.session=None;st.session_state.replay_pos=0;st.session_state.replay_play=False;st.rerun()

    presentation=st.toggle('Chế độ trình chiếu',value=False,key='presentation')

    st.caption('Nghiên cứu phát hiện tấn công IoT\n\nRF · DT · SVM')

if presentation:

    st.markdown('<style>[data-testid="stMetricValue"]{font-size:2.7rem!important}.stMarkdown p{font-size:1.1rem}h1{font-size:2.5rem!important}</style>',unsafe_allow_html=True)

{'Tổng quan':overview,'Phân tích dữ liệu':analysis,'So sánh mô hình':comparison,'Phát lại dữ liệu':replay,'Báo cáo':reports,'Cấu hình':configuration}[st.session_state.page]()

st.markdown('<div class="footer"><span>IoT SENTINEL · Nghiên cứu phát hiện tấn công IoT</span><span>CSV · Random Forest · Decision Tree · SVM</span></div>',unsafe_allow_html=True)
