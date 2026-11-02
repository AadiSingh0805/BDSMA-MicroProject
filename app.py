"""
TubePulse: Large-Scale Comment Sentiment Mining and Audience Feedback Analytics
Distributed Processing: PySpark SQL & UDF
NLP: NLTK VADER Lexicon & TF-IDF Bigrams
Visuals: Plotly & Streamlit
"""

import os
import importlib
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

import youtube_api
import spark_engine
importlib.reload(youtube_api)
importlib.reload(spark_engine)

from youtube_api import extract_video_id, fetch_youtube_comments, get_fallback_data, get_api_key
from spark_engine import run_pyspark_pipeline

# Page Setup
st.set_page_config(
    page_title="TubePulse - Audience Feedback Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimalist Dark Stylesheet
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #090d16;
        color: #e2e8f0;
    }

    .stApp {
        background-color: #090d16;
    }

    .metric-card {
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .metric-value {
        font-family: 'Fira Code', monospace;
        font-size: 1.55rem;
        font-weight: 700;
        color: #38bdf8;
        line-height: 1.2;
    }

    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 4px;
    }

    .metric-sub {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 4px;
        font-family: 'Fira Code', monospace;
    }

    .comment-item {
        background: #0d1527;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }

    .stButton > button {
        background: #0284c7 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        height: 42px !important;
    }
    .stButton > button:hover {
        background: #0369a1 !important;
    }

    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2rem !important;
        max-width: 1320px !important;
    }

    header[data-testid="stHeader"] {
        background: rgba(9, 13, 22, 0.8) !important;
    }
</style>
""", unsafe_allow_html=True)

api_key = get_api_key()

# Application Header
h_left, h_right = st.columns([8, 4])
with h_left:
    st.markdown("<h2 style='margin:0; font-weight:700; color:#f8fafc; letter-spacing:-0.5px;'>TubePulse</h2>", unsafe_allow_html=True)
    st.markdown("<p style='margin:0 0 16px 0; font-size:0.875rem; color:#94a3b8;'>Large-Scale Comment Sentiment Mining & Audience Reception Analytics</p>", unsafe_allow_html=True)

with h_right:
    api_color = "#10b981" if api_key else "#64748b"
    api_label = "API: Live Connected (.env)" if api_key else "API: Offline Benchmark"
    st.markdown(f"""
    <div style='text-align:right; font-family:Fira Code; font-size:0.75rem; margin-top:8px;'>
        <span style='color:{api_color}; margin-right:12px;'>● {api_label}</span>
        <span style='color:#38bdf8;'>PySpark 3.5 Engine</span>
    </div>
    """, unsafe_allow_html=True)

# Search Input Bar
c_input, c_btn = st.columns([9, 3])

with c_input:
    url_input = st.text_input(
        "YouTube Video URL or ID:",
        value="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        placeholder="Paste any YouTube URL (e.g. https://www.youtube.com/watch?v=...)",
        label_visibility="collapsed"
    )

with c_btn:
    analyze_trigger = st.button("Run Analytics", use_container_width=True)

target_id = extract_video_id(url_input)

# Initialize Session State
if "analyzed_target_id" not in st.session_state:
    st.session_state.analyzed_target_id = None
    st.session_state.pipeline_results = None
    st.session_state.video_meta = None
    st.session_state.raw_comments = None
    st.session_state.api_notice = None

# Determine if we should trigger a new analysis
needs_analysis = (
    analyze_trigger or 
    st.session_state.analyzed_target_id != target_id or 
    st.session_state.pipeline_results is None
)

if needs_analysis:
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    def handle_progress(pct: int, msg: str):
        progress_placeholder.progress(pct)
        status_placeholder.markdown(
            f"<div style='font-family:Fira Code; font-size:0.78rem; color:#38bdf8; margin: 4px 0 12px 0;'>[{pct}%] {msg}</div>",
            unsafe_allow_html=True
        )

    # Ingestion
    video_meta, raw_comments, api_notice = fetch_youtube_comments(
        target_id,
        max_comments=500,
        api_key=api_key,
        on_progress=handle_progress
    )

    # PySpark Processing
    results = run_pyspark_pipeline(
        raw_comments,
        video_meta["views"],
        video_meta["likes"],
        on_progress=handle_progress
    )

    # Clear Progress Bar
    progress_placeholder.empty()
    status_placeholder.empty()

    # Cache into session state
    st.session_state.analyzed_target_id = target_id
    st.session_state.pipeline_results = results
    st.session_state.video_meta = video_meta
    st.session_state.raw_comments = raw_comments
    st.session_state.api_notice = api_notice

# Retrieve cached results
pipeline_results = st.session_state.pipeline_results
video_meta = st.session_state.video_meta
raw_comments = st.session_state.raw_comments
api_notice = st.session_state.api_notice

# Show API Notice / Warning if applicable
if api_notice:
    st.warning(f"**Notice:** {api_notice}")

# Video Player & Core KPI Metrics Section
col_player, col_kpis = st.columns([5, 7])

with col_player:
    # Embedded Video Player
    video_embed_url = f"https://www.youtube.com/watch?v={target_id}"
    st.video(video_embed_url)
    st.markdown(f"""
    <div style="background:#0b1120; border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px 14px; margin-top:6px;">
        <span style="font-size:0.875rem; font-weight:600; color:#f1f5f9; display:block; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{video_meta['title']}</span>
        <span style="font-size:0.75rem; color:#94a3b8;">Channel: {video_meta['channel']} | ID: <span style="font-family:'Fira Code'; color:#38bdf8;">{target_id}</span></span>
    </div>
    """, unsafe_allow_html=True)

with col_kpis:
    # 2x2 Grid of Key Metrics
    row1_c1, row1_c2 = st.columns(2)
    like_ratio = pipeline_results['discrepancy']['like_ratio']
    
    with row1_c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Public Views</div>
            <div class="metric-value">{video_meta['views']:,}</div>
            <div class="metric-sub">YouTube Data API v3</div>
        </div>
        """, unsafe_allow_html=True)
        
    with row1_c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Public Likes</div>
            <div class="metric-value">{video_meta['likes']:,}</div>
            <div class="metric-sub">{like_ratio}% Like-to-View Ratio</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    
    row2_c1, row2_c2 = st.columns(2)
    clean_pct = pipeline_results['clean_ratio']
    disc = pipeline_results['discrepancy']
    status_color = disc['color']
    
    with row2_c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sampled Comments</div>
            <div class="metric-value">{len(raw_comments):,}</div>
            <div class="metric-sub">{clean_pct}% Verified Audience</div>
        </div>
        """, unsafe_allow_html=True)
        
    with row2_c2:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 3px solid {status_color};">
            <div class="metric-label">Discrepancy Index</div>
            <div class="metric-value" style="color: {status_color};">{disc['score']} / 100</div>
            <div class="metric-sub" style="color: {status_color}; font-weight:600;">{disc['status']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# Analytics Visualizations
col_left, col_right = st.columns([5, 7])

with col_left:
    st.markdown("<h4 style='font-size:0.95rem; font-weight:600; color:#f1f5f9; margin-bottom:4px;'>Sentiment Polarity Split</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.75rem; color:#64748b; margin-bottom:12px;'>Scored via PySpark VADER compound UDF</p>", unsafe_allow_html=True)
    
    split = pipeline_results["sentiment_split"]
    
    donut_fig = go.Figure(data=[go.Pie(
        labels=['Positive', 'Neutral', 'Negative'],
        values=[split['positive'], split['neutral'], split['negative']],
        hole=0.72,
        marker=dict(colors=['#10b981', '#64748b', '#f43f5e']),
        textinfo='label+percent',
        insidetextorientation='horizontal'
    )])
    donut_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=0, b=0, l=0, r=0),
        height=210,
        showlegend=False,
        font=dict(color='#e2e8f0', size=11)
    )
    st.plotly_chart(donut_fig, use_container_width=True)
    
    # Distribution breakdown pills
    p1, p2, p3 = st.columns(3)
    p1.markdown(f"<div style='text-align:center; padding:8px; background:#0f172a; border-radius:6px;'><div style='font-size:0.7rem; color:#10b981; font-weight:600;'>POSITIVE</div><div style='font-family:Fira Code; font-size:1.1rem; font-weight:700;'>{split['positive']}%</div></div>", unsafe_allow_html=True)
    p2.markdown(f"<div style='text-align:center; padding:8px; background:#0f172a; border-radius:6px;'><div style='font-size:0.7rem; color:#94a3b8; font-weight:600;'>NEUTRAL</div><div style='font-family:Fira Code; font-size:1.1rem; font-weight:700;'>{split['neutral']}%</div></div>", unsafe_allow_html=True)
    p3.markdown(f"<div style='text-align:center; padding:8px; background:#0f172a; border-radius:6px;'><div style='font-size:0.7rem; color:#f43f5e; font-weight:600;'>NEGATIVE</div><div style='font-family:Fira Code; font-size:1.1rem; font-weight:700;'>{split['negative']}%</div></div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<h4 style='font-size:0.95rem; font-weight:600; color:#f1f5f9; margin-bottom:4px;'>Recurring Topics (TF-IDF Bigrams)</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.75rem; color:#64748b; margin-bottom:12px;'>Most frequent 2-word keyphrases driving sentiment</p>", unsafe_allow_html=True)
    
    bar_fig = go.Figure(go.Bar(
        x=pipeline_results['bigram_freqs'],
        y=pipeline_results['bigram_labels'],
        orientation='h',
        marker=dict(
            color='#6366f1',
            line=dict(color='#818cf8', width=1)
        )
    ))
    bar_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=0, b=0, l=0, r=0),
        height=190,
        xaxis=dict(gridcolor='rgba(255,255,255,0.06)', tickfont=dict(family='Fira Code', size=10, color='#94a3b8')),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, color='#e2e8f0')),
        font=dict(color='#e2e8f0')
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    # Spam Telemetry Indicator
    spam_pct = pipeline_results['spam_ratio']
    spam_count = pipeline_results['spam_count']
    clean_count = pipeline_results['clean_count']
    
    st.markdown(f"""
    <div style="background:#0f172a; border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:8px 12px; margin-top:8px;">
        <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#94a3b8; margin-bottom:4px;">
            <span>Spam & Bot Rule Telemetry</span>
            <span style="font-family:Fira Code; color:{ '#f43f5e' if spam_pct > 15 else '#94a3b8' };">{spam_pct}% Flagged</span>
        </div>
        <div style="height:6px; background:#1e293b; border-radius:3px; overflow:hidden; display:flex;">
            <div style="width:{100-spam_pct}%; background:#10b981;"></div>
            <div style="width:{spam_pct}%; background:#f43f5e;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#64748b; font-family:Fira Code; margin-top:4px;">
            <span>Verified: {clean_count}</span>
            <span>Bots (WhatsApp/Telegram/Crypto): {spam_count}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

# Comment Explorer Feed (Sortable & Filterable)
st.markdown("<h4 style='font-size:0.95rem; font-weight:600; color:#f1f5f9; margin-bottom:4px;'>Audience Feedback Stream</h4>", unsafe_allow_html=True)
st.markdown("<p style='font-size:0.75rem; color:#64748b; margin-bottom:12px;'>Extracted comments ranked by community like engagement</p>", unsafe_allow_html=True)

filter_choice = st.radio(
    "Filter Stream:",
    ["All Feedback", "Positive Polarity", "Negative Polarity", "Flagged Spam"],
    horizontal=True,
    label_visibility="collapsed"
)

processed_df = pipeline_results["processed_df"]
if filter_choice == "Positive Polarity":
    display_df = processed_df[processed_df["category"] == "Positive"]
elif filter_choice == "Negative Polarity":
    display_df = processed_df[processed_df["category"] == "Negative"]
elif filter_choice == "Flagged Spam":
    display_df = processed_df[processed_df["is_spam"]]
else:
    display_df = processed_df

display_df = display_df.sort_values(by="like_count", ascending=False)

for _, row in display_df.head(6).iterrows():
    color = "#10b981" if row["category"] == "Positive" else ("#f43f5e" if row["category"] == "Negative" else "#64748b")
    st.markdown(f"""
    <div class="comment-item">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <span style="font-family:Fira Code; font-size:0.8rem; font-weight:600; color:#38bdf8;">@{row['author']}</span>
            <div>
                <span style="font-family:Fira Code; font-size:0.75rem; color:#64748b; margin-right:12px;">Likes: {row['like_count']:,}</span>
                <span style="font-family:Fira Code; font-size:0.7rem; color:{color}; border:1px solid {color}44; padding:1px 6px; border-radius:4px;">
                    {row['category']} ({row['compound']:.2f})
                </span>
            </div>
        </div>
        <div style="font-size:0.85rem; color:#cbd5e1; line-height:1.4;">{row['text']}</div>
    </div>
    """, unsafe_allow_html=True)
