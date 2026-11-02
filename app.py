"""
Sentube: Large-Scale Comment Sentiment Mining and Audience Feedback Analytics
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

from youtube_api import extract_video_id, fetch_youtube_comments, get_fallback_data, get_api_key, get_browse_feed
from spark_engine import run_pyspark_pipeline

# Page Setup
st.set_page_config(
    page_title="Sentube - Browse & Audience Feedback Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Minimalist Dark Stylesheet (YouTube Dark Palette)
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

    /* Header Clearance & Layout */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1360px !important;
    }

    /* Clean Navbar Alignment */
    div[data-testid="stHorizontalBlock"] {
        align-items: center !important;
    }

    /* Search Input Styling */
    div[data-testid="stTextInput"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stTextInput"] > div {
        margin: 0 !important;
    }
    div[data-testid="stTextInput"] input {
        height: 42px !important;
        border-radius: 8px !important;
        background: #0f172a !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        font-size: 0.875rem !important;
        color: #f8fafc !important;
        padding: 0 16px !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 1px #38bdf8 !important;
    }

    /* YouTube-style Category Chips */
    div[data-testid="stRadio"] > div {
        flex-direction: row !important;
        gap: 8px !important;
        overflow-x: auto !important;
        padding: 4px 0 10px 0 !important;
    }
    div[data-testid="stRadio"] label {
        background: #0f172a !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        padding: 6px 16px !important;
        font-size: 0.8rem !important;
        color: #94a3b8 !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        margin: 0 !important;
    }
    div[data-testid="stRadio"] label:hover {
        background: #1e293b !important;
        color: #f8fafc !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    div[data-testid="stRadio"] label[data-checked="true"],
    div[data-testid="stRadio"] div[aria-checked="true"] {
        background: #f8fafc !important;
        color: #090d16 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stRadio"] input {
        display: none !important;
    }

    /* Clickable Video Card Link */
    a.video-card-link {
        text-decoration: none !important;
        display: block !important;
        color: inherit !important;
        margin-bottom: 22px;
    }
    .video-card {
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        cursor: pointer;
    }
    a.video-card-link:hover .video-card {
        border-color: rgba(56, 189, 248, 0.45);
        transform: translateY(-3px);
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.5);
    }
    a.video-card-link:hover .video-title {
        color: #38bdf8 !important;
    }
    .video-card img {
        width: 100%;
        aspect-ratio: 16 / 9;
        object-fit: cover;
        display: block;
        background: #070b12;
    }
    .video-card-body {
        padding: 12px 14px 14px 14px;
    }
    .video-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #f8fafc;
        line-height: 1.35;
        margin-bottom: 6px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        height: 2.5rem;
        transition: color 0.15s ease;
    }
    .video-channel {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .video-meta-row {
        font-size: 0.72rem;
        color: #64748b;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Metric Cards */
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

    /* Comment row styling */
    .comment-item {
        background: #0d1527;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }

    /* Buttons */
    .stButton > button {
        background: #0284c7 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        height: 40px !important;
    }
    .stButton > button:hover {
        background: #0369a1 !important;
    }
</style>
""", unsafe_allow_html=True)

# State Management: Current View & Selected Video
if "current_view" not in st.session_state:
    st.session_state.current_view = "browse"
if "selected_video_id" not in st.session_state:
    st.session_state.selected_video_id = None
if "selected_category" not in st.session_state:
    st.session_state.selected_category = "All"
if "analyzed_id" not in st.session_state:
    st.session_state.analyzed_id = None
    st.session_state.pipeline_results = None
    st.session_state.video_meta = None
    st.session_state.raw_comments = None
    st.session_state.api_notice = None

# Support URL query parameter ?v=XXXXX for direct click redirection
if "v" in st.query_params and st.query_params["v"]:
    param_vid = st.query_params["v"]
    if st.session_state.selected_video_id != param_vid or st.session_state.current_view != "watch":
        st.session_state.selected_video_id = param_vid
        st.session_state.current_view = "watch"

api_key = get_api_key()

# Pixel-Perfect Aligned Top Navbar (YouTube Style)
nav_col1, nav_col2, nav_col3 = st.columns([2.8, 6.4, 2.8], vertical_alignment="center")

with nav_col1:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px;">
        <span style="font-size:1.4rem; font-weight:800; color:#f8fafc; letter-spacing:-0.5px; line-height:1;">
            Sen<span style="color:#38bdf8;">tube</span>
        </span>
        <span style="font-family:'Fira Code'; font-size:0.68rem; color:#64748b; background:#0f172a; border:1px solid rgba(255,255,255,0.08); padding:2px 6px; border-radius:4px; white-space:nowrap;">
            PySpark 3.5
        </span>
    </div>
    """, unsafe_allow_html=True)

with nav_col2:
    search_query = st.text_input(
        "Search or paste YouTube URL:",
        value="",
        placeholder="Paste any YouTube URL or ID (press Enter to analyze)...",
        label_visibility="collapsed",
        key="global_nav_search"
    )
    if search_query.strip():
        extracted = extract_video_id(search_query)
        if extracted:
            st.query_params["v"] = extracted
            st.session_state.selected_video_id = extracted
            st.session_state.current_view = "watch"
            st.rerun()

with nav_col3:
    api_color = "#10b981" if api_key else "#64748b"
    api_label = "● API: Connected (.env)" if api_key else "● API: Offline Benchmark"
    st.markdown(f"""
    <div style='text-align:right; font-family:"Fira Code"; font-size:0.75rem; white-space:nowrap;'>
        <span style='color:{api_color}; background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.2); padding:5px 12px; border-radius:6px;'>
            {api_label}
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

# ==========================================
# VIEW 1: BROWSE LANDING PAGE (YouTube Feed)
# ==========================================
if st.session_state.current_view == "browse":
    
    # YouTube-style Category Filter Chips (Horizontal Pills)
    categories = ["All", "Tech Reviews", "AI & Big Data", "Gaming", "Science", "Music"]
    
    if hasattr(st, "pills"):
        selected_cat = st.pills(
            "Category",
            categories,
            default=st.session_state.selected_category,
            label_visibility="collapsed"
        )
        if selected_cat != st.session_state.selected_category:
            st.session_state.selected_category = selected_cat
            st.rerun()
    else:
        selected_cat = st.radio(
            "Category",
            categories,
            index=categories.index(st.session_state.selected_category) if st.session_state.selected_category in categories else 0,
            horizontal=True,
            label_visibility="collapsed"
        )
        if selected_cat != st.session_state.selected_category:
            st.session_state.selected_category = selected_cat
            st.rerun()

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # Fetch Curated Videos for Selected Category
    videos = get_browse_feed(st.session_state.selected_category)

    # Display 3-Column Video Feed Grid (Whole Card Clickable!)
    cols = st.columns(3)
    for idx, vid in enumerate(videos):
        col = cols[idx % 3]
        with col:
            thumb_url = vid.get("thumb") or f"https://i.ytimg.com/vi/{vid['id']}/hqdefault.jpg"
            st.markdown(f"""
            <a href="/?v={vid['id']}" target="_self" class="video-card-link">
                <div class="video-card">
                    <img src="{thumb_url}" onerror="this.onerror=null; this.src='https://i.ytimg.com/vi/aircAruvnKk/hqdefault.jpg';" alt="Thumbnail" />
                    <div class="video-card-body">
                        <div class="video-title">{vid['title']}</div>
                        <div class="video-channel">{vid['channel']}</div>
                        <div class="video-meta-row">
                            <span>{vid['views']} • {vid['published']}</span>
                            <span style="color:{vid.get('tag_color', '#38bdf8')}; font-weight:600; font-family:Fira Code; font-size:0.7rem;">
                                {vid.get('discrepancy_preview', 'Live')}
                            </span>
                        </div>
                    </div>
                </div>
            </a>
            """, unsafe_allow_html=True)

# ==========================================
# VIEW 2: WATCH & SCROLL ANALYTICS VIEW
# ==========================================
elif st.session_state.current_view == "watch":
    
    target_id = st.session_state.selected_video_id or "aircAruvnKk"

    # Navigation strip: Back Button + Current Video ID
    top_nav_left, top_nav_right = st.columns([4, 8], vertical_alignment="center")
    with top_nav_left:
        if st.button("← Back to Browse Feed", use_container_width=False):
            st.query_params.clear()
            st.session_state.selected_video_id = None
            st.session_state.current_view = "browse"
            st.rerun()
    with top_nav_right:
        st.markdown(f"""
        <div style='text-align:right; font-family:Fira Code; font-size:0.75rem; color:#64748b;'>
            Target Video ID: <span style='color:#38bdf8; font-weight:600;'>{target_id}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # 1. Main Embedded Video Player
    video_url = f"https://www.youtube.com/watch?v={target_id}"
    st.video(video_url)

    # Check if we need to run pipeline analysis for this video ID
    needs_computation = (
        st.session_state.analyzed_id != target_id or 
        st.session_state.pipeline_results is None
    )

    if needs_computation:
        prog_bar = st.empty()
        status_msg = st.empty()

        def on_pipe_progress(pct: int, msg: str):
            prog_bar.progress(pct)
            status_msg.markdown(
                f"<div style='font-family:Fira Code; font-size:0.78rem; color:#38bdf8; margin: 4px 0 12px 0;'>[{pct}%] {msg}</div>",
                unsafe_allow_html=True
            )

        # Ingestion
        video_meta, raw_comments, api_notice = fetch_youtube_comments(
            target_id,
            max_comments=500,
            api_key=api_key,
            on_progress=on_pipe_progress
        )

        # PySpark Transformation
        results = run_pyspark_pipeline(
            raw_comments,
            video_meta["views"],
            video_meta["likes"],
            on_progress=on_pipe_progress
        )

        # Clear progress bar completely
        prog_bar.empty()
        status_msg.empty()

        # Cache in session state
        st.session_state.analyzed_id = target_id
        st.session_state.pipeline_results = results
        st.session_state.video_meta = video_meta
        st.session_state.raw_comments = raw_comments
        st.session_state.api_notice = api_notice

    # Retrieve cached video and analytics data
    pipeline_results = st.session_state.pipeline_results
    video_meta = st.session_state.video_meta
    raw_comments = st.session_state.raw_comments
    api_notice = st.session_state.api_notice

    # Video Title & Details Header directly below player
    st.markdown(f"""
    <div style="background:#0b1120; border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:12px 16px; margin: 8px 0 20px 0; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
        <div>
            <h3 style="margin:0 0 4px 0; font-size:1.15rem; font-weight:700; color:#f8fafc;">{video_meta['title']}</h3>
            <div style="font-size:0.8rem; color:#94a3b8;">
                Channel: <strong style="color:#e2e8f0;">{video_meta['channel']}</strong> | 
                Views: {video_meta['views']:,} | 
                Likes: {video_meta['likes']:,}
            </div>
        </div>
        <div>
            <a href="{video_url}" target="_blank" style="font-family:'Fira Code'; font-size:0.75rem; color:#38bdf8; text-decoration:none; background:#0f172a; border:1px solid rgba(56,189,248,0.25); padding:6px 12px; border-radius:6px; display:inline-block;">Open on YouTube ↗</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if api_notice:
        st.warning(f"**Notice:** {api_notice}")

    # ==========================================
    # ON SCROLL: FULL ANALYTICS DASHBOARD SUITE
    # ==========================================
    st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.08); margin:24px 0;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin:0 0 4px 0; font-size:1.1rem; font-weight:700; color:#f8fafc;'>Audience Sentiment & Distributed Reception Telemetry</h3>", unsafe_allow_html=True)
    st.markdown("<p style='margin:0 0 20px 0; font-size:0.78rem; color:#64748b;'>Distributed text sanitization, NLTK VADER compound polarity UDF, TF-IDF bigrams & Discrepancy Index</p>", unsafe_allow_html=True)

    # 4 Key Metric Cards
    k1, k2, k3, k4 = st.columns(4)
    like_ratio = pipeline_results['discrepancy']['like_ratio']
    clean_pct = pipeline_results['clean_ratio']
    disc = pipeline_results['discrepancy']
    status_color = disc['color']

    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Public Views</div>
            <div class="metric-value">{video_meta['views']:,}</div>
            <div class="metric-sub">YouTube Data API v3</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Public Likes</div>
            <div class="metric-value">{video_meta['likes']:,}</div>
            <div class="metric-sub">{like_ratio}% Like-to-View Ratio</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sampled Comments</div>
            <div class="metric-value">{len(raw_comments):,}</div>
            <div class="metric-sub">{clean_pct}% Verified Audience</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 3px solid {status_color};">
            <div class="metric-label">Discrepancy Index</div>
            <div class="metric-value" style="color: {status_color};">{disc['score']} / 100</div>
            <div class="metric-sub" style="color: {status_color}; font-weight:600;">{disc['status']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # Visualizations: Sentiment Donut + Bigrams
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
