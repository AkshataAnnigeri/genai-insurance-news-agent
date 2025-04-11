#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import plotly.express as px
from insurance_news_agent import (
    tool,
    structure_articles,
    enrich_articles_with_research_references,
    enrich_article_with_gpt_full,
    search_query
)

# ✅ Safe enrichment with error handling
def get_enriched_articles():
    try:
        response = tool.invoke({"query": search_query})
        if not response:
            st.warning("⚠️ Tavily returned no response.")
            return []

        print("✅ Tavily response received.")
        structured = structure_articles(response)
        if not structured:
            st.warning("⚠️ No articles were structured from the response.")
            return []

        enriched_with_refs = enrich_articles_with_research_references(structured)
        enriched_articles = [enrich_article_with_gpt_full(a) for a in enriched_with_refs]

        print(f"✅ Enriched {len(enriched_articles)} articles.")
        return enriched_articles

    except Exception as e:
        st.error(f"❌ Error while loading articles: {e}")
        return []

# --- Page Setup ---
st.set_page_config(page_title="Insurance Information AI Agent", layout="wide")
st_autorefresh(interval=60 * 60 * 1000)

# --- Load Articles with Caching ---
@st.cache_data(ttl=3600)
def load_articles():
    articles = get_enriched_articles()
    if not articles:
        return pd.DataFrame()  # Return empty DataFrame if nothing found
    return pd.DataFrame(articles)

df = load_articles()

if df.empty:
    st.error("❌ No articles found. Try refreshing later or check your query settings.")
    st.stop()

# --- Preprocess Dates and Sentiment ---
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

sentiment_remap = {
    "Negative": "High Risk",
    "Positive": "Low Risk",
    "Neutral": "Neutral"
}
df['sentiment'] = df['sentiment'].map(sentiment_remap).fillna("Neutral")

# --- Filter for Today by Default ---
default_start = pd.Timestamp.now().normalize()
default_df = df[df['date'] >= default_start]

# --- Sidebar Filters ---
st.sidebar.title("Navigation")
time_options = {
    "Today": 0,
    "Last 1 Hour": 1,
    "Last 12 Hours": 12,
    "Last 24 Hours": 24,
    "Last 2 Days": 48,
    "Last 5 Days": 120,
    "Last 1 Month": 720
}
selected_period = st.sidebar.selectbox("Select Period", list(time_options.keys()), index=0)
hours = time_options[selected_period]

if selected_period == "Today":
    df_filtered = default_df.copy()
else:
    threshold = pd.Timestamp.now() - pd.Timedelta(hours=hours)
    df_filtered = df[df['date'] >= threshold].copy()

# --- Icons & Colors ---
severity_icons = {
    "Critical": "🔴", "High Risk": "🟠", "Caution": "🟡", "Moderate": "🟡",
    "Low Risk": "🟢", "Neutral": "🔵"
}
severity_colors = {
    "Critical": "red", "High Risk": "orange", "Caution": "yellow", "Moderate": "yellow",
    "Low Risk": "green", "Neutral": "blue"
}

# --- Main View ---
st.title(f"Articles from {selected_period}")
if 'selected_article' not in st.session_state:
    st.session_state.selected_article = None

if df_filtered.empty:
    st.warning("No articles found for the selected period.")
else:
    for idx, row in df_filtered.iterrows():
        icon = severity_icons.get(row['sentiment'], "📰")
        btn_label = f"{icon} {row['title']} ({row['date'].strftime('%Y-%m-%d')})"
        if st.button(btn_label, key=f"article_{idx}"):
            st.session_state.selected_article = row

    # --- Article Details ---
    if st.session_state.selected_article is not None:
        article = st.session_state.selected_article
        st.markdown("---")
        st.header(f"{article['title']}")
        st.markdown(f"**Source:** [{article['source']}]({article['url']})")
        st.markdown(f"**Date:** {article['date'].strftime('%Y-%m-%d')}")
        st.markdown(f"**Location:** {article['location']}")
        st.markdown(f"**Category:** {article['category']} | **Severity:** {article['sentiment']}")
        st.markdown("---")
        st.subheader("Summary")
        st.write(article["summary"])
        st.subheader("Financial Impact")
        st.write(article["financial_impact"])
        st.subheader("Recommendation")
        st.write(article["recommendation"])
        if article.get("references"):
            st.subheader("References")
            for ref in article["references"]:
                st.markdown(f"- [{ref['name']}]({ref['url']})")

# --- Analytics ---
with st.expander("📊 Analytics Overview", expanded=False):
    st.subheader("Trending News (High Severity)")
    trending = df_filtered[df_filtered['sentiment'].isin(["High Risk", "Critical"])].head(5)
    for _, row in trending.iterrows():
        icon = severity_icons.get(row['sentiment'], "📰")
        st.write(f"{icon} {row['title']} ({row['date'].strftime('%Y-%m-%d')})")

    st.subheader("Sentiment Distribution")
    sentiment_counts = df_filtered['sentiment'].value_counts()
    if not sentiment_counts.empty:
        pie_fig = px.pie(
            names=sentiment_counts.index,
            values=sentiment_counts.values,
            color=sentiment_counts.index,
            color_discrete_map=severity_colors,
            title="Sentiment Distribution"
        )
        st.plotly_chart(pie_fig, use_container_width=True)

    st.subheader("Top Risk Categories")
    cat_counts = df_filtered['category'].value_counts()
    if not cat_counts.empty:
        st.bar_chart(cat_counts)

    st.subheader("Severity Map")
    map_df = df_filtered.groupby(['location', 'sentiment'], as_index=False).size()
    map_df["color"] = map_df["sentiment"].map(severity_colors)
    if not map_df.empty:
        map_fig = px.scatter_geo(
            map_df,
            locations="location",
            locationmode='country names',
            color="sentiment",
            size="size",
            hover_name="location",
            color_discrete_map=severity_colors,
            projection="natural earth",
            title="News Severity by Region"
        )
        st.plotly_chart(map_fig, use_container_width=True)

    st.subheader("Analyst Recommendations")
    critical_regions = map_df[map_df['sentiment'] == "Critical"]['location'].tolist()
    if critical_regions:
        st.error(f"Immediate attention needed on: {', '.join(critical_regions)}")
    else:
        st.success("✅ No immediate critical alerts.")

# --- Sidebar Info ---
st.sidebar.markdown("### About This Agent")
st.sidebar.info(
    "Real-time news insights tailored for insurance analysts. "
    "Includes article summaries, sentiment, maps, and analyst recommendations."
)
