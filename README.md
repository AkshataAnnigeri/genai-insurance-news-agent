# GenAI Insurance News Agent

## Overview
This project presents a GenAI-powered intelligent news agent and interactive dashboard designed to assist insurance analysts, underwriters, and risk managers in identifying and evaluating real-time risks. It integrates Generative AI with live news scraping, NLP-based structuring, and LLM-driven enrichment to transform unstructured news content into actionable intelligence. The system classifies news articles by domain (e.g., Climate Risk, InsurTech, Regulatory Updates), extracts key metadata like location, sentiment, and financial impact, and generates analyst-ready summaries and recommendations. The insights are visualized through a user-friendly dashboard built in Streamlit, enabling professionals to monitor severity trends, geographic risk clusters, and evolving market exposures — all in real time.

## Objectives
- Scrape and structure real-time news articles related to insurance and climate risk.
- Enrich articles using NLP and LLMs to extract location, category, date, financial impact, and recommendations.
- Categorize news into actionable risk domains for analysts.
- Provide a dashboard for sentiment tracking, regional severity, top risk categories, and analyst insights.

## Tools and Technologies Used
- **Python**: Core development language.
- **LangChain + GPT-3.5 Turbo**: Used to generate structured summaries and recommendations.
- **Tavily API**: For real-time search and news scraping.
- **Spacy**: For named entity recognition (locations, organizations).
- **Streamlit**: To build the interactive analytics dashboard.
- **Plotly Express**: For visualization (Pie Charts, Maps, Bar Charts).
- **Pandas**: Data transformation and filtering.

## Project Workflow
1. **News Scraping**: Performed using Tavily Search API based on domain-specific keywords.
2. **Article Structuring**: Extracted article titles, publication dates, locations, and domains using SpaCy + custom functions.
3. **Enrichment with GenAI**: Articles are passed to GPT-3.5 via LangChain with a detailed prompt to extract summary, stakeholders, impact, sentiment, and references.
4. **Dashboard Visualization**: News is visualized by severity, domain, geography, and time using Streamlit.

## Features
- **Real-Time Querying** with dynamic filters for date range.
- **Domain Categorization**: Climate Risk, Insurance Exposure, InsurTech, and more.
- **Sentiment Mapping**: Classifies articles as Low Risk, High Risk, Critical, etc.
- **Financial Impact Summarization**: GPT highlights economic effects when present.
- **Geographic Risk Mapping**: Global scatter plots highlight regional concerns.
- **Analyst Dashboard**: Visual insights for quick decision-making.

## Sample Use Cases
- Insurance analysts monitoring climate impact.
- Underwriters assessing regulatory and financial trends.
- Reinsurers tracking global risk events by region.
- Executives exploring market exposures from real-time news.

## Status
- Completed MVP with working enrichment pipeline and dashboard.  
- Future scope includes model fine-tuning, deeper entity extraction, and integration with external insurance data APIs.

---

## Author
Akshata Gurayya Annigeri

---
**Description**: Real-time GenAI-powered news analysis tool for surfacing insurance-related climate and market risks through structured summaries and dashboard insights.

