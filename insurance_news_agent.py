#!/usr/bin/env python
# coding: utf-8

# climate_ai_agent.py

import os
import json
import html
import re
import spacy
import unicodedata
from datetime import datetime, timedelta
from urllib.parse import urlparse
from collections import Counter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
dotenv_path = os.path.expanduser("~/.env")
load_dotenv(dotenv_path)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Load SpaCy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Initialize LangChain Components
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", api_key=OPENAI_API_KEY, temperature=0.4)
tool = TavilySearchResults(max_results=10, api_key=TAVILY_API_KEY)

# Define Keywords
all_keywords = ["climate change", "insurance market", "reinsurance", "insurtech", "loss ratio", "insurance regulation"]
trusted_sources = {"TNFD": ["tnfd.global"], "IPCC": ["ipcc.ch"], "Swiss Re": ["swissre.com"]}

# Build search query
search_query = " OR ".join(all_keywords)

def enrich_articles_with_research_references(articles):
    enriched = []
    for article in articles:
        references = []
        for org, domains in trusted_sources.items():
            if any(domain in article["url"] for domain in domains):
                references.append(org)
        article["research_references"] = references if references else ["None"]
        enriched.append(article)
    return enriched


def safe_parse_response(response):
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            return []
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        return response.get("results", [])
    return []

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = unicodedata.normalize("NFKD", text)
    text = ''.join(ch for ch in text if ch.isprintable())
    return re.sub(r'\s+', ' ', text).strip()

def extract_domain_as_source(url):
    return urlparse(url).netloc.replace("www.", "")

def extract_location(text):
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
    return Counter(locations).most_common(1)[0][0] if locations else "Unknown Location"

def categorize_article(text):
    text_lower = text.lower()
    categories = {"Climate Risk": ["climate change"], "Insurance Exposures": ["insured loss"], "InsurTech": ["insurtech"]}
    for category, keywords in categories.items():
        if any(k in text_lower for k in keywords):
            return category
    return "Uncategorized"

def extract_best_date(url, title, content):
    patterns = [r'(20\d{2})[-/](\d{2})[-/](\d{2})', r'(\d{1,2} \w+ 20\d{2})', r'(\w+ \d{1,2}, 20\d{2})']
    for text in [url, title, content]:
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return datetime.strptime(match.group(0), "%Y-%m-%d").strftime("%Y-%m-%d")
                except:
                    continue
    return datetime.today().strftime("%Y-%m-%d")

def structure_articles(response):
    structured = []
    raw_results = safe_parse_response(response)
    print(f"Parsed {len(raw_results)} results.")
    for result in raw_results:
        url = result.get("url", "")
        title = result.get("title", "") or os.path.basename(url) or "No Title"
        source = result.get("source", "") or extract_domain_as_source(url)
        raw_content = result.get("content", "")
        full_content = clean_text(raw_content)
        summary_input = " ".join(full_content.split()[:800])
        date = extract_best_date(url, title, full_content)
        location = extract_location(full_content)
        category = categorize_article(full_content)
        structured.append({"title": title, "url": url, "source": source, "date": date, "location": location, "category": category, "summary_input": summary_input, "full_content": full_content})
    return structured
def enrich_article_with_gpt_full(article):
    prompt = f"""
You are an expert AI analyst at a global insurance firm. Your task is to analyze the provided article and clearly summarize it into an insightful, detailed, and highly structured summary explicitly crafted for risk analysts and decision-makers in the insurance industry.

Explicitly include the following critical information in your summary clearly and concisely:

1. **Main Issue/Event**: Clearly describe what specifically happened (e.g., wildfire, flood, policy change, regulation update, insurtech development).

2. **Date and Location**: Clearly state when and where exactly the issue/event occurred or will occur.

3. **Key Stakeholders**: Clearly mention any important companies, organizations, governments, or individuals involved or affected directly.

4. **Impacts on Insurance Business**:
   - Describe clearly any direct or indirect effects on the insurance and reinsurance industry.
   - Clearly indicate potential changes in underwriting practices or risks to exposure.

5. **Regulatory or Policy Changes**: Clearly state any significant policy, regulatory, or legislative developments described in the article.

6. **Financial and Economic Implications**:
   - Clearly summarize any explicitly mentioned or implied financial losses, premiums adjustments, claims impacts, or economic repercussions.

7. **Technological Implications (if mentioned)**:
   - Describe clearly how technology or innovation (e.g., AI, IoT, blockchain, etc.) is involved or impacted.

8. **Environmental and Social Impact (if applicable)**:
   - Clearly summarize any mentioned environmental or social consequences.

9. **Immediate Action Recommendations**: Clearly and explicitly provide an actionable recommendation for insurance analysts to manage or monitor the described risk or situation.

10. **Explicitly Suggested External References**:
   - Clearly list relevant external references, explicitly providing both the name and the complete URL to official reports, trusted documents, or authoritative sources.

Structure your output clearly and explicitly as this JSON format:

{{
  "title": "...",
  "url": "...",
  "date": "...",
  "source": "...",
  "category": "...",
  "location": "...",
  "summary": "...",  # Clearly structured and comprehensive covering above points.
  "references": [
    {{"name": "IPCC", "url": "https://ipcc.ch/report.pdf"}},
    {{"name": "TNFD", "url": "https://tnfd.global/report.pdf"}}
  ],
  "sentiment": "...",
  "recommendation": "...",
  "financial_impact": "..."
}}

Clearly use only the provided structured article content explicitly below to craft your response:

Title: {article.get("title", "")}
URL: {article.get("url", "")}
Date: {article.get("date", "")}
Source: {article.get("source", "")}
Summary: {article.get("summary_input", "")}
Content: {article.get("full_content", "")[:3000]}
"""
    try:
        response = llm.invoke(prompt)
        return json.loads(response.content)
    except Exception as e:
        print("LangChain GPT Error:", e)
        return {
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "date": article.get("date", ""),
            "source": article.get("source", ""),
            "category": "Uncategorized",
            "summary": "GPT failed.",
            "references": [],
            "sentiment": "Unknown",
            "recommendation": "N/A",
            "financial_impact": "Not specified"
        }



def main():
    today_str = datetime.today().strftime("%B %d, %Y")
    yesterday_str = (datetime.today() - timedelta(days=1)).strftime("%B %d, %Y")
    query = f"({search_query}) AND ({today_str} OR {yesterday_str} OR this week)"
    print("Running query:", query)
    try:
        response = tool.invoke({"query": query})
        print("Raw response:", json.dumps(response, indent=2) if isinstance(response, dict) else response)
    except Exception as e:
        print("Tavily invoke failed:", e)
        return

    structured = structure_articles(response)
    if not structured:
        print("❌ No structured articles were found.")
        return

    for article in structured:
        print(f"\n✅ {article['title']}\nDate: {article['date']}\nCategory: {article['category']}")

if __name__ == "__main__":
    main()