# -*- coding: utf-8 -*-
 
"""
arXiv Paper Digest Generator

This script processes a CSV file of arXiv papers, scores their relevance,
clusters them using the Gemini API, and generates a dynamic HTML report
that visualizes these clusters.

It includes self-contained unit tests.

Setup:
1.  Install required libraries:
    pip install pandas tqdm google-generativeai arxiv python-dotenv

2.  Set up your Google AI API Key:
    Create a file named .env in the same directory and add the following line:
    GEMINI_API_KEY="YOUR_API_KEY_HERE"
    Alternatively, set it as an environment variable in your system:
    export GEMINI_API_KEY="YOUR_API_KEY"

Execution:
- To run the main process:
  python paper_digest_generator.py path/to/your_papers.csv

- To run the built-in unit tests:
  python paper_digest_generator.py --test
"""

import os
import csv
import json
import re
import time
import logging
import argparse
import textwrap
from collections import deque
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import arxiv
from google import genai


class RateLimiter:
    def __init__(self, max_requests, period_seconds):
        self.max_requests = max_requests
        self.period_seconds = period_seconds
        self.request_timestamps = deque()

    def __enter__(self):
        """进入 with 块时调用：等待一个可用的请求槽位"""
        while True:
            now = time.monotonic()
            
            while self.request_timestamps and self.request_timestamps[0] <= now - self.period_seconds:
                self.request_timestamps.popleft()

            if len(self.request_timestamps) < self.max_requests:
                break

            time_to_wait = self.request_timestamps[0] + self.period_seconds - now
            logging.debug(f"Rate limit reached ({self.max_requests}/{self.period_seconds}s), waiting {time_to_wait:.2f} seconds...")
            time.sleep(time_to_wait)
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出 with 块时调用：记录本次请求的时间戳"""
        self.request_timestamps.append(time.monotonic())
        return False

rate_limiter = RateLimiter(max_requests=30, period_seconds=60)

# --- Robust JSON Parsing Utility ---

# Characters that are plausible candidates for insertion/replacement in a broken JSON string.
PLAUSIBLE_CHARS = '"{}[],: '

def _apply_simple_fixes(json_str: str) -> tuple[str, list[str]]:
    """
    Applies a series of common, simple fixes for LLM-generated JSON.
    """
    original_str = json_str
    mods = []
    
    # 1. Extract JSON object/array from surrounding text
    # Handles markdown code blocks and conversational text.
    match = re.search(r'```(?:json)?\s*([\[{].*[\]}])\s*```', json_str, re.DOTALL)
    if match:
        json_str = match.group(1)
        mods.append("Extracted JSON from markdown code block.")
    else:
        # Fallback to finding the first and last brace/bracket
        start_brace = json_str.find('{')
        start_bracket = json_str.find('[')
        
        start_pos = -1
        
        if start_brace == -1 and start_bracket == -1:
            # No JSON structure found, nothing to extract
            pass
        elif start_brace == -1:
            start_pos = start_bracket
        elif start_bracket == -1:
            start_pos = start_brace
        else:
            start_pos = min(start_brace, start_bracket)
            
        if start_pos != -1:
            end_brace = json_str.rfind('}')
            end_bracket = json_str.rfind(']')
            end_pos = max(end_brace, end_bracket)
            
            if end_pos > start_pos:
                new_str = json_str[start_pos:end_pos+1]
                if new_str != json_str:
                    json_str = new_str
                    mods.append("Extracted content between first and last brace/bracket.")

    # 2. Remove JavaScript-style comments
    temp_str = re.sub(r'//.*', '', json_str)
    temp_str = re.sub(r'/\*.*?\*/', '', temp_str, flags=re.DOTALL)
    if temp_str != json_str:
        mods.append("Removed JavaScript-style comments.")
        json_str = temp_str

    # 3. Fix trailing commas
    temp_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    if temp_str != json_str:
        mods.append("Removed trailing commas.")
        json_str = temp_str
        
    # 4. Fix improper backslash escapes (like in LaTeX or Windows paths)
    # This regex finds a backslash that is NOT followed by a valid JSON escape char.
    # The original regex was too conservative and allowed valid but unintended
    # escapes like \t or \n. This version is more aggressive for LLM-like
    # errors, preserving only explicit `\"`, `\\`, `\/`, and `\u` sequences.
    temp_str = re.sub(r'\\(?![\\"\/u])', r'\\\\', json_str)
    if temp_str != json_str:
        mods.append("Added escaping to invalid backslashes.")
        json_str = temp_str

    # 5. Replace single quotes with double quotes (common LLM mistake)
    # This is a heuristic and might not work for strings containing apostrophes,
    # but it's a very common and simple fix.
    if "'" in json_str:
        try:
            # A more careful replacement: only for keys and values if possible
            # For simplicity, we try a global replacement and see if it parses.
            temp_str = json_str.replace("'", '"')
            json.loads(temp_str) # Check if this simple fix works
            mods.append("Replaced single quotes with double quotes.")
            json_str = temp_str
        except json.JSONDecodeError:
            # The simple replacement didn't work, we'll let the BFS handle it.
            pass

    if not mods:
        mods.append("No simple fixes applied.")
        
    return json_str, mods


def fix_invalid_json(invalid_json_str: str) -> tuple[str | None, list[str]]:
    """
    Uses pre-computation and a time-limited Breadth-First Search (BFS) to find 
    the minimum character modifications to make an invalid JSON string valid.
    """
    if not invalid_json_str or not invalid_json_str.strip():
        return "{}", ["Original string was empty, returned a valid empty object."]

    processed_str, pre_mods = _apply_simple_fixes(invalid_json_str)
    try:
        json.loads(processed_str)
        return processed_str, pre_mods
    except json.JSONDecodeError:
        if '{' not in processed_str and '[' not in processed_str:
            return None, ["Could not find a solution. The string does not appear to contain a JSON object or array."]
        pass

    start_time = time.time()
    queue = deque([(processed_str, pre_mods)])
    visited = {processed_str}
    
    max_operations = 100000 
    operation_count = 0
    deletion_only_mode = False
    
    while queue:
        elapsed_time = time.time() - start_time
        if elapsed_time > 10.0:
            return None, ["Failed to find a solution within the 10-second time limit."]
        
        if not deletion_only_mode and elapsed_time > 3.0:
            deletion_only_mode = True
            new_mods = pre_mods + ["Switched to deletion-only mode after 3s."]
            queue.clear()
            visited.clear()
            queue.append((processed_str, new_mods))
            visited.add(processed_str)
            continue

        operation_count += 1
        if operation_count > max_operations:
            return None, ["Search space too large, aborted after 100,000 states."]

        current_str, mods = queue.popleft()

        try:
            result = json.loads(current_str)
            if isinstance(result, (dict, list)):
                return current_str, mods
        except json.JSONDecodeError:
            pass

        for i in range(len(current_str)):
            next_str = current_str[:i] + current_str[i+1:]
            if next_str and next_str not in visited:
                visited.add(next_str)
                queue.append((next_str, mods + [f"Deleted '{current_str[i]}' at index {i}"]))

        if not deletion_only_mode:
            for i in range(len(current_str)):
                for char in PLAUSIBLE_CHARS:
                    if current_str[i] == char: continue
                    next_str = current_str[:i] + char + current_str[i+1:]
                    if next_str not in visited:
                        visited.add(next_str)
                        queue.append((next_str, mods + [f"Replaced '{current_str[i]}' with '{char}' at index {i}"]))
            
            for i in range(len(current_str) + 1):
                for char in PLAUSIBLE_CHARS:
                    next_str = current_str[:i] + char + current_str[i:]
                    if next_str not in visited:
                        visited.add(next_str)
                        queue.append((next_str, mods + [f"Inserted '{char}' at index {i}"]))

    return None, ["Could not find a solution by exhausting the search queue."]

# --- Configuration ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("dotenv not installed, skipping. Ensure GEMINI_API_KEY is set in your environment.")

API_KEY = os.getenv("GEMINI_API_KEY")

# Global API Client
GEMINI_CLIENT = None

# File Paths
INTEREST_FILE = Path("academic_interest.txt")

# Model Configuration
SCORING_TRANSLATION_MODEL = "gemma-3-27b-it" 
CLUSTERING_MODEL = "gemini-2.5-flash"

# Processing Parameters
RELEVANCE_THRESHOLD = 3  
MAX_PAPERS_FOR_CLUSTERING = 50 
API_RETRY_ATTEMPTS = 5
API_RETRY_DELAY = 20 # seconds

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Core Functions ---

def setup_api():
    """Configures the Google Generative AI API client."""
    global GEMINI_CLIENT
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please set it to your API key.")
    GEMINI_CLIENT = genai.Client(api_key=API_KEY)
    logging.info("Google Generative AI Client configured successfully.")

def get_abstract_from_arxiv(arxiv_id: str) -> str:
    """Fetches the abstract for a given arXiv ID."""
    try:
        clean_id = arxiv_id.replace("arXiv:", "")
        search = arxiv.Search(id_list=[clean_id])
        paper = next(search.results())
        return paper.summary.replace('\n', ' ')
    except StopIteration:
        logging.warning(f"Paper with ID {arxiv_id} not found on arXiv.")
        return ""
    except Exception as e:
        logging.error(f"Failed to fetch abstract for {arxiv_id}: {e}")
        return ""

def score_and_translate_paper(paper_data: dict, academic_interest: str) -> dict | None:
    """Scores, and if relevant, translates a paper using Gemini."""
    title = paper_data.get('title', '')
    arxiv_id = paper_data.get('id', '')
    logging.info(f"Processing paper: {arxiv_id} - {title}")
    
    abstract = get_abstract_from_arxiv(arxiv_id)
    if not abstract:
        return None

    prompt = textwrap.dedent(f"""
        As an expert research assistant, evaluate a paper based on my interests, provide a score, and translate it.

        My Academic Interests:
        ---
        {academic_interest}
        ---
        Paper:
        - Title: "{title}"
        - Abstract: "{abstract}"

        Instructions:
        1. Relevance Score: Rate relevance to my interests (1-5).
        2. Translation: If score is >= {RELEVANCE_THRESHOLD}, translate title and abstract to Simplified Chinese.
        3. Output: A single, valid JSON object. No markdown.

        JSON Structure:
        {{
          "relevance_score": <integer>,
          "chinese_title": "<chinese translation or empty string>",
          "chinese_abstract": "<chinese translation or empty string>"
        }}
    """)
    
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            with rate_limiter:
                response = GEMINI_CLIENT.models.generate_content(
                    model=SCORING_TRANSLATION_MODEL, contents=prompt)
            
            try:
                # First, try to parse with simple cleaning
                cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
                result = json.loads(cleaned_response_text)
            except json.JSONDecodeError:
                # If it fails, use the robust fixer on the original text
                logging.warning(f"Initial JSON parsing failed for {arxiv_id}. Attempting to fix...")
                fixed_json_str, mods = fix_invalid_json(response.text)
                if not fixed_json_str:
                    # Raise an error to be caught by the outer 'except' and trigger a retry
                    raise ValueError(f"JSON fixer failed to produce a result. Fix attempts: {mods}")
                logging.info(f"Successfully fixed JSON for {arxiv_id}. Modifications: {mods}")
                result = json.loads(fixed_json_str)

            score = result.get("relevance_score", 0)
            if score >= RELEVANCE_THRESHOLD:
                processed_paper = paper_data.copy()
                processed_paper.update({
                    'score': score, 'abstract': abstract,
                    'chinese_title': result.get('chinese_title', ''),
                    'chinese_abstract': result.get('chinese_abstract', '')
                })
                if not processed_paper['chinese_title'] or not processed_paper['chinese_abstract']:
                    logging.warning(f"Paper {arxiv_id} scored {score} but translation was empty. Skipping.")
                    return None
                logging.info(f"Paper {arxiv_id} is relevant (Score: {score}).")
                return processed_paper
            else:
                logging.info(f"Paper {arxiv_id} is not relevant (Score: {score}).")
                return None
        except (json.JSONDecodeError, AttributeError, KeyError, ValueError) as e:
            logging.error(f"Error parsing Gemini response for {arxiv_id} on attempt {attempt+1}: {e}. Response: {response.text}")
        except Exception as e:
            logging.error(f"API call failed for {arxiv_id} on attempt {attempt+1}: {e}")
        time.sleep(API_RETRY_DELAY * (2 ** attempt))
    logging.error(f"Failed to process paper {arxiv_id} after {API_RETRY_ATTEMPTS} attempts.")
    return None

def generate_json_clusters(papers: list, academic_interest: str) -> dict | None:
    """
    Uses an LLM to cluster papers into thematic groups based on academic interest.

    Args:
        papers: A list of processed paper dictionaries.
        academic_interest: The user's academic interest for context.

    Returns:
        A dictionary where keys are cluster titles and values are lists of arXiv IDs,
        or None on failure.
    """
    if not papers:
        return {}

    logging.info(f"Generating thematic clusters for {len(papers)} papers.")
    
    papers_json_str = json.dumps(
        [{
            "arxiv_id": p['id'],
            "title": p['title'],
            "abstract": p['abstract']
        } for p in papers],
        indent=2,
        ensure_ascii=False
    )
    
    prompt = textwrap.dedent(f"""
        As an expert research analyst, your task is to thematically cluster a list of academic papers based on my stated interests.

        My Academic Interest Profile (for context):
        ---
        {academic_interest}
        ---

        Input Data (JSON list of papers):
        ---
        {papers_json_str}
        ---

        Clustering Instructions:
        1.  **Analyze & Categorize**: Read my interest profile and the abstracts of all papers. Identify 2-4 primary thematic clusters that best summarize the collection of papers. Cluster titles should be concise and in **Simplified Chinese** (e.g., "大模型理论基础", "多智能体系统应用", "AI对齐与安全").
        2.  **Assign Papers**: Assign each paper to the most relevant cluster. A paper can belong to only one cluster.
        3.  **Rank within Clusters**: For each cluster, rank the assigned papers from most to least relevant to the cluster's theme.
        4.  **Format Output**: Your entire response MUST be a single, valid JSON object. Do not include any text, explanations, or markdown formatting before or after the JSON.
        5.  **JSON Structure**:
            - The JSON object should have cluster titles as keys.
            - The value for each key should be a list of strings.
            - Each string in the list is the `arxiv_id` of a paper.
            - The list of `arxiv_id`s must be sorted by relevance to the cluster theme (most relevant first).
            - Limit each cluster to a maximum of 10 papers.

        Example JSON Output:
        {{
          "大模型理论基础": [
            "arXiv:2507.01234",
            "arXiv:2507.05678"
          ],
          "多智能体系统应用": [
            "arXiv:2507.09999",
            "arXiv:2507.08888",
            "arXiv:2507.07777"
          ]
        }}
    """)
    
    for attempt in range(API_RETRY_ATTEMPTS):
        response_text = ""
        try:
            with rate_limiter:
                response = GEMINI_CLIENT.models.generate_content(
                    model=CLUSTERING_MODEL,
                    contents=prompt
                )
            response_text = response.text

            try:
                cleaned_response_text = response_text.strip().replace("```json", "").replace("```", "").strip()
                clusters = json.loads(cleaned_response_text)
            except json.JSONDecodeError:
                logging.warning(f"Initial cluster JSON parsing failed. Attempting to fix...")
                fixed_json_str, mods = fix_invalid_json(response_text)
                if not fixed_json_str:
                    raise ValueError(f"JSON fixer failed for clusters. Fix attempts: {mods}")
                logging.info(f"Successfully fixed cluster JSON. Modifications: {mods}")
                clusters = json.loads(fixed_json_str)
            
            # Basic validation
            if isinstance(clusters, dict) and all(isinstance(k, str) and isinstance(v, list) for k, v in clusters.items()):
                logging.info(f"Successfully generated {len(clusters)} clusters.")
                return clusters
            else:
                raise ValueError(f"Generated JSON does not match the expected structure. Got: {clusters}")

        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Error parsing or validating cluster response on attempt {attempt+1}: {e}. Response: {response_text}")
        except Exception as e:
            logging.error(f"Cluster generation API call failed on attempt {attempt+1}: {e}")
        
        time.sleep(API_RETRY_DELAY * (2 ** attempt))

    logging.error(f"Failed to generate clusters after {API_RETRY_ATTEMPTS} attempts.")
    return None

def create_dynamic_html_report(papers_data_file: Path, clusters_file: Path, output_html_file: Path):
    """
    Generates a self-contained HTML file that dynamically fetches and renders clustered paper data.

    Args:
        papers_data_file: Path to the JSON file containing all paper details.
        clusters_file: Path to the JSON file containing the cluster information.
        output_html_file: Path to write the final HTML file.
    """
    logging.info(f"Creating dynamic HTML report '{output_html_file}'...")
    
    html_template = textwrap.dedent(f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>今日学术速递</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", "Arial", sans-serif; line-height: 1.6; color: #333; background-color: #f8f9fa; margin: 0; padding: 20px; }}
            .container {{ max-width: 1000px; margin: 0 auto; background-color: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.05); }}
            h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #e9ecef; padding-bottom: 10px; }}
            h1 {{ text-align: center; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #dee2e6; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f8f9fa; }}
            tr:hover {{ background-color: #e9ecef; }}
            a {{ color: #007bff; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            .paper-row {{ position: relative; cursor: default; }}
            .tooltip-text {{ visibility: hidden; opacity: 0; width: 600px; background-color: #fff; color: #333; text-align: left; border-radius: 6px; padding: 15px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -300px; box-shadow: 0 5px 15px rgba(0,0,0,0.2); transition: opacity 0.2s ease-in-out; pointer-events: none; border: 1px solid #ccc; }}
            .paper-row:hover .tooltip-text {{ visibility: visible; opacity: 1; }}
            .paper-title {{ font-weight: bold; }}
            .authors {{ font-style: italic; color: #555; margin-top: 8px; display: block; }}
            .abstract {{ margin-top: 10px; }}
            #loader {{ text-align: center; font-size: 1.2em; padding: 50px; }}
        </style>
    </head>
    <body>
        <div class="container" id="report-container">
            <h1>今日学术速递</h1>
            <p id="intro">正在加载和渲染论文数据...</p>
            <div id="loader">
                <p>Loading...</p>
            </div>
            <div id="clusters-content"></div>
        </div>

        <script>
            document.addEventListener('DOMContentLoaded', () => {{
                const papersFile = '{papers_data_file.name}';
                const clustersFile = '{clusters_file.name}';
                const container = document.getElementById('clusters-content');
                const loader = document.getElementById('loader');
                const intro = document.getElementById('intro');

                Promise.all([
                    fetch(papersFile).then(res => res.json()),
                    fetch(clustersFile).then(res => res.json())
                ])
                .then(([papersData, clustersData]) => {{
                    loader.style.display = 'none';
                    if (Object.keys(clustersData).length === 0) {{
                        intro.textContent = '今日未发现相关主题的论文分组。';
                        return;
                    }}
                    
                    const papersMap = new Map(papersData.map(p => [p.id, p]));
                    intro.textContent = `根据您的学术兴趣，今日的论文已为您整理成以下 ${'{Object.keys(clustersData).length}'} 个主题。将鼠标悬停在论文标题上可查看详细摘要。`;

                    for (const [clusterTitle, paperIds] of Object.entries(clustersData)) {{
                        const section = document.createElement('section');
                        const title = document.createElement('h2');
                        title.textContent = clusterTitle;
                        section.appendChild(title);

                        const table = document.createElement('table');
                        const thead = `<thead><tr><th>标题 (Title)</th><th>操作 (Actions)</th></tr></thead>`;
                        table.innerHTML = thead;
                        const tbody = document.createElement('tbody');

                        paperIds.forEach(paperId => {{
                            const paper = papersMap.get(paperId);
                            if (!paper) return;

                            const row = document.createElement('tr');
                            row.className = 'paper-row';
                            
                            row.innerHTML = `
                                <td>
                                    <div class="paper-title">${'{paper.chinese_title}'}</div>
                                    <div>${'{paper.title}'}</div>
                                    <div class="tooltip-text">
                                        <strong>作者 (Authors):</strong>
                                        <div class="authors">${'{paper.authors}'}</div>
                                        <strong style="margin-top:10px; display:block;">中文摘要 (Abstract_CN):</strong>
                                        <div class="abstract">${'{paper.chinese_abstract}'}</div>
                                    </div>
                                </td>
                                <td>
                                    <a href="${'{paper.abs_link}'}" target="_blank">Abs</a> |
                                    <a href="${'{paper.pdf_link}'}" target="_blank">PDF</a>
                                </td>
                            `;
                            tbody.appendChild(row);
                        }});
                        table.appendChild(tbody);
                        section.appendChild(table);
                        container.appendChild(section);
                    }}
                }})
                .catch(error => {{
                    loader.style.display = 'none';
                    intro.textContent = '加载论文数据失败。请检查文件是否存在以及网络连接。';
                    console.error('Error loading data:', error);
                }});
            }});
        </script>
    </body>
    </html>
    """)
    output_html_file.write_text(html_template, encoding='utf-8')
    logging.info(f"Successfully created dynamic HTML report: {output_html_file}")


def main(csv_input_path_str: str):
    """Main function to run the paper processing pipeline."""
    csv_input_file = Path(csv_input_path_str)
    json_cache_file = csv_input_file.with_suffix('.json')
    cluster_json_file = csv_input_file.with_name(f"clusters_{json_cache_file.name}")
    html_output_file = csv_input_file.with_suffix('.html')

    if html_output_file.exists():
        logging.info(f"Output file '{html_output_file}' already exists. Skipping.")
        return

    try:
        setup_api()
    except ValueError as e:
        logging.error(e)
        return

    if not csv_input_file.exists():
        logging.error(f"Input file not found: '{csv_input_file}'"); return
    if not INTEREST_FILE.exists():
        logging.warning(f"Interest file '{INTEREST_FILE}' not found. Creating a dummy file.")
        INTEREST_FILE.write_text("My research focuses on large language models (LLMs), especially their reasoning abilities, alignment with human values, and applications in complex problem-solving. I am also interested in multi-agent systems and game theory applications in AI.")

    logging.info(f"--- Processing {csv_input_file} ---")
    academic_interest = INTEREST_FILE.read_text(encoding='utf-8')
    relevant_papers = []

    if json_cache_file.exists():
        logging.info(f"Found existing JSON cache '{json_cache_file}'. Loading relevant papers.")
        try:
            with open(json_cache_file, 'r', encoding='utf-8') as f:
                relevant_papers = json.load(f)
            logging.info(f"Successfully loaded {len(relevant_papers)} papers from cache.")
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Error reading JSON cache '{json_cache_file}': {e}. Reprocessing from CSV.")
            relevant_papers = []

    if not relevant_papers:
        logging.info("--- Starting Sub-step 1: Scoring and Translating Papers ---")
        try:
            df = pd.read_csv(csv_input_file)
            papers_to_process = df.to_dict('records')
        except Exception as e:
            logging.error(f"Error reading input CSV file: {e}"); return

        with tqdm(total=len(papers_to_process), desc="Processing papers") as pbar:
            for paper_row in papers_to_process:
                processed = score_and_translate_paper(paper_row, academic_interest)
                if processed:
                    relevant_papers.append(processed)
                pbar.update(1)

        if relevant_papers:
            logging.info(f"Saving {len(relevant_papers)} relevant papers to cache: {json_cache_file}")
            with open(json_cache_file, 'w', encoding='utf-8') as f:
                json.dump(relevant_papers, f, indent=2, ensure_ascii=False)

    if not relevant_papers:
        logging.warning("No relevant papers found. Exiting.")
        html_output_file.write_text("<html><body><h1>今日学术速递</h1><p>根据您的学术兴趣，今日没有发现相关论文。</p></body></html>", encoding='utf-8')
        return

    logging.info("--- Starting Sub-step 2: Clustering Papers ---")
    clusters = None
    if cluster_json_file.exists():
        logging.info(f"Found existing cluster file '{cluster_json_file}'. Loading clusters.")
        with open(cluster_json_file, 'r', encoding='utf-8') as f:
            clusters = json.load(f)
    else:
        relevant_papers.sort(key=lambda x: x['score'], reverse=True)
        papers_for_clustering = relevant_papers[:MAX_PAPERS_FOR_CLUSTERING]
        clusters = generate_json_clusters(papers_for_clustering, academic_interest)
        if clusters:
            logging.info(f"Saving {len(clusters)} clusters to: {cluster_json_file}")
            with open(cluster_json_file, 'w', encoding='utf-8') as f:
                json.dump(clusters, f, indent=2, ensure_ascii=False)
    
    if not clusters:
        logging.error("Failed to generate or load paper clusters. Cannot create HTML report.")
        return

    logging.info('Pipeline complete.')
    # can be shared, comment out
    # logging.info("--- Starting Sub-step 3: Generating Dynamic HTML Report ---")
    # create_dynamic_html_report(json_cache_file, cluster_json_file, html_output_file) 
    # logging.info(f"Pipeline complete. Report generated at: {html_output_file}")


# --- Unit Tests ---

class TestPaperDigestGenerator(unittest.TestCase):

    def setUp(self):
        """Set up test files and mock data."""
        self.test_dir = Path("test_output")
        self.test_dir.mkdir(exist_ok=True)
        
        self.test_csv = self.test_dir / "test_papers.csv"
        self.test_interest = Path("academic_interest.txt") # Keep this in root for main()
        
        self.papers_data = [
            {'id': 'arXiv:1234.5678', 'title': 'Relevant Paper 1', 'authors': 'Dr. A', 'subjects': 'cs.AI', 'abs_link': 'http://abs/1', 'pdf_link': 'http://pdf/1'},
            {'id': 'arXiv:9876.5432', 'title': 'Irrelevant Paper', 'authors': 'Dr. B', 'subjects': 'math.CO', 'abs_link': 'http://abs/2', 'pdf_link': 'http://pdf/2'},
            {'id': 'arXiv:1111.2222', 'title': 'Relevant Paper 2', 'authors': 'Dr. C', 'subjects': 'cs.AI', 'abs_link': 'http://abs/3', 'pdf_link': 'http://pdf/3'}
        ]
        
        with open(self.test_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.papers_data[0].keys())
            writer.writeheader()
            writer.writerows(self.papers_data)

        self.test_interest.write_text("AI and LLMs")
        self.mock_academic_interest = "AI and LLMs"

    def tearDown(self):
        """Clean up test files."""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        if self.test_interest.exists():
            # In a real scenario, you might not want to delete this, but for testing it's clean
            self.test_interest.unlink()

    @patch('__main__.arxiv.Search')
    def test_get_abstract_from_arxiv_success(self, mock_arxiv_search):
        mock_result = MagicMock()
        mock_result.summary = "This is a test abstract."
        mock_arxiv_search.return_value.results.return_value = iter([mock_result])
        abstract = get_abstract_from_arxiv("1234.5678")
        self.assertEqual(abstract, "This is a test abstract.")

    @patch('__main__.GEMINI_CLIENT')
    @patch('__main__.get_abstract_from_arxiv')
    def test_score_and_translate_paper_relevant(self, mock_get_abstract, mock_gemini_client):
        mock_get_abstract.return_value = "An abstract about AI."
        mock_response = MagicMock()
        mock_response.text = json.dumps({"relevance_score": 5, "chinese_title": "相关标题", "chinese_abstract": "相关摘要"})
        mock_gemini_client.models.generate_content.return_value = mock_response
        result = score_and_translate_paper(self.papers_data[0], self.mock_academic_interest)
        self.assertIsNotNone(result)
        self.assertEqual(result['score'], 5)
        self.assertEqual(result['chinese_title'], "相关标题")

    @patch('__main__.GEMINI_CLIENT')
    def test_generate_json_clusters(self, mock_gemini_client):
        """Test thematic cluster generation."""
        mock_response = MagicMock()
        mock_cluster_data = {
          "大模型理论基础": ["arXiv:1234.5678"],
          "AI应用": ["arXiv:1111.2222"]
        }
        mock_response.text = json.dumps(mock_cluster_data)
        mock_gemini_client.models.generate_content.return_value = mock_response

        # Sample processed papers
        processed_papers = [
            {'id': 'arXiv:1234.5678', 'title': 'Paper A', 'abstract': '...'},
            {'id': 'arXiv:1111.2222', 'title': 'Paper B', 'abstract': '...'}
        ]
        
        clusters = generate_json_clusters(processed_papers, self.mock_academic_interest)
        
        self.assertIsNotNone(clusters)
        self.assertEqual(clusters, mock_cluster_data)
        self.assertIn("大模型理论基础", clusters)
        self.assertEqual(clusters["大模型理论基础"], ["arXiv:1234.5678"])

    def test_create_dynamic_html_report(self):
        """Test the creation of the dynamic HTML file."""
        papers_data_file = self.test_dir / "test_papers.json"
        clusters_file = self.test_dir / "test_clusters.json"
        output_html_file = self.test_dir / "report.html"

        # Create dummy data files
        dummy_papers = [{"id": "arXiv:1234.5678", "title": "Test Paper", "chinese_title": "测试论文", "authors": "Dr. Test", "chinese_abstract": "摘要", "abstract": "Abstract", "abs_link": "abs", "pdf_link": "pdf"}]
        dummy_clusters = {"测试主题": ["arXiv:1234.5678"]}
        
        with open(papers_data_file, 'w', encoding='utf-8') as f:
            json.dump(dummy_papers, f)
        with open(clusters_file, 'w', encoding='utf-8') as f:
            json.dump(dummy_clusters, f)

        # Run the function
        create_dynamic_html_report(papers_data_file, clusters_file, output_html_file)

        # Check the output
        self.assertTrue(output_html_file.exists())
        html_content = output_html_file.read_text(encoding='utf-8')
        
        self.assertIn('<!DOCTYPE html>', html_content)
        self.assertIn('<div id="clusters-content">', html_content)
        # Check if the JS fetches the correct filenames
        self.assertIn(f"const papersFile = '{papers_data_file.name}';", html_content)
        self.assertIn(f"const clustersFile = '{clusters_file.name}';", html_content)
        self.assertIn('Promise.all([', html_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="arXiv Paper Digest Generator.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'csv_file',
        nargs='?',
        default=None,
        help="Path to the input CSV file to be processed."
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run the built-in unit tests instead of the main script.'
    )
    args = parser.parse_args()

    if args.test:
        print("--- Running Unit Tests ---")
        import sys
        sys.argv = [sys.argv[0]]
        unittest.main()
    elif args.csv_file:
        main(args.csv_file)
    else:
        print("Error: No action specified. Please provide a CSV file to process or use the --test flag.")
        parser.print_help()
