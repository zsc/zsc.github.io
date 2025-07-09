# -*- coding: utf-8 -*-

"""
arXiv Paper Digest Generator

This script processes a CSV file of arXiv papers, scores their relevance to a user's
academic interest using the Gemini API, translates relevant papers, and finally
generates a summary HTML report, also using the Gemini API.

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
import time
import logging
import argparse
import textwrap
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import arxiv
from google import genai

# --- Configuration ---
# Load API Key from environment variable for security
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("dotenv not installed, skipping. Ensure GEMINI_API_KEY is set in your environment.")

API_KEY = os.getenv("GEMINI_API_KEY")

# Global API Client
GEMINI_CLIENT = None

# File Paths
# Input CSV and Output HTML are now handled dynamically in the main() function.
INTEREST_FILE = Path("academic_interest.txt")

# Model Configuration
SCORING_TRANSLATION_MODEL = "gemma-3-27b-it" # Using a more powerful model for better JSON compliance and translation
HTML_GENERATION_MODEL = "gemma-3-27b-it"

# Processing Parameters
RELEVANCE_THRESHOLD = 3  # Minimum score (out of 5) to be considered relevant
MAX_PAPERS_FOR_HTML = 30 # Max papers to send for HTML generation
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 5 # seconds

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
    """
    Fetches the abstract for a given arXiv ID using the arxiv library.
    
    Args:
        arxiv_id: The ID of the paper (e.g., "2507.06221").
        
    Returns:
        The paper's abstract as a string, or an empty string if not found.
    """
    try:
        # The arxiv library expects IDs without the 'arXiv:' prefix
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
    """
    Sub-step 1: Scores, and if relevant, translates a paper using Gemini.
    
    Args:
        paper_data: A dictionary containing paper info (id, title, etc.).
        academic_interest: A string describing the user's academic interests.
        
    Returns:
        A dictionary with processed data if relevant, otherwise None.
    """
    title = paper_data.get('title', '')
    arxiv_id = paper_data.get('id', '')
    
    logging.info(f"Processing paper: {arxiv_id} - {title}")
    
    abstract = get_abstract_from_arxiv(arxiv_id)
    if not abstract:
        return None

    prompt = textwrap.dedent(f"""
        As an expert research assistant, your task is to evaluate an academic paper based on my interests, provide a relevance score, and translate its title and abstract into Chinese.

        My Academic Interests:
        ---
        {academic_interest}
        ---

        Paper to Analyze:
        - Title: "{title}"
        - Abstract: "{abstract}"

        Instructions:
        1.  **Relevance Score**: Rate the paper's relevance to my academic interests on a scale of 1 to 5, where 1 is "not relevant at all" and 5 is "highly relevant".
        2.  **Translation**: If and only if the relevance score is 3 or higher, translate the title and abstract into Simplified Chinese. Otherwise, leave the translation fields empty.
        3.  **Output Format**: Your entire response MUST be a single, valid JSON object. Do not include any text, explanations, or markdown formatting before or after the JSON.

        JSON Structure:
        {{
          "relevance_score": <integer from 1 to 5>,
          "chinese_title": "<chinese translation of the title or empty string>",
          "chinese_abstract": "<chinese translation of the abstract or empty string>"
        }}
    """)
    
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            response = GEMINI_CLIENT.models.generate_content(
                model=SCORING_TRANSLATION_MODEL,
                contents=prompt
            )
            # Clean up potential markdown code fences
            cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(cleaned_response_text)
            
            score = result.get("relevance_score", 0)
            if score >= RELEVANCE_THRESHOLD:
                processed_paper = paper_data.copy()
                processed_paper.update({
                    'score': score,
                    'abstract': abstract,
                    'chinese_title': result.get('chinese_title', ''),
                    'chinese_abstract': result.get('chinese_abstract', '')
                })
                # Ensure translation is not empty for relevant papers
                if not processed_paper['chinese_title'] or not processed_paper['chinese_abstract']:
                    logging.warning(f"Paper {arxiv_id} scored {score} but translation was empty. Skipping.")
                    return None
                logging.info(f"Paper {arxiv_id} is relevant (Score: {score}).")
                return processed_paper
            else:
                logging.info(f"Paper {arxiv_id} is not relevant (Score: {score}).")
                return None

        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            logging.error(f"Error parsing Gemini response for {arxiv_id} on attempt {attempt+1}: {e}. Response text: {response.text}")
        except Exception as e:
            logging.error(f"API call failed for {arxiv_id} on attempt {attempt+1}: {e}")
        
        time.sleep(API_RETRY_DELAY)

    logging.error(f"Failed to process paper {arxiv_id} after {API_RETRY_ATTEMPTS} attempts.")
    return None

def generate_html_report(papers: list, academic_interest: str) -> str:
    """
    Sub-step 2: Generates an HTML report from a list of papers using Gemini.
    
    Args:
        papers: A list of processed paper dictionaries.
        academic_interest: The user's academic interest for context.
        
    Returns:
        A string containing the full HTML report.
    """
    if not papers:
        return "<html><body><h1>No relevant papers found.</h1></body></html>"

    logging.info(f"Generating HTML report for {len(papers)} papers.")
    
    # Prepare data for the prompt
    papers_json_str = json.dumps(
        [{
            "english_title": p['title'],
            "chinese_title": p['chinese_title'],
            "english_abstract": p['abstract'],
            "chinese_abstract": p['chinese_abstract'],
            "abs_link": p['abs_link'],
            "pdf_link": p['pdf_link'],
            "authors": p['authors']
        } for p in papers],
        indent=2,
        ensure_ascii=False
    )
    
    prompt = textwrap.dedent(f"""
        As an expert web developer and data analyst, create a single, self-contained, and aesthetically pleasing HTML file to summarize a list of academic papers.

        My Academic Interest Profile (for context):
        ---
        {academic_interest}
        ---

        Input Data:
        A JSON list of papers is provided below. Each paper has English/Chinese titles and abstracts, authors, and links.
        ---
        {papers_json_str}
        ---

        HTML Generation Requirements:
        1.  **Language**: The entire report must be in **Simplified Chinese**.
        2.  **Overall Structure**:
            - A main title (e.g., "今日学术速递").
            - A brief introductory sentence.
            - A main table titled "综合推荐 Top 10" showing the 10 most relevant papers overall.
            - **Create 2-3 additional sections** with tables for different dimensions. Analyze the provided papers and create meaningful categories like "理论创新 Top 10", "方法论/应用 Top 10", or "交叉学科研究 Top 10". You decide the best categories based on the paper list. Each table should show up to 10 relevant papers.
        3.  **Table Format**:
            - Each table row should represent one paper.
            - **Visible Content**: The primary cell should contain the paper's title (Chinese first, then English).
            - **Links**: The title cell should be a hyperlink (`<a>`) pointing to the paper's `abs_link`. Also, provide a separate `[PDF]` link pointing to the `pdf_link`.
        4.  **Fast Hover (Tooltip) Functionality**:
            - **Crucial Requirement**: When the user's mouse hovers over a table row (`<tr>`), a custom tooltip MUST appear **instantly**.
            - **Do NOT use the default browser `title` attribute**, as its appearance is too slow.
            - **Implementation**: Use a pure CSS solution for maximum speed.
                a.  The table row (`<tr>`) should have `position: relative;` and a class, for instance, `paper-row`.
                b.  Inside one of the cells (e.g., the title cell), place a `<span>` or `<div>` with a class like `tooltip-text`. This element will hold the tooltip content.
                c.  **Tooltip Content**: The tooltip must contain: The full list of authors, the Chinese abstract, and the English abstract. Use `<br>` tags for newlines within the tooltip for better formatting.
                d.  **CSS for the Tooltip**:
                    - The `.tooltip-text` should be hidden by default (`visibility: hidden; opacity: 0;`).
                    - Position it absolutely (`position: absolute;`) with a high `z-index` (e.g., `z-index: 1;`). Give it a proper background color (e.g., `#f9f9f9`), padding, border, and `box-shadow` to make it look like a popup.
                    - Add a smooth fade-in transition: `transition: opacity 0.2s ease-in-out;`. **Do NOT add a `transition-delay`** to ensure it appears immediately.
                    - On hover (`.paper-row:hover .tooltip-text`), make the tooltip visible: `visibility: visible; opacity: 1;`.
        5.  **Styling (CSS)**:
            - Use an internal `<style>` tag in the `<head>`.
            - Choose a clean, modern, academic-friendly style. Use a sans-serif font like 'Inter', 'Helvetica Neue', or 'Arial'.
            - Style tables, headers, and links for readability. Use alternating row colors (`:nth-child(even)`) for tables.
        6.  **Final Output**:
            - The output MUST be a complete, valid HTML5 document starting with `<!DOCTYPE html>` and ending with `</html>`.
            - Do NOT include any explanations, comments, or markdown fences like ```html ... ``` outside of the HTML code itself.
    """)
    
    try:
        response = GEMINI_CLIENT.models.generate_content(
            model=HTML_GENERATION_MODEL,
            contents=prompt
        )
        # Clean up the output, removing potential markdown fences
        html_content = response.text.strip()
        if html_content.startswith("```html"):
            html_content = html_content[7:]
        if html_content.endswith("```"):
            html_content = html_content[:-3]
        return html_content.strip()
    except Exception as e:
        logging.error(f"HTML generation failed: {e}")
        return f"<html><body><h1>Error</h1><p>Failed to generate HTML report: {e}</p></body></html>"

def main(csv_input_path_str: str):
    """Main function to run the paper processing pipeline."""
    csv_input_file = Path(csv_input_path_str)
    html_output_file = csv_input_file.with_suffix('.html')

    if html_output_file.exists():
        logging.info(f"Output file '{html_output_file}' already exists. Skipping.")
        return

    try:
        setup_api()
    except ValueError as e:
        logging.error(e)
        return

    # Check for required input files
    if not csv_input_file.exists():
        logging.error(f"Input file not found: '{csv_input_file}'")
        return
    if not INTEREST_FILE.exists():
        logging.warning(f"Interest file '{INTEREST_FILE}' not found. Creating a dummy file.")
        INTEREST_FILE.write_text("My research focuses on large language models (LLMs), especially their reasoning abilities, alignment with human values, and applications in complex problem-solving. I am also interested in multi-agent systems and game theory applications in AI.")
        logging.info(f"Created dummy interest file: {INTEREST_FILE}")

    logging.info(f"--- Processing {csv_input_file} ---")
    
    # Sub-step 1: Scan CSV, score, and translate
    logging.info("--- Starting Sub-step 1: Scoring and Translating Papers ---")
    try:
        df = pd.read_csv(csv_input_file)
        academic_interest = INTEREST_FILE.read_text(encoding='utf-8')
    except Exception as e:
        logging.error(f"Error reading input files: {e}")
        return

    papers_to_process = df.to_dict('records')
    relevant_papers = []
    
    with tqdm(total=len(papers_to_process), desc="Processing papers") as pbar:
        for paper_row in papers_to_process:
            processed = score_and_translate_paper(paper_row, academic_interest)
            if processed:
                relevant_papers.append(processed)
            pbar.update(1)

    if not relevant_papers:
        logging.warning("No relevant papers found after processing. Exiting.")
        html_output_file.write_text("<html><body><h1>今日学术速递</h1><p>根据您的学术兴趣，今日没有发现相关论文。</p></body></html>", encoding='utf-8')
        return

    # Sub-step 2: Sort, filter, and generate HTML
    logging.info("--- Starting Sub-step 2: Generating HTML Report ---")
    
    # Sort by score and take the top N
    relevant_papers.sort(key=lambda x: x['score'], reverse=True)
    top_papers = relevant_papers[:MAX_PAPERS_FOR_HTML]
    
    html_content = generate_html_report(top_papers, academic_interest)
    
    html_output_file.write_text(html_content, encoding='utf-8')
    logging.info(f"Successfully generated HTML report: {html_output_file}")


# --- Unit Tests ---

class TestPaperDigestGenerator(unittest.TestCase):

    def setUp(self):
        """Set up test files and mock data."""
        self.test_csv = Path("test_日期.csv")
        self.test_interest = Path("test_academic_interest.txt")
        
        with open(self.test_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'title', 'authors', 'subjects', 'abs_link', 'pdf_link'])
            writer.writerow(['arXiv:1234.5678', 'Relevant Paper Title', 'Dr. A', 'cs.AI', 'http://abs/1', 'http://pdf/1'])
            writer.writerow(['arXiv:9876.5432', 'Irrelevant Paper Title', 'Dr. B', 'math.CO', 'http://abs/2', 'http://pdf/2'])

        self.test_interest.write_text("AI and LLMs")
        
        self.mock_academic_interest = "AI and LLMs"
        self.relevant_paper_data = {
            'id': 'arXiv:1234.5678', 'title': 'Relevant Paper Title', 'authors': 'Dr. A', 
            'subjects': 'cs.AI', 'abs_link': 'http://abs/1', 'pdf_link': 'http://pdf/1'
        }

    def tearDown(self):
        """Clean up test files."""
        if self.test_csv.exists():
            self.test_csv.unlink()
        if self.test_interest.exists():
            self.test_interest.unlink()

    @patch('__main__.arxiv.Search')
    def test_get_abstract_from_arxiv_success(self, mock_arxiv_search):
        """Test successful abstract fetching."""
        mock_result = MagicMock()
        mock_result.summary = "This is a test abstract."
        mock_arxiv_search.return_value.results.return_value = iter([mock_result])
        
        abstract = get_abstract_from_arxiv("1234.5678")
        self.assertEqual(abstract, "This is a test abstract.")

    @patch('__main__.arxiv.Search')
    def test_get_abstract_from_arxiv_not_found(self, mock_arxiv_search):
        """Test when paper is not found on arXiv."""
        mock_arxiv_search.return_value.results.return_value = iter([])
        abstract = get_abstract_from_arxiv("0000.0000")
        self.assertEqual(abstract, "")

    @patch('__main__.GEMINI_CLIENT')
    @patch('__main__.get_abstract_from_arxiv')
    def test_score_and_translate_paper_relevant(self, mock_get_abstract, mock_gemini_client):
        """Test processing a relevant paper."""
        mock_get_abstract.return_value = "An abstract about AI."
        
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "relevance_score": 5,
            "chinese_title": "相关的论文标题",
            "chinese_abstract": "关于AI的摘要。"
        })
        mock_gemini_client.models.generate_content.return_value = mock_response
        
        result = score_and_translate_paper(self.relevant_paper_data, self.mock_academic_interest)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['score'], 5)
        self.assertEqual(result['chinese_title'], "相关的论文标题")
        self.assertEqual(result['title'], "Relevant Paper Title")
        mock_get_abstract.assert_called_once_with('arXiv:1234.5678')
        mock_gemini_client.models.generate_content.assert_called_once()
        called_with_model = mock_gemini_client.models.generate_content.call_args.kwargs['model']
        self.assertEqual(called_with_model, SCORING_TRANSLATION_MODEL)

    @patch('__main__.GEMINI_CLIENT')
    @patch('__main__.get_abstract_from_arxiv')
    def test_score_and_translate_paper_irrelevant(self, mock_get_abstract, mock_gemini_client):
        """Test processing an irrelevant paper."""
        mock_get_abstract.return_value = "An abstract about combinatorics."
        
        mock_response = MagicMock()
        mock_response.text = '{"relevance_score": 1, "chinese_title": "", "chinese_abstract": ""}'
        mock_gemini_client.models.generate_content.return_value = mock_response

        irrelevant_paper_data = {'id': 'arXiv:9876.5432', 'title': 'Irrelevant Paper Title'}
        result = score_and_translate_paper(irrelevant_paper_data, self.mock_academic_interest)
        
        self.assertIsNone(result)

    @patch('__main__.GEMINI_CLIENT')
    def test_generate_html_report(self, mock_gemini_client):
        """Test HTML report generation."""
        mock_response = MagicMock()
        mock_response.text = "<!DOCTYPE html><html><body>Mocked HTML</body></html>"
        mock_gemini_client.models.generate_content.return_value = mock_response

        papers = [{
            'title': 'Test Paper', 'chinese_title': '测试论文',
            'abstract': 'Test abstract.', 'chinese_abstract': '测试摘要。',
            'score': 5, 'abs_link': 'http://abs/1', 'pdf_link': 'http://pdf/1', 'authors': 'Dr. A'
        }]
        
        html = generate_html_report(papers, self.mock_academic_interest)
        
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Mocked HTML", html)
        # Check that the prompt contains the paper data
        call_args = mock_gemini_client.models.generate_content.call_args.kwargs
        self.assertEqual(call_args['model'], HTML_GENERATION_MODEL)
        prompt_arg = call_args['contents']
        self.assertIn('"english_title": "Test Paper"', prompt_arg)
        self.assertIn('"chinese_title": "测试论文"', prompt_arg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="arXiv Paper Digest Generator.",
        formatter_class=argparse.RawTextHelpFormatter # To format help text nicely
    )
    parser.add_argument(
        'csv_file',
        nargs='?',  # Makes the argument optional
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
        # To run tests, we need to pass the script name to unittest.main
        # We remove the command-line arguments to prevent recursion.
        import sys
        sys.argv = [sys.argv[0]]
        unittest.main()
    elif args.csv_file:
        main(args.csv_file)
    else:
        print("Error: No action specified. Please provide a CSV file to process or use the --test flag.")
        parser.print_help()
