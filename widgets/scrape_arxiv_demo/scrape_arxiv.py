#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scrape_arxiv.py

A script to scrape the most recent day's submissions from the arXiv cs.AI category
and save them as a terse, information-rich CSV file named after the date.

This script also includes self-contained unit tests.

Dependencies:
- requests
- beautifulsoup4

Install them using pip:
pip install requests beautifulsoup4

Usage:
1. To run the scraper:
   python scrape_arxiv.py

2. To run the built-in unit tests:
   python scrape_arxiv.py test
"""

import sys
import unittest
import requests
from bs4 import BeautifulSoup
import os
import tempfile
import csv
import datetime

# --- Configuration ---
URL = "https://arxiv.org/list/cs.AI/recent?skip=0&show=2000"
FALLBACK_FILENAME = "arxiv_cs_ai_latest.csv"

# --- Core Scraper Logic ---

def fetch_page_content(url: str) -> str:
    """Fetches the HTML content of a given URL."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}", file=sys.stderr)
        sys.exit(1)

def parse_latest_articles(html_content: str) -> tuple[str, list[dict]]:
    """
    Parses HTML to find the most recent submission date and all associated articles.
    
    The structure is assumed to be:
    <dl id='articles'>
        <h3>Date Heading</h3>
        <dt>Article 1 Info</dt>
        <dd>Article 1 Details</dd>
        ...
        <h3>Older Date Heading</h3>
        ...
    </dl>
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    articles = []

    # Find the main <dl> container for all articles
    articles_dl = soup.find('dl', id='articles')
    if not articles_dl:
        return "No article list (<dl id='articles'>) found", []

    # Find the first h3, which corresponds to the latest date
    first_h3 = articles_dl.find('h3')
    if not first_h3:
        return "No date heading (<h3>) found in article list", []
    
    date_str = first_h3.get_text(strip=True)

    # Iterate through the siblings of the first h3 to find dt/dd pairs
    # belonging to the most recent day.
    for tag in first_h3.find_next_siblings():
        # Stop when we encounter the next date heading
        if tag.name == 'h3':
            break
        
        # We only care about <dt> tags, as they mark the start of an entry
        if tag.name == 'dt':
            dt = tag
            # The corresponding <dd> should be the very next sibling
            dd = dt.find_next_sibling('dd')
            
            if not dd:
                continue # Skip if <dt> has no corresponding <dd>

            # --- Extraction logic ---
            arxiv_id_tag = dt.find('a', title='Abstract')
            arxiv_id = arxiv_id_tag.get_text(strip=True) if arxiv_id_tag else "N/A"
            abs_link = arxiv_id_tag['href'] if arxiv_id_tag else "#"

            links = {
                link.get_text(strip=True): link['href']
                for link in dt.find_all('a')
                if link.get('href')
            }
            
            title = dd.find('div', class_='list-title').get_text(strip=True).replace('Title:', '')
            authors = [a.get_text(strip=True) for a in dd.find('div', class_='list-authors').find_all('a')]
            subjects = dd.find('div', class_='list-subjects').get_text(strip=True).replace('Subjects:', '')

            articles.append({
                'id': arxiv_id,
                'abs_link': f"https://arxiv.org{abs_link}",
                'title': title,
                'authors': ", ".join(authors),
                'subjects': subjects,
                'links': links
            })
            
    return date_str, articles

def generate_csv_output(articles: list[dict], filename: str) -> None:
    """Generates a CSV file from the scraped article data."""
    headers = ['id', 'title', 'authors', 'subjects', 'abs_link', 'pdf_link']
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            for article in articles:
                # Construct the full PDF link from the relative path
                pdf_relative_link = article.get('links', {}).get('pdf', '')
                pdf_full_link = f"https://arxiv.org{pdf_relative_link}" if pdf_relative_link else ""
                
                row = {
                    'id': article.get('id'),
                    'title': article.get('title'),
                    'authors': article.get('authors'),
                    'subjects': article.get('subjects'),
                    'abs_link': article.get('abs_link'),
                    'pdf_link': pdf_full_link
                }
                writer.writerow(row)
        
        # Suppress print statement during testing
        if 'unittest' not in sys.modules:
            print(f"Successfully generated CSV file: '{filename}'")
    except IOError as e:
        print(f"Error writing to file {filename}: {e}", file=sys.stderr)
        sys.exit(1)

def run_scraper():
    """Main function to orchestrate the scraping process."""
    print(f"Fetching data from {URL}...")
    html_content = fetch_page_content(URL)
    
    print("Parsing latest articles...")
    date_str, articles = parse_latest_articles(html_content)
    
    if not articles:
        print(f"No articles found for the latest date ({date_str}). Exiting.")
        return
        
    print(f"Found {len(articles)} articles for date: {date_str}")
    
    # Generate filename from date, e.g., "YYYY-MM-DD.csv"
    try:
        # Extract date part, e.g., "9 Jul 2025" from "Wed, 9 Jul 2025 (...)"
        date_part_str = date_str.split('(')[0].strip().split(', ')[1]
        date_obj = datetime.datetime.strptime(date_part_str, "%d %b %Y")
        filename = date_obj.strftime("%Y-%m-%d") + ".csv"
    except (IndexError, ValueError):
        print(f"Could not parse date '{date_str}'. Using fallback filename.", file=sys.stderr)
        filename = FALLBACK_FILENAME

    print(f"Generating CSV output at '{filename}'...")
    generate_csv_output(articles, filename)

# --- Unit Tests ---

class TestArxivScraper(unittest.TestCase):
    
    def setUp(self):
        """Set up test data reflecting the real page structure."""
        self.sample_html = """
        <!DOCTYPE html>
        <html>
        <body>
          <div id='dlpage'>
            <dl id='articles'>
              <h3>Wed, 9 Jul 2025 (showing 2 of 2 entries )</h3>
              <dt>
                <a name='item1'>[1]</a>
                <a href="/abs/2507.06221" title="Abstract">arXiv:2507.06221</a>
                [<a href="/pdf/2507.06221" title="Download PDF">pdf</a>]
              </dt>
              <dd>
                <div class='meta'>
                  <div class='list-title mathjax'><span class='descriptor'>Title:</span>First Article Title</div>
                  <div class='list-authors'><a href="#">Author A</a>, <a href="#">Author B</a></div>
                  <div class='list-subjects'><span class='descriptor'>Subjects:</span><span class="primary-subject">AI</span></div>
                </div>
              </dd>
              <dt>
                <a name='item2'>[2]</a>
                <a href="/abs/2507.06222" title="Abstract">arXiv:2507.06222</a>
                [<a href="/pdf/2507.06222" title="Download PDF">pdf</a>]
              </dt>
              <dd>
                <div class='meta'>
                  <div class='list-title mathjax'><span class='descriptor'>Title:</span>Second Article Title</div>
                  <div class='list-authors'><a href="#">Author C</a></div>
                  <div class='list-subjects'><span class='descriptor'>Subjects:</span> AI; Computation and Language (cs.CL)</div>
                </div>
              </dd>
              <h3>Tue, 8 Jul 2025 (showing 1 of 1 entries)</h3>
              <dt>
                <a href="/abs/2507.01111" title="Abstract">arXiv:2507.01111</a>
              </dt>
              <dd>
                <div class='list-title'><span class='descriptor'>Title:</span>Older Article</div>
              </dd>
            </dl>
          </div>
        </body>
        </html>
        """
        self.mock_articles = [
            {
                'id': 'arXiv:1234.56789',
                'abs_link': 'https://arxiv.org/abs/1234.56789',
                'title': 'Test Title, with a Comma',
                'authors': 'John Doe, Jane Smith',
                'subjects': 'Artificial Intelligence (cs.AI)',
                'links': {'pdf': '/pdf/1234.56789'}
            }
        ]

    def test_parse_latest_articles(self):
        """Test the core parsing logic on sample HTML, ensuring it stops at the next date."""
        date_str, articles = parse_latest_articles(self.sample_html)
        
        self.assertEqual(date_str, "Wed, 9 Jul 2025 (showing 2 of 2 entries )")
        self.assertEqual(len(articles), 2, "Should only find 2 articles from the latest day")
        
        # Check first article
        article1 = articles[0]
        self.assertEqual(article1['id'], "arXiv:2507.06221")
        self.assertEqual(article1['title'], "First Article Title")
        self.assertEqual(article1['authors'], "Author A, Author B")
        self.assertEqual(article1['abs_link'], "https://arxiv.org/abs/2507.06221")
        self.assertIn('pdf', article1['links'])
        self.assertEqual(article1['links']['pdf'], '/pdf/2507.06221')

        # Check second article
        article2 = articles[1]
        self.assertEqual(article2['id'], "arXiv:2507.06222")
        self.assertEqual(article2['title'], "Second Article Title")
        self.assertEqual(article2['authors'], "Author C")
        self.assertEqual(article2['subjects'], "AI; Computation and Language (cs.CL)")

    def test_parse_no_articles(self):
        """Test parsing HTML where the main article list is missing."""
        html_without_dl = "<h3>A Date</h3><p>Some text</p>"
        date_str, articles = parse_latest_articles(html_without_dl)
        self.assertEqual(date_str, "No article list (<dl id='articles'>) found")
        self.assertEqual(len(articles), 0)

    def test_generate_csv_output(self):
        """Test the CSV file generation."""
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8', suffix=".csv") as tmp:
            filename = tmp.name
        
        try:
            generate_csv_output(self.mock_articles, filename)
            
            with open(filename, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                lines = list(reader)
            
            expected_header = ['id', 'title', 'authors', 'subjects', 'abs_link', 'pdf_link']
            expected_row = [
                'arXiv:1234.56789', 'Test Title, with a Comma', 'John Doe, Jane Smith',
                'Artificial Intelligence (cs.AI)', 'https://arxiv.org/abs/1234.56789',
                'https://arxiv.org/pdf/1234.56789'
            ]
            
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0], expected_header)
            self.assertEqual(lines[1], expected_row)
            
        finally:
            os.remove(filename)

# --- Main Execution ---
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'test':
        print("Running unit tests...")
        # Prevent unittest from trying to parse our 'test' argument
        unittest.main(argv=sys.argv[:1])
    else:
        run_scraper()
