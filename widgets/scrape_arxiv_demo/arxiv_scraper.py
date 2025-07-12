import os
import csv
import time
import requests
import arxiv
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from tqdm import tqdm
import logging

# --- Configuration ---
START_YEAR = 2025
START_MONTH = 1
END_YEAR = 2025
END_MONTH = 6
CATEGORY = 'cs.AI'
OUTPUT_CSV_FILE = 'arxiv_ai_papers.csv'
BATCH_SIZE = 10  # Number of records to write at a time
LIST_PAGE_SIZE = 2000 # Number of results to fetch per page for the paper list
ID_CACHE_DIR = 'id_cache' # Directory to store cached paper ID lists

# --- Retry and Backoff Configuration ---
MAX_RETRIES = 5
INITIAL_BACKOFF = 2  # seconds

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_paper_ids_for_month(year, month):
    """
    Fetches all paper IDs for a given month and category from the arXiv list page.
    Uses pagination and caches results to a local file to avoid re-fetching.
    """
    # --- 1. Setup cache file path ---
    os.makedirs(ID_CACHE_DIR, exist_ok=True)
    cache_filename = os.path.join(ID_CACHE_DIR, f"{CATEGORY.replace('.', '_')}_{year}-{month:02d}.txt")

    # --- 2. Check if a cached list exists ---
    if os.path.isfile(cache_filename):
        logging.info(f"Loading paper IDs from cache: {cache_filename}")
        with open(cache_filename, 'r', encoding='utf-8') as f:
            paper_ids = [line.strip() for line in f if line.strip()]
        logging.info(f"Loaded {len(paper_ids)} IDs from cache.")
        return paper_ids

    # --- 3. If no cache, fetch from the web using pagination ---
    logging.info(f"Cache not found. Fetching paper IDs for {year}-{month:02d} from arXiv.")
    year_short = str(year)[-2:]
    month_str = f"{month:02d}"
    url_base = f"https://arxiv.org/list/{CATEGORY}/20{year_short}-{month_str}"
    
    all_paper_ids = []
    skip_count = 0
    is_first_page = True

    while True:
        paginated_url = f"{url_base}?skip={skip_count}&show={LIST_PAGE_SIZE}"
        ids_on_page = []
        
        for attempt in range(MAX_RETRIES):
            try:
                logging.info(f"Fetching paper list from {paginated_url}")
                response = requests.get(paginated_url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                dt_tags = soup.find_all('dt')
                for dt in dt_tags:
                    link = dt.find('a', title='Abstract')
                    if link and link.text.strip().startswith('arXiv:'):
                        ids_on_page.append(link.text.strip().split(':')[1])
                
                break # Success, exit retry loop
            
            except requests.exceptions.RequestException as e:
                logging.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed to fetch list page: {e}")
                if attempt + 1 == MAX_RETRIES:
                    logging.error(f"Could not fetch paper list page {paginated_url} after {MAX_RETRIES} retries.")
                    if all_paper_ids:
                        logging.warning("Returning partial list of IDs due to network failure. Will not cache.")
                        return all_paper_ids
                    return []
                backoff_time = INITIAL_BACKOFF * (2 ** attempt)
                logging.info(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
        
        if not ids_on_page and is_first_page:
            logging.warning(f"No papers found for {year}-{month_str}. It might be a future date or an empty month.")
            return [] # Month is empty, no need to cache an empty file.
        
        all_paper_ids.extend(ids_on_page)
        
        # Check for the last page: if we got fewer results than we asked for
        if len(ids_on_page) < LIST_PAGE_SIZE:
            logging.info(f"Reached the last page for {year}-{month_str}. Total papers found: {len(all_paper_ids)}")
            break
        
        skip_count += LIST_PAGE_SIZE
        is_first_page = False
        time.sleep(1) # Be polite to the server

    # --- 4. Save the complete list to cache ---
    if all_paper_ids:
        logging.info(f"Saving {len(all_paper_ids)} paper IDs to cache: {cache_filename}")
        with open(cache_filename, 'w', encoding='utf-8') as f:
            for paper_id in all_paper_ids:
                f.write(f"{paper_id}\n")
    
    return all_paper_ids

def fetch_paper_details(paper_id):
    """
    Fetches detailed metadata for a single paper ID using the arxiv library.
    Includes retry and backoff logic.
    """
    for attempt in range(MAX_RETRIES):
        try:
            # Search for the paper by its ID
            search = arxiv.Search(id_list=[paper_id], max_results=1)
            paper = next(search.results())
            
            # Format the data
            authors = ', '.join([author.name for author in paper.authors])
            # Clean up abstract by replacing newlines with spaces for better CSV compatibility
            abstract = paper.summary.replace('\n', ' ').strip()
            subjects = ', '.join(paper.categories)
            
            return {
                "ID": paper.entry_id.split('/')[-1], # Get the clean ID
                "Title": paper.title,
                "Authors": authors,
                "Subjects": subjects,
                "Abstract": abstract,
                "Abstract Link": paper.entry_id,
                "PDF Link": paper.pdf_url
            }

        except Exception as e:
            logging.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed to fetch details for {paper_id}: {e}")
            if attempt + 1 == MAX_RETRIES:
                logging.error(f"Could not fetch details for {paper_id} after {MAX_RETRIES} retries.")
                return None
            backoff_time = INITIAL_BACKOFF * (2 ** attempt)
            logging.info(f"Retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
            
def write_batch_to_csv(data_batch, filename):
    """
    Appends a batch of data to a CSV file. Creates the file and header if it doesn't exist.
    """
    # Check if file exists to determine if we need to write the header
    file_exists = os.path.isfile(filename)
    
    # Use 'a' for append mode, which is crucial for batch writing
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["ID", "Title", "Authors", "Subjects", "Abstract", "Abstract Link", "PDF Link"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerows(data_batch)
    logging.info(f"Successfully wrote a batch of {len(data_batch)} records to {filename}")


def main():
    """
    Main function to orchestrate the scraping process.
    """
    # Generate list of months to process
    months_to_process = []
    current_date = datetime(END_YEAR, END_MONTH, 1)
    start_date = datetime(START_YEAR, START_MONTH, 1)
    
    while current_date >= start_date:
        months_to_process.append((current_date.year, current_date.month))
        # Move to the previous month
        if current_date.month == 1:
            current_date = current_date.replace(year=current_date.year - 1, month=12)
        else:
            current_date = current_date.replace(month=current_date.month - 1)

    # Outer progress bar for months
    with tqdm(total=len(months_to_process), desc="Processing Months") as month_pbar:
        for year, month in months_to_process:
            month_pbar.set_description(f"Processing Month {year}-{month:02d}")
            
            paper_ids = get_paper_ids_for_month(year, month)
            
            if not paper_ids:
                month_pbar.update(1)
                continue

            data_batch = []
            
            # Inner progress bar for papers within a month
            with tqdm(total=len(paper_ids), desc=f"Papers in {year}-{month:02d}", leave=False) as paper_pbar:
                for paper_id in paper_ids:
                    details = fetch_paper_details(paper_id)
                    if details:
                        data_batch.append(details)
                    
                    # When batch is full, write it to the CSV and reset
                    if len(data_batch) >= BATCH_SIZE:
                        write_batch_to_csv(data_batch, OUTPUT_CSV_FILE)
                        data_batch = [] # Clear the batch
                    
                    paper_pbar.update(1)
                    time.sleep(1)

            # Write any remaining items in the last batch for the month
            if data_batch:
                write_batch_to_csv(data_batch, OUTPUT_CSV_FILE)

            month_pbar.update(1)
            
    logging.info(f"Scraping complete. Data saved to {OUTPUT_CSV_FILE}")


if __name__ == "__main__":
    main()
