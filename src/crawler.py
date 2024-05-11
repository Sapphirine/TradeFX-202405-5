import os
import re
import requests
from requests.exceptions import HTTPError, RequestException, Timeout
from bs4 import BeautifulSoup
import jsonlines
import sys
import datetime
import time
import random

# Function to convert date string to timestamp
def date_to_timestamp(date_str):
    parts = date_str.split(', ')
    if len(parts) > 1:
        date_str_no_weekday = parts[1]
    else:
        date_str_no_weekday = parts[0]

    date_and_time = date_str_no_weekday.split(' GMT')[0]  # Shave off the timezone
    date_format = "%d/%m/%Y | %H:%M"
    try:
        date_obj = datetime.datetime.strptime(date_and_time, date_format)
        timestamp = date_obj.timestamp()
        return timestamp
    except ValueError as e:
        print(f"Error converting date: {e}")
        return None

# Function to extract briefs from webpage
def get_briefs_from_webpage(soup,size):
    script_tags = soup.find_all('script')
    pattern = re.compile(r'expandedContent:"(.*?)"')
    briefs = []
    for script_tag in script_tags:
        script_content = script_tag.string
        
        if script_content and 'window.__NUXT__=' in script_content:
            script_content_cleaned = script_content.replace('\\u002', '').replace('\\r', '').replace('\\n', ' ').replace('&nbsp;','').replace('\\u003','')
            matches = pattern.findall(script_content_cleaned)
            briefs = matches
            break
    if len(briefs) == size:
        return briefs
    else:
        return ['' for i in range(size)]


try:
    # Get page range from command line arguments
    if len(sys.argv) != 3:
        print("Usage: python fxnews.py <start_page> <end_page>")
        sys.exit(1)

    start_page = int(sys.argv[1])
    end_page = int(sys.argv[2])

    if start_page > end_page:
        print("Start page cannot be greater than end page.")
        sys.exit(1)

    # Get the current working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the directory containing the JSONL files
    jsonl_dir = os.path.join(current_dir, 'forexnews')
    os.makedirs(jsonl_dir, exist_ok=True)
    
    for page_num in range(start_page, end_page + 1):

        attempts = 0
        success = False
        while attempts < 3 and not success:  # Try at most two times
            try:
                print(f"working on page {page_num}")
                # Get the HTML content from the website
                url = f'https://www.forexlive.com/page/{page_num}/'
                response = requests.get(url, timeout=15)
                response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
                html_content = response.text

                # Parse the HTML
                soup = BeautifulSoup(html_content, 'html.parser')

                # Extract all instances of headline, brief, and date
                headlines = [headline.get_text(strip=True).replace('\n', '') for headline in soup.find_all('h3', class_='article-slot__title title bottom')]
                briefs = [li.get_text(strip=True) for li in soup.find_all('li', class_='text-body tldr__item bold')]
                dates = [date.get_text(strip=True).replace('\n', '') for date in soup.find_all('div', class_='publisher-details__date')]
                timestamps = [date_to_timestamp(date) for date in dates]
                if len(briefs) != len(headlines):
                    briefs = get_briefs_from_webpage(soup,len(headlines))
                    

                # Construct the absolute path to the JSONL file within the directory
                jsonl_file_path = os.path.join(jsonl_dir, f'{page_num}.jsonl')

                # Write extracted information to a JSONL file
                with jsonlines.open(jsonl_file_path, mode='w') as writer:
                    for headline, brief, timestamp in zip(headlines, briefs, timestamps):
                        data = {
                            'headline': headline,
                            'brief': brief,
                            'date': timestamp
                        }
                        writer.write(data)

                success = True
                current_time = datetime.datetime.now()
                formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"page {page_num} completed at {formatted_time}")

            except Timeout as e:
                print(f"Timeout error on page {page_num}: {e}")
                attempts += 1
                if attempts < 3:
                    print("Waiting 2 minutes before retrying due to timeout...")
                    time.sleep(120)
                else:
                    print("Failed to fetch page after 2 attempts due to timeout.")


            except requests.HTTPError as e:
                print(f"HTTP error on page {page_num}: {e}")
                attempts += 1
                if attempts < 3:
                    print("Waiting 2 minutes before retrying...")
                    time.sleep(120)

            
            if page_num % 100 == 0:
                sleep_time = random.randint(65, 105)   # Randomly rest for every 100 requests
                print("Reached 100 pages, taking a 65-105 second break...")
                time.sleep(sleep_time)
            else:
                time.sleep(random.randint(1,5))       # Wait some time before going to next page


except requests.HTTPError as e:
    print("HTTP error occurred:", e)
except requests.RequestException as e:
    print("Error fetching content:", e)
except ValueError:
    print("Please provide valid integer values for start and end pages.")
except Exception as e:
    print("An error occurred:", e)