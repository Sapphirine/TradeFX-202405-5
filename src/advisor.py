import json
import pandas as pd
import warnings
from openai import OpenAI
import httpx
import re
import requests
from bs4 import BeautifulSoup
import datetime
# Suppress all warnings
warnings.filterwarnings("ignore")

# Initialize OpenAI client with specified base URL and API key
client = OpenAI(
    base_url="https://oneapi.xty.app/v1",
    api_key="sk-yhVE3zTZ8OclKAEV04EcF6C637D14b7d9d1d6cFa70A2B96c",
    http_client=httpx.Client(
        base_url="https://oneapi.xty.app/v1", follow_redirects=True),
)


# Function to extract currency data from a CSV for a specified period
def get_currency_info(input_csv, currency, start_date, end_date):
    df = pd.read_csv(input_csv)
    df['date'] = pd.to_datetime(df['date'], unit='s')  # Convert Unix timestamp to datetime object

    start_date = pd.to_datetime(start_date)  # Convert start date string to datetime
    end_date = pd.to_datetime(end_date)  # Convert end date string to datetime

    # Create a mask to filter data within the date range
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    filtered_df = df[mask]

    # Further filter data where the response contains the specified currency
    currency_mask = filtered_df['response'].str.contains(f"'currency': '{currency}'")
    filtered_df = filtered_df[currency_mask]

    # Extract the numeric score from the response string and convert to float
    filtered_df['score'] = filtered_df['response'].str.extract("'score': ([-+]?[0-9]*\.?[0-9]+)").astype(float)

    # Calculate total and average score
    total_score = filtered_df['score'].sum()
    average_score = total_score / 365.00  # Normalize the score over 365 days

    # Compile the results into a dictionary
    result = {
        'start': start_date.strftime('%Y-%m-%d'),
        'end': end_date.strftime('%Y-%m-%d'),
        'currency': currency,
        'sum': total_score,
        'avg': average_score
    }

    return result

# Function to convert a date string to a Unix timestamp
def date_to_timestamp(date_str):
    parts = date_str.split(', ')
    date_str_no_weekday = parts[1] if len(parts) > 1 else parts[0]

    date_and_time = date_str_no_weekday.split(' GMT')[0]  # Remove timezone information
    date_format = "%d/%m/%Y | %H:%M"
    try:
        date_obj = datetime.datetime.strptime(date_and_time, date_format)
        timestamp = date_obj.timestamp()
        return timestamp
    except ValueError as e:
        print(f"Error converting date: {e}")
        return None



def get_briefs_from_webpage(soup, size):
    # Find all <script> tags in the webpage
    script_tags = soup.find_all('script')
    # Define a regular expression pattern to match the expandedContent field in the script
    pattern = re.compile(r'expandedContent:"(.*?)"')
    # Get a list to store the briefs
    briefs = []
    # Iterate over all <script> tags
    for script_tag in script_tags:
        script_content = script_tag.string
        # If the <script> tag contains content and it contains a specific identifier
        if script_content and 'window.__NUXT__=' in script_content:
            # Clean the script content by removing special characters
            script_content_cleaned = script_content.replace('\\u002', '').replace(
                '\\r', '').replace('\\n', ' ').replace('&nbsp;', '').replace('\\u003', '').replace('CFpECulECliE', '').replace('?CFstrongECFh2ECpEAt', '').replace('CpE', '').replace('CFliECliE', '')
            # Find matches using the regular expression pattern
            matches = pattern.findall(script_content_cleaned)
            # Store the matches in the briefs list
            briefs = matches
            break
    # If the length of briefs list equals size, it means enough briefs were found
    if len(briefs) == size:
        return briefs
    else:
        # Otherwise return a list of empty strings with length equal to size
        return ['' for i in range(size)]


def askGPT(headline, brief):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Act as a sentiment analysis service of a financial platform. Based only on the following headline and brief '{headline}\n{brief}', provide a sentiment score for the related forex or the currency of related country. The sentiment score should range from -10.0 to +10.0, where -10.0 strongly suggests selling, 0 suggests holding, and +10.0 strongly suggests buying. Provide the sentiment in JSON format e.g. {{'cuuency': 'USD','score':-3.4}}. A score of -10.0 to +10.0 represents the sentiment from strongly sell to strongly buy. Provide only {{'cuuency':'NA','score':0}} in JSON format if the headline is unrelated to a forex."}
        ]
    )
    return completion.choices[0].message.content


# Function to crawl news articles from a specific URL and analyze their relevance to a given currency
def crawl(currency, max_count=10):
    page_num = 0
    news_list = []
    while len(news_list) < max_count:
        page_num += 1
        url = f'https://www.forexlive.com/page/{page_num}/'
        response = requests.get(url, timeout=15)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        html_content = response.text

        # Parse the HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract all instances of headline, brief, and date
        headlines = [headline.get_text(strip=True).replace(
            '\n', '') for headline in soup.find_all('h3', class_='article-slot__title title bottom')]

        briefs = [li.get_text(strip=True) for li in soup.find_all(
            'li', class_='text-body tldr__item bold')]
        dates = [date.get_text(strip=True).replace('\n', '') for date in soup.find_all(
            'div', class_='publisher-details__date')]
        timestamps = [date_to_timestamp(date) for date in dates]
        for headline, brief, timestamp in zip(headlines, briefs, timestamps):
            raw_gpt_json = askGPT(headline, brief).replace("'", '"').replace(
                'Reminder: My knowledge is up to date only until January 2022, and I cannot provide real-time information.', '')
            try:
                j = json.loads(raw_gpt_json)
                
            except json.JSONDecodeError as e:
                print(e)
            else:
                # Filter news based on currency relevance
                if j['currency'] != 'NA':
                    if len(j['currency'])==3 and (j['currency'] == currency or j['currency'] == 'USD'):
                        analysis = {
                        'currency': j['currency'], 'headline': headline, 'brief': brief, 'timestamp': timestamp, 'score': j['score']
                        }
                        news_list.append(analysis)
                    elif len(j['currency'])==6 and (currency in j['currency'] and 'USD' in j['currency']):
                        analysis = {
                        'currency': j['currency'][0:4], 'headline': headline, 'brief': brief, 'timestamp': timestamp, 'score': j['score']
                        }
                        news_list.append(analysis)
    news_df = pd.DataFrame(news_list)
    news_df['date'] = pd.to_datetime(
        news_df['timestamp'], unit='s').dt.strftime('%d/%m/%Y %H:%M:%S')
    return news_df

# Function to retrieve the latest news articles relevant to a specific currency
def get_n_latest_news(n, currency, input_csv):
    df =  crawl(currency, max_count=n)  #pd.read_csv(input_csv)
    df['date'] = pd.to_datetime(
        df['timestamp'], unit='s').dt.strftime('%d/%m/%Y %H:%M:%S')

    # Filter based on exact match of currency code or name, including USD
    currency_mask = ((df['currency'] == currency) |
                     (df['currency'].str.lower() == currency.lower()) |
                     (currency.lower() == 'usd') |
                     (df['currency'].str.lower() == 'usd'))

    filtered_df = df[currency_mask]

    filtered_df = filtered_df.sort_values(by='date', ascending=False)
    latest_news = filtered_df.head(n)

    # Select only 'headline', 'brief', 'score', and 'currency' columns
    latest_news = latest_news[[
        'date', 'headline', 'brief', 'score', 'currency']]

    # Convert DataFrame to a list of dictionaries
    news_list = latest_news.to_dict(orient='records')

    return news_list