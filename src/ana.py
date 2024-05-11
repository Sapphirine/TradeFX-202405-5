import re
import os
import json
from openai import OpenAI
import httpx
import datetime
from requests.exceptions import HTTPError, RequestException, Timeout
import backoff


# https://github.com/xtekky/gpt4free?tab=readme-ov-file#chatcompletion


# if we dont't use this funtion the files just can't be sorted
def sort_numeric(file_name):
    # Use regular expression to find all numbers in the file names
    numbers = re.findall(r'\d+', file_name)
    # Turn the number part into int and return 0 if there's no numbers
    return int(numbers[0]) if numbers else 0


client = OpenAI(
    base_url="https://oneapi.xty.app/v1", 
    api_key="sk-yhVE3zTZ8OclKAEV04EcF6C637D14b7d9d1d6cFa70A2B96c",
    http_client=httpx.Client(base_url="https://oneapi.xty.app/v1", follow_redirects=True),
)

# Function decorator to handle exponential backoff for HTTP exceptions
@backoff.on_exception(backoff.expo,
                      (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError),
                      max_tries=5,
                      giveup=lambda e: e.response is not None and e.response.status_code < 500)
def askGPT(headline,brief):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": f"Act as a sentiment analysis service of a financial platform. Based only on the following headline and brief '{headline}\n{brief}', provide a sentiment score for the related forex or the currency of related country. The sentiment score should range from -10.0 to +10.0, where -10.0 strongly suggests selling, 0 suggests holding, and +10.0 strongly suggests buying. Provide the sentiment in JSON format e.g. {{'cuuency': 'USD','score':-3.4}}. A score of -10.0 to +10.0 represents the sentiment from strongly sell to strongly buy. Provide only {{'cuuency':'NA','score':0}} in JSON format if the headline is unrelated to a forex."}
    ]
    )
    return completion.choices[0].message.content

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute paths to the directories containing JSONL files and analysis files
jsonl_dir = os.path.join(current_dir, 'raw_news')
ana_dir = os.path.join(current_dir, 'analysis')
os.makedirs(jsonl_dir, exist_ok=True)
os.makedirs(ana_dir, exist_ok=True)

# List files in the folder
files = os.listdir(jsonl_dir)
# Filter JSONL files if needed
jsonl_files = sorted([f for f in files if f.endswith('.jsonl')], key = sort_numeric)


# These two indexes were used because we experienced frequent network issues
request_count = 0
start_index = 9876


# Loop through each JSONL file starting from the specified index
for file_name in jsonl_files[start_index:]:
    attempts = 0
    success = False

    # Construct the file path for the JSONL file
    file_path = os.path.join(jsonl_dir, file_name)
    jsonl = []
    with open(file_path, 'r') as file:
        # Read each line (JSON object) from the file
        for line in file:
            # Process JSON object
            data = json.loads(line)
            # Call askGPT function to generate sentiment analysis based on headline and brief
            res = askGPT(data['headline'],data['brief'])
            analysis = {
                'response':res, 'headline':data['headline'], 'brief':data['brief'],'date':data['date']
                }
            # Append analysis dictionary to jsonl list
            jsonl.append(analysis)

            request_count += 1

    # Open the analysis JSONL file for writing
    jsonl_file_path = os.path.join(ana_dir, file_name)
    with open(jsonl_file_path, 'w') as writer:
        for item in jsonl:
            # Convert dictionary to JSON string
            json_line = json.dumps(item)
            # Write JSON string followed by newline character
            writer.write(json_line + '\n')

    # Get the current time and format it
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{file_name} completed at {formatted_time}")
    success = True
        