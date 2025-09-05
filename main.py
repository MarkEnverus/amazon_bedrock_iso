import json
import requests
from datetime import datetime
from pydantic import BaseModel, field_validator
import boto3
from botocore.exceptions import ClientError
import os
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

os.environ["AWS_PROFILE"] = "genai-power-user"
os.environ["ANTHROPIC_MODEL"] = "q-developer-user"
os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"

class IntradayItem(BaseModel):
    symbol: str
    close: float
    tradedatetimeutc: datetime
    volume: int

    @field_validator('tradedatetimeutc', mode='before')
    def parse_datetime(cls, value):
        from datetime import datetime
        formats = [
            "%m/%d/%Y %I:%M:%S %p",  # M/dd/yyyy HH:mm:ss AMPM
            "%Y-%m-%dT%H:%M:%S",     # ISO format
            "%Y-%m-%d %H:%M:%S",     # Common format
        ]
        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except (ValueError, TypeError):
                continue
        raise ValueError(f"Date format not recognized: {value}")

def summarize_intraday(items):
    client = boto3.client('bedrock-runtime', region_name='us-east-2')
    prompt = "Summarize the following intraday data:\n"
    for item in items:
        prompt += f"{item.symbol}, {item.close}, {item.tradedatetimeutc}, {item.volume}\n"

    # Create a Bedrock Runtime client in the AWS Region of your choice.
    client = boto3.client("bedrock-runtime", region_name="us-east-1")

    # Set the model ID, e.g., Claude 3 Haiku.
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"

    # Format the request payload using the model's native structure.
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "temperature": 0.5,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    try:
        # Invoke the model with the request.
        response = client.invoke_model(modelId=model_id, body=request)

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)

    # Decode the response body.
    model_response = json.loads(response["body"].read())

    # Extract and print the response text.
    response_text = model_response["content"][0]["text"]
    print(response_text)


def get_intraday_data():
    """
    Fetches intraday data from GVSI API and returns parsed JSON response.
    """
    auth = ("markj", "password!")
    url = "https://webservice.gvsi.com/api/v3/getintraday/"
    params = {
        "symbols": "#CAISO00000030000000000152",
        "fields": "symbol,close,tradedatetimeutc,volume",
        "output": "json",
        "includeheaders": "true",
        "startdate": "08/13/2025",
        "enddate": "08/27/2025"
    }

    try:
        response = requests.get(url, params=params, auth=auth)
        response.raise_for_status()

        # Parse JSON response
        data = response.json()

        # If you want to use Pydantic for validation
        items = [IntradayItem(**item) for item in data]
        return items

        # Or return raw JSON data
        # return data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None


def grab_data():
    """
    Retrieves intraday data, parses it with Pydantic model and prints it to console
    """
    # Get data from the API
    items = get_intraday_data()

    if not items:
        print("No data retrieved")
        return

    summarize_intraday(items)
    # Print header
    print(f"{'Symbol':<30} {'Close':>10} {'DateTime UTC':<25} {'Volume':>8}")
    print("-" * 75)

    # Loop through and print each item
    for item in items:
        print(
            f"{item.symbol:<30} {item.close:>10.5f} {item.tradedatetimeutc.strftime('%Y-%m-%d %H:%M:%S'):<25} {item.volume:>8}")

    print(f"\nTotal items retrieved: {len(items)}")


def get_daily_temperature_socal():
    """
    Fetches daily temperature data for Southern California (Los Angeles) from Open-Meteo API.
    Returns a list of dicts with date and temperature.
    """
    import datetime
    lat, lon = 34.05, -118.25  # Los Angeles coordinates
    # Convert MM/DD/YYYY to ISO format
    start_date_str = "08/13/2025"
    end_date_str = "08/27/2025"
    start_date = datetime.datetime.strptime(start_date_str, "%m/%d/%Y").date().isoformat()
    end_date = datetime.datetime.strptime(end_date_str, "%m/%d/%Y").date().isoformat()
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": "America/Los_Angeles"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        # Extract dates and temperatures
        dates = data["daily"]["time"]
        temps_max = data["daily"]["temperature_2m_max"]
        temps_min = data["daily"]["temperature_2m_min"]
        return [
            {"date": date, "temp_max": tmax, "temp_min": tmin}
            for date, tmax, tmin in zip(dates, temps_max, temps_min)
        ]
    except Exception as e:
        print(f"ERROR: Can't fetch weather data. Reason: {e}")
        return []


def aggregate_intraday_by_date(items):
    """
    Aggregates intraday items by date, returning a dict of date -> average close price.
    """
    from collections import defaultdict
    import datetime
    prices_by_date = defaultdict(list)
    for item in items:
        date = item.tradedatetimeutc.date() if hasattr(item.tradedatetimeutc, 'date') else datetime.datetime.strptime(item.tradedatetimeutc, "%Y-%m-%d %H:%M:%S").date()
        prices_by_date[date].append(item.close)
    avg_prices = {date.isoformat(): sum(prices)/len(prices) for date, prices in prices_by_date.items()}
    return avg_prices

def combine_price_weather(avg_prices, weather_data):
    """
    Combines average daily prices and weather data by date.
    Returns a list of dicts with date, avg_price, temp_max, temp_min.
    """
    combined = []
    for w in weather_data:
        date = w["date"]
        if date in avg_prices:
            combined.append({
                "date": date,
                "avg_price": avg_prices[date],
                "temp_max": w["temp_max"],
                "temp_min": w["temp_min"]
            })
    return combined


def summarize_correlation_bedrock(combined_data):
    """
    Summarizes the correlation between power prices and weather using Bedrock.
    """
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    # Format the prompt
    prompt = "Analyze the correlation between daily average power prices and weather (max/min temperature) for Southern California. Here is the data:\n\n"
    prompt += "Date, Avg_Price, Temp_Max, Temp_Min\n"
    for entry in combined_data:
        prompt += f"{entry['date']}, {entry['avg_price']:.2f}, {entry['temp_max']}, {entry['temp_min']}\n"
    prompt += "\nSummarize any patterns, trends, or correlations you observe."
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "temperature": 0.5,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }
    request = json.dumps(native_request)
    try:
        response = client.invoke_model(modelId=model_id, body=request)
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        return None
    model_response = json.loads(response["body"].read())
    response_text = model_response["content"][0]["text"]
    print("\nCorrelation Summary:\n", response_text)
    return response_text


def test_lancedb_with_real_data():
    """Test LanceDB with your actual power and weather data"""

    # Get your existing data
    items = get_intraday_data()
    if not items:
        print("No intraday data retrieved")
        return

    avg_prices = aggregate_intraday_by_date(items)
    weather_data = get_daily_temperature_socal()
    if not weather_data:
        print("No weather data retrieved")
        return

    combined_data = combine_price_weather(avg_prices, weather_data)

    # Test LanceDB storage
    print("🚀 Testing LanceDB with real data...")

    db = lancedb.connect("./power_weather_test_db")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    records = []
    for entry in combined_data:
        text_description = f"Date: {entry['date']}, Average Power Price: ${entry['avg_price']:.2f}, Max Temperature: {entry['temp_max']}°C, Min Temperature: {entry['temp_min']}°C"
        embedding = model.encode(text_description).tolist()

        records.append({
            **entry,
            "text_description": text_description,
            "vector": embedding
        })

    table = db.create_table("real_power_weather", records, mode="overwrite")
    print(f"✅ Stored {len(records)} real records in LanceDB")

    # Test queries
    test_queries = [
        "hottest days with high power prices",
        "correlation between temperature and pricing",
        "August 15th weather and prices"
    ]

    for query in test_queries:
        query_embedding = model.encode(query).tolist()
        results = table.search(query_embedding).limit(3).to_pandas()
        print(f"\n🔍 Query: '{query}'")
        for _, row in results.iterrows():
            print(f"  📅 {row['date']}: ${row['avg_price']:.2f}, {row['temp_max']}°C max")


def ask_rag_question(question):
    """RAG query using your existing LanceDB data"""
    db = lancedb.connect("./power_weather_test_db")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    try:
        table = db.open_table("real_power_weather")
    except:
        print("❌ Run test_lancedb_with_real_data() first to create the database")
        return None

    # Search your stored embeddings
    query_embedding = model.encode(question).tolist()
    results = table.search(query_embedding).limit(5).to_pandas()

    print(f"🔍 Found {len(results)} relevant records")

    # Build context from your stored text_descriptions
    context = "Historical power price and weather data:\n\n"
    for _, row in results.iterrows():
        context += f"• {row['text_description']}\n"

    # Enhanced prompt for Bedrock
    prompt = f"""{context}

Question: {question}

Analyze the historical data above and provide insights about Southern California power prices and weather patterns."""

    # Call Bedrock
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"

    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 800,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    }

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(native_request))
        model_response = json.loads(response["body"].read())
        return model_response["content"][0]["text"]
    except Exception as e:
        return f"Error calling Bedrock: {e}"


# Add this to your main function
if __name__ == '__main__':
    # Your existing code...
    items = get_intraday_data()
    if not items:
        print("No intraday data retrieved")
    else:
        avg_prices = aggregate_intraday_by_date(items)
        weather_data = get_daily_temperature_socal()
        if not weather_data:
            print("No weather data retrieved")
        else:
            combined_data = combine_price_weather(avg_prices, weather_data)
            summarize_correlation_bedrock(combined_data)

    print("🚀 Testing LanceDB with your real data...")
    # Test LanceDB
    test_lancedb_with_real_data()
    # Now test RAG queries
    print("\n" + "=" * 50)
    print("🤖 Testing RAG Questions")
    print("=" * 50)

    questions = [
        "What were the power prices like on the hottest days?",
        "How do temperatures affect power pricing?",
        "Which day had the highest power price?"
    ]

    for q in questions:
        print(f"\n❓ {q}")
        answer = ask_rag_question(q)
        print(f"🤖 {answer}")
