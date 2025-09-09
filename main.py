"""
Power Price and Weather Correlation Analysis with RAG

This module provides functionality to:
1. Fetch Southern California power pricing data from GVSI API
2. Retrieve weather data from Open-Meteo API
3. Analyze correlations using AWS Bedrock (Claude)
4. Store data in LanceDB for Retrieval-Augmented Generation (RAG)
5. Answer natural language questions about the data using RAG

Dependencies:
- boto3: AWS SDK for Bedrock integration
- requests: HTTP client for API calls
- pydantic: Data validation and parsing
- lancedb: Vector database for RAG
- sentence-transformers: Text embeddings
- pandas: Data manipulation

Environment Variables Required:
- AWS_PROFILE: AWS profile for Bedrock access
- ANTHROPIC_MODEL: Model configuration
- CLAUDE_CODE_USE_BEDROCK: Enable Bedrock usage
"""

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

# Environment configuration for AWS Bedrock
os.environ["AWS_PROFILE"] = "genai-power-user"
os.environ["ANTHROPIC_MODEL"] = "q-developer-user"
os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"


class IntradayItem(BaseModel):
    """
    Pydantic model for validating and parsing intraday power pricing data.

    Attributes:
        symbol (str): Power market symbol identifier (e.g., CAISO node)
        close (float): Closing price in $/MWh
        tradedatetimeutc (datetime): Trading timestamp in UTC
        volume (int): Trading volume
    """
    symbol: str
    close: float
    tradedatetimeutc: datetime
    volume: int

    @field_validator('tradedatetimeutc', mode='before')
    def parse_datetime(cls, value):
        """
        Validates and parses datetime strings in multiple formats.

        Args:
            value: DateTime string in various formats

        Returns:
            datetime: Parsed datetime object

        Raises:
            ValueError: If datetime format is not recognized
        """
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
    """
    Generates an AI summary of intraday power pricing data using AWS Bedrock.

    Args:
        items (List[IntradayItem]): List of validated intraday data items

    Returns:
        None: Prints summary to console

    Side Effects:
        - Makes API call to AWS Bedrock Claude model
        - Prints analysis to stdout
    """
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
    Fetches intraday power pricing data from GVSI API for Southern California.

    Returns:
        List[IntradayItem] | None: List of validated intraday items or None on error

    API Details:
        - Endpoint: GVSI API v3 getintraday
        - Symbol: #CAISO00000030000000000152 (Southern California node)
        - Date range: August 13-27, 2025
        - Fields: symbol, close price, datetime, volume

    Authentication:
        Uses HTTP basic auth with hardcoded credentials (markj/password!)
    """
    auth = ("markj", "password!")
    url = "https://webservice.gvsi.com/api/v3/getintraday/"
    params = {
        "symbols": "#CAISO00000030000000000152",  # Southern California CAISO node
        "fields": "symbol,close,tradedatetimeutc,volume",
        "output": "json",
        "includeheaders": "true",
        "startdate": "08/13/2025",
        "enddate": "08/27/2025"
    }

    try:
        response = requests.get(url, params=params, auth=auth)
        response.raise_for_status()

        # Parse JSON response and validate with Pydantic
        data = response.json()
        items = [IntradayItem(**item) for item in data]
        return items

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None


def grab_data():
    """
    Legacy function: Retrieves intraday data and prints formatted output.

    Side Effects:
        - Calls get_intraday_data() and summarize_intraday()
        - Prints formatted table of all retrieved data points
        - Displays total count of items retrieved

    Note:
        This function is primarily for debugging and initial data exploration.
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
    Fetches daily temperature data for Southern California from Open-Meteo API.

    Returns:
        List[Dict[str, Any]]: List of daily weather records with keys:
            - date (str): ISO format date (YYYY-MM-DD)
            - temp_max (float): Maximum temperature in Celsius
            - temp_min (float): Minimum temperature in Celsius

    API Details:
        - Location: Los Angeles (34.05°N, -118.25°W)
        - Date range: August 13-27, 2025
        - Timezone: America/Los_Angeles
        - Data: Daily temperature maximums and minimums

    Returns empty list on API errors.
    """
    import datetime
    lat, lon = 34.05, -118.25  # Los Angeles coordinates

    # Convert MM/DD/YYYY to ISO format for API compatibility
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

        # Extract and combine date/temperature arrays into structured records
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
    Aggregates intraday power pricing data by date to create daily averages.

    Args:
        items (List[IntradayItem]): List of intraday data points

    Returns:
        Dict[str, float]: Dictionary mapping ISO date strings to average prices
            - Key: Date in YYYY-MM-DD format
            - Value: Average closing price for that date in $/MWh

    Process:
        1. Groups all intraday prices by date
        2. Calculates arithmetic mean for each date
        3. Returns date->price mapping for correlation analysis
    """
    from collections import defaultdict
    import datetime

    prices_by_date = defaultdict(list)

    # Group prices by date
    for item in items:
        # Handle both datetime objects and string formats
        if hasattr(item.tradedatetimeutc, 'date'):
            date = item.tradedatetimeutc.date()
        else:
            date = datetime.datetime.strptime(item.tradedatetimeutc, "%Y-%m-%d %H:%M:%S").date()
        prices_by_date[date].append(item.close)

    # Calculate daily averages
    avg_prices = {
        date.isoformat(): sum(prices)/len(prices)
        for date, prices in prices_by_date.items()
    }
    return avg_prices


def combine_price_weather(avg_prices, weather_data):
    """
    Combines daily average power prices with weather data by matching dates.

    Args:
        avg_prices (Dict[str, float]): Daily average prices keyed by ISO date
        weather_data (List[Dict]): Weather records with date, temp_max, temp_min

    Returns:
        List[Dict[str, Any]]: Combined records with keys:
            - date (str): ISO date string
            - avg_price (float): Average power price in $/MWh
            - temp_max (float): Maximum temperature in Celsius
            - temp_min (float): Minimum temperature in Celsius

    Note:
        Only returns records where both price and weather data exist for a date.
        This ensures clean data for correlation analysis.
    """
    combined = []
    for weather_record in weather_data:
        date = weather_record["date"]
        if date in avg_prices:
            combined.append({
                "date": date,
                "avg_price": avg_prices[date],
                "temp_max": weather_record["temp_max"],
                "temp_min": weather_record["temp_min"]
            })
    return combined


def summarize_correlation_bedrock(combined_data):
    """
    Uses AWS Bedrock (Claude) to analyze correlation between power prices and weather.

    Args:
        combined_data (List[Dict]): Combined price/weather records

    Returns:
        str | None: AI-generated correlation analysis or None on error

    Process:
        1. Formats data into CSV-like text for AI analysis
        2. Sends structured prompt to Claude via Bedrock
        3. Requests pattern identification and correlation insights
        4. Returns analysis focusing on temperature-price relationships

    Side Effects:
        - Prints correlation summary to console
        - Makes API call to AWS Bedrock (charges apply)
    """
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"

    # Build structured prompt with data
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
    """
    Tests LanceDB vector database functionality with real power and weather data.

    Returns:
        None

    Process:
        1. Fetches real power pricing and weather data
        2. Creates embeddings using sentence-transformers
        3. Stores data in LanceDB with vector indices
        4. Tests semantic search with sample queries
        5. Displays search results for verification

    Storage Schema:
        - date: ISO date string
        - avg_price: Average power price ($/MWh)
        - temp_max/temp_min: Temperature data (Celsius)
        - text_description: Natural language description for embedding
        - vector: 384-dimensional embedding vector

    Side Effects:
        - Creates ./power_weather_test_db/ directory
        - Downloads sentence transformer model (~90MB on first run)
        - Prints test query results
    """
    # Get your existing data pipeline
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

    # Initialize LanceDB and embedding model
    print("🚀 Testing LanceDB with real data...")
    db = lancedb.connect("./power_weather_test_db")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create records with embeddings
    records = []
    for entry in combined_data:
        # Create natural language description for semantic search
        text_description = f"Date: {entry['date']}, Average Power Price: ${entry['avg_price']:.2f}, Max Temperature: {entry['temp_max']}°C, Min Temperature: {entry['temp_min']}°C"
        embedding = model.encode(text_description).tolist()

        records.append({
            **entry,  # Include all original fields
            "text_description": text_description,
            "vector": embedding  # 384-dimensional vector for semantic search
        })

    # Store in LanceDB with vector index
    table = db.create_table("real_power_weather", records, mode="overwrite")
    print(f"✅ Stored {len(records)} real records in LanceDB")

    # Test semantic search queries
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
    """
    Answers natural language questions using Retrieval-Augmented Generation (RAG).

    Args:
        question (str): Natural language question about power prices/weather

    Returns:
        str | None: AI-generated answer with supporting data or None on error

    RAG Process:
        1. RETRIEVAL: Converts question to embedding vector
        2. SEARCH: Finds most relevant historical data in LanceDB
        3. AUGMENTATION: Builds context from retrieved records
        4. GENERATION: Sends context + question to Bedrock for analysis

    Example Questions:
        - "What were power prices like on the hottest days?"
        - "How do temperatures affect power pricing?"
        - "Which day had the highest power price?"

    Dependencies:
        - Requires ./power_weather_test_db/ with real_power_weather table
        - Run test_lancedb_with_real_data() first to populate database
    """
    db = lancedb.connect("./power_weather_test_db")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    try:
        table = db.open_table("real_power_weather")
    except:
        print("❌ Run test_lancedb_with_real_data() first to create the database")
        return None

    # RETRIEVAL: Search for semantically similar records
    query_embedding = model.encode(question).tolist()
    results = table.search(query_embedding).limit(5).to_pandas()

    print(f"🔍 Found {len(results)} relevant records")

    # AUGMENTATION: Build context from retrieved data
    context = "Historical power price and weather data:\n\n"
    for _, row in results.iterrows():
        context += f"• {row['text_description']}\n"

    # Enhanced prompt for Bedrock with retrieved context
    prompt = f"""{context}

Question: {question}

Analyze the historical data above and provide insights about Southern California power prices and weather patterns."""

    # GENERATION: Call Bedrock with enhanced context
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"

    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 800,
        "temperature": 0.1,  # Low temperature for factual responses
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    }

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(native_request))
        model_response = json.loads(response["body"].read())
        return model_response["content"][0]["text"]
    except Exception as e:
        return f"Error calling Bedrock: {e}"


# Main execution pipeline
if __name__ == '__main__':
    """
    Main execution pipeline demonstrating the complete workflow:
    
    1. Data Collection: Fetch power prices and weather data
    2. Data Processing: Combine and aggregate data by date  
    3. AI Analysis: Generate correlation insights with Bedrock
    4. RAG Setup: Store data in LanceDB vector database
    5. Interactive Queries: Answer natural language questions using RAG
    
    The pipeline showcases integration between:
    - External APIs (GVSI, Open-Meteo)
    - AWS Bedrock (Claude AI model)
    - LanceDB (vector database)
    - Sentence Transformers (embeddings)
    """
    print("🚀 Starting Power Price & Weather Analysis Pipeline...")

    # Step 1-3: Data collection and initial analysis
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

    # Step 4: Setup RAG database
    test_lancedb_with_real_data()

    # Step 5: Test RAG queries
    print("\n" + "=" * 50)
    print("🤖 Testing RAG Questions")
    print("=" * 50)

    # Example questions demonstrating RAG capabilities
    questions = [
        "What were the power prices like on the hottest days?",
        "How do temperatures affect power pricing?",
        "Which day had the highest power price?"
    ]

    for q in questions:
        print(f"\n❓ {q}")
        answer = ask_rag_question(q)
        print(f"🤖 {answer}")
