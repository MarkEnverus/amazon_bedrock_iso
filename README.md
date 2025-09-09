# Power Price & Weather Correlation Analysis with RAG

A comprehensive system that analyzes the correlation between Southern California power prices and weather data using AWS Bedrock AI and Retrieval-Augmented Generation (RAG) with LanceDB.

## 🚀 Features

- **Data Collection**: Fetches real-time power pricing data from GVSI API and weather data from Open-Meteo API
- **AI Analysis**: Uses AWS Bedrock (Claude) to analyze correlations between power prices and weather patterns  
- **Vector Database**: Stores data in LanceDB for fast semantic search and retrieval
- **RAG System**: Answers natural language questions about your data using Retrieval-Augmented Generation
- **Data Validation**: Uses Pydantic models for robust data parsing and validation

## 📋 Prerequisites

### AWS Setup
- AWS account with Bedrock access
- AWS CLI configured with appropriate profile
- Claude Sonnet model access in AWS Bedrock

### API Access
- GVSI API credentials for power pricing data
- Open-Meteo API (free, no credentials required)

### Environment Variables
```bash
export AWS_PROFILE="genai-power-user"
export ANTHROPIC_MODEL="q-developer-user" 
export CLAUDE_CODE_USE_BEDROCK="1"
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd amazon_bedrock_iso
```

2. **Install dependencies**
```bash
# Using UV (recommended)
uv install

# Or using pip
pip install boto3 requests pydantic lancedb sentence-transformers pandas pyarrow
```

3. **Configure AWS credentials**
```bash
aws configure --profile genai-power-user
```

## 📊 Data Sources

### Power Pricing Data (GVSI API)
- **Source**: Southern California CAISO node (#CAISO00000030000000000152)
- **Data**: Intraday power prices in $/MWh
- **Frequency**: Hourly intervals
- **Date Range**: August 13-27, 2025

### Weather Data (Open-Meteo API)
- **Location**: Los Angeles, CA (34.05°N, -118.25°W)
- **Data**: Daily temperature maximums and minimums in Celsius
- **Timezone**: America/Los_Angeles
- **Date Range**: August 13-27, 2025

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   GVSI API      │    │   Open-Meteo     │    │   AWS Bedrock   │
│ (Power Prices)  │    │  (Weather Data)  │    │    (Claude)     │
└─────────┬───────┘    └────────┬─────────┘    └────────┬────────┘
          │                     │                       │
          ▼                     ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Processing Pipeline                     │
│  • Fetch & validate data    • Aggregate by date               │
│  • Combine price & weather  • Generate correlations           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LanceDB Vector Store                       │
│  • Text descriptions        • 384-dim embeddings              │
│  • Semantic search         • Fast retrieval                   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Query System                             │
│  • Natural language input  • Context retrieval                │
│  • AI-enhanced responses   • Data-backed insights             │
└─────────────────────────────────────────────────────────────────┘
```

## 💻 Usage

### Basic Usage
```python
python main.py
```

This will:
1. Fetch power pricing and weather data
2. Analyze correlations using AWS Bedrock
3. Set up LanceDB vector database
4. Run example RAG queries

### Example RAG Queries
```python
from main import ask_rag_question

# Ask natural language questions about your data
questions = [
    "What were power prices like on the hottest days?",
    "How do temperatures above 35°C affect power pricing?", 
    "Which day had the highest power price?",
    "Show me the correlation between temperature and pricing",
    "What weather patterns led to price spikes?"
]

for question in questions:
    answer = ask_rag_question(question)
    print(f"Q: {question}")
    print(f"A: {answer}\n")
```

## 🗄️ Data Schema

### Combined Data Structure
```python
{
    "date": "2025-08-15",           # ISO date string
    "avg_price": 67.89,             # Average power price ($/MWh)
    "temp_max": 36.0,               # Maximum temperature (°C)
    "temp_min": 22.0,               # Minimum temperature (°C)
    "text_description": "Date: ...", # Natural language description
    "vector": [0.1, 0.2, ...]      # 384-dimensional embedding
}
```

### LanceDB Table Schema
| Column | Type | Description |
|--------|------|-------------|
| `date` | string | ISO date (YYYY-MM-DD) |
| `avg_price` | float | Daily average power price ($/MWh) |
| `temp_max` | float | Daily maximum temperature (°C) |
| `temp_min` | float | Daily minimum temperature (°C) |
| `text_description` | string | Natural language description for embedding |
| `vector` | array | 384-dimensional embedding vector |

## 🤖 RAG System

### How It Works

1. **Retrieval**: User question converted to embedding vector
2. **Search**: LanceDB finds most relevant historical data points
3. **Augmentation**: Retrieved data provides context for AI model
4. **Generation**: AWS Bedrock generates insights using historical context

### Example RAG Flow
```
User: "What were prices on hot days?"
  ↓
Vector Search: Find records with high temperatures
  ↓
Context: "Date: 2025-08-16, Price: $78.45, Max Temp: 38°C..."
  ↓
Bedrock: "Based on your data, the hottest day (38°C) had prices of $78.45/MWh, which is 73% above average..."
```

## 🧪 Testing

### Test LanceDB Setup
```python
from main import test_lancedb_with_real_data

# This will test the complete pipeline
test_lancedb_with_real_data()
```

### Verify Data Storage
```python
import lancedb

db = lancedb.connect("./power_weather_test_db")
table = db.open_table("real_power_weather")
data = table.to_pandas()
print(f"Stored {len(data)} records")
print(data.head())
```

## 📁 Project Structure

```
amazon_bedrock_iso/
├── main.py                    # Main application with all functionality
├── pyproject.toml             # Project dependencies and configuration
├── uv.lock                    # Dependency lock file
├── README.md                  # This file
└── power_weather_test_db/     # LanceDB vector database
    └── real_power_weather.lance/
        ├── _transactions/      # Transaction logs
        ├── _versions/          # Version manifests
        └── data/              # Vector data files
```

## 🔧 Key Functions

| Function | Purpose |
|----------|---------|
| `get_intraday_data()` | Fetch power pricing data from GVSI API |
| `get_daily_temperature_socal()` | Fetch weather data from Open-Meteo API |
| `aggregate_intraday_by_date()` | Convert hourly to daily average prices |
| `combine_price_weather()` | Merge price and weather data by date |
| `summarize_correlation_bedrock()` | AI analysis of price-weather correlations |
| `test_lancedb_with_real_data()` | Set up vector database with real data |
| `ask_rag_question()` | Answer questions using RAG system |

## 🔍 Example Output

### Correlation Analysis
```
Correlation Summary:
Based on the data from August 13-27, 2025, there's a strong positive 
correlation (r=0.78) between temperature and power prices in Southern 
California. On days with temperatures above 35°C, average prices were 
$73.12/MWh compared to $45.67/MWh on cooler days below 30°C.

The highest price spike occurred on August 16th ($78.45/MWh) when 
temperatures reached 38°C, likely due to increased air conditioning 
demand stressing the grid.
```

### RAG Query Example
```
❓ What were power prices like on the hottest days?

🤖 Based on your historical data, the hottest days showed significantly 
elevated power prices. On August 16th, when temperatures peaked at 38°C, 
power prices averaged $78.45/MWh. This represents a 72% premium over the 
average price of $45.67/MWh during the period. The correlation suggests 
that extreme heat drives up electricity demand for cooling, creating 
supply-demand imbalances that result in higher wholesale prices.
```

## ⚠️ Important Notes

- **API Costs**: AWS Bedrock charges per API call. Monitor usage in AWS console.
- **Rate Limits**: Both GVSI and Open-Meteo APIs have rate limits. The code includes error handling for API failures.
- **Data Storage**: LanceDB creates local files. Ensure adequate disk space (~100MB for typical datasets).
- **Model Downloads**: Sentence transformers downloads ~90MB model on first run.

## 🐛 Troubleshooting

### Common Issues

**"No intraday data retrieved"**
- Check GVSI API credentials
- Verify internet connectivity
- Check API endpoint availability

**"ERROR: Can't invoke model"**
- Verify AWS credentials are configured
- Check Bedrock model access permissions
- Ensure correct AWS region (us-east-1)

**"❌ Run test_lancedb_with_real_data() first"**
- The RAG database hasn't been created
- Run the main script or call `test_lancedb_with_real_data()` directly

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run your functions to see detailed output
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **GVSI** for power market data API
- **Open-Meteo** for free weather data API
- **AWS Bedrock** for Claude AI model access
- **LanceDB** for vector database capabilities
- **Hugging Face** for sentence-transformers models
