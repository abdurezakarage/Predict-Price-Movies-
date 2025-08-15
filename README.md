# Integrated Sentiment-Price Analysis for Stock Market Prediction

## Project Overview

This project implements a comprehensive analysis system that combines **sentiment analysis of financial news** with **quantitative analysis of stock price data** to identify correlations and potential trading opportunities. The system addresses the feedback from the previous submission by providing:

- ✅ **Complete sentiment analysis implementation** using VADER and TextBlob
- ✅ **Explicit correlation analysis** between news sentiment and stock price movements
- ✅ **Integrated analysis workflow** combining both sentiment and quantitative components
- ✅ **Modular code structure** with clear separation of concerns
- ✅ **Comprehensive data handling** for both news and stock data

## Key Features

### 🔍 Sentiment Analysis
- **VADER Sentiment Analysis**: Financial domain-optimized sentiment scoring
- **TextBlob Analysis**: Alternative sentiment analysis with subjectivity scoring
- **Sentiment Trend Analysis**: Daily sentiment patterns and trends
- **Publisher Analysis**: Sentiment patterns by news source
- **Keyword Analysis**: Sentiment around specific financial terms

### 📊 Quantitative Analysis
- **Technical Indicators**: SMA, RSI, MACD, Bollinger Bands
- **Financial Metrics**: Returns, volatility, Sharpe ratio, drawdown
- **Volume Analysis**: Price-volume relationships and patterns
- **Trading Signals**: Technical indicator-based buy/sell signals

### 🔗 Integrated Analysis
- **Sentiment-Price Correlation**: Direct correlation between news sentiment and stock returns
- **Leading Indicators**: Sentiment as predictor of future price movements
- **Volume-Sentiment Relationships**: Analysis of trading volume and sentiment interactions
- **Trading Signals**: Sentiment-based trading recommendations
- **Comprehensive Reporting**: Integrated analysis summaries

## Project Structure

```
Predict-Price-Movies-/
├── src/
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── sentiment_Anaalysis.py   # Sentiment analysis and integration
│   ├── quanitative_Analysis.py  # Technical and financial analysis
│   ├── main_analysis.py         # Main execution workflow
│   └── _init_.py
├── data/                        # Data storage directory
├── tests/                       # Test files
├── notbooks/                    # Jupyter notebooks
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Predict-Price-Movies-
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (first run only):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('vader_lexicon')
   ```

## Usage

### Quick Demo
Run a simplified analysis to verify everything works:
```bash
python src/main_analysis.py --demo
```

### Full Analysis
Run the complete integrated analysis workflow:
```bash
python src/main_analysis.py
```

### Custom Analysis
```python
from src.data_loader import DataLoader
from src.sentiment_Anaalysis import SentimentAnalysis, IntegratedAnalysis
from src.quanitative_Analysis import QuanitativeAnalysis

# Load data
data_loader = DataLoader()
stock_data = data_loader.load_stock_data("AAPL", "2023-01-01", "2023-12-31")
news_data = data_loader.create_sample_news_data("AAPL", "2023-01-01", "2023-12-31")

# Perform analysis
sentiment_analyzer = SentimentAnalysis(news_data, "AAPL")
news_with_sentiment = sentiment_analyzer.calculate_sentiment_scores()

quant_analyzer = QuanitativeAnalysis(stock_data, "AAPL")
stock_with_indicators = quant_analyzer.calculate_technical_indicators()

integrated_analyzer = IntegratedAnalysis(news_with_sentiment, stock_with_indicators, "AAPL")
integrated_data = integrated_analyzer.prepare_data()

# Analyze correlations
integrated_analyzer.analyze_sentiment_price_correlation()
```

## Data Requirements

### Stock Data
- **Format**: OHLCV (Open, High, Low, Close, Volume) data
- **Source**: Automatically downloaded via yfinance
- **Frequency**: Daily data recommended
- **Required Columns**: `date`, `Open`, `High`, `Low`, `Close`, `Volume`

### News Data
- **Format**: CSV with financial news articles
- **Required Columns**: `date`, `headline`, `publisher`
- **Content**: Financial news headlines about the stock
- **Source**: Can be real data or generated sample data

## Analysis Workflow

1. **Data Loading**: Load stock and news data
2. **Data Validation**: Ensure compatibility and quality
3. **Sentiment Analysis**: Calculate sentiment scores for news articles
4. **Quantitative Analysis**: Calculate technical indicators and financial metrics
5. **Data Integration**: Align sentiment and stock data by date
6. **Correlation Analysis**: Analyze relationships between sentiment and price movements
7. **Leading Indicator Analysis**: Test sentiment as predictor of future returns
8. **Trading Signal Generation**: Create sentiment-based trading recommendations
9. **Reporting**: Generate comprehensive analysis summaries

## Key Outputs

### Visualizations
- Sentiment distribution and trends
- Technical indicator charts
- Correlation heatmaps
- Sentiment-price relationship plots
- Trading signal charts

### Metrics
- Sentiment statistics (VADER, TextBlob)
- Technical indicator values
- Financial performance metrics
- Correlation coefficients
- Trading signal performance

### Data Files
- Processed stock data with indicators
- News data with sentiment scores
- Integrated datasets for further analysis

## Technical Details

### Sentiment Analysis Methods
- **VADER**: Valence Aware Dictionary and sEntiment Reasoner
  - Optimized for social media and financial text
  - Provides compound, positive, negative, and neutral scores
- **TextBlob**: General-purpose sentiment analysis
  - Polarity (-1 to 1) and subjectivity (0 to 1) scores
  - Good for comparative analysis

### Technical Indicators
- **Moving Averages**: 20, 50, and 200-day SMAs
- **RSI**: Relative Strength Index (14-period)
- **MACD**: Moving Average Convergence Divergence
- **Financial Metrics**: Returns, volatility, Sharpe ratio, drawdown

### Integration Methods
- **Date Alignment**: Resample sentiment data to daily frequency
- **Correlation Analysis**: Pearson correlation between sentiment and returns
- **Lagged Analysis**: Test sentiment as leading indicator
- **Volume Integration**: Combine sentiment with trading volume analysis

## Performance Considerations

- **Data Size**: Handles datasets with thousands of articles and price points
- **Processing Time**: Sentiment analysis scales linearly with article count
- **Memory Usage**: Efficient pandas operations for large datasets
- **Caching**: Processed data can be saved and reloaded

## Troubleshooting

### Common Issues
1. **NLTK Data Missing**: Run the NLTK download commands
2. **TA-Lib Installation**: Use `talib-binary` for easier installation
3. **Data Compatibility**: Ensure date ranges overlap between datasets
4. **Memory Issues**: Process smaller date ranges for very large datasets

### Error Messages
- Check data column names match requirements
- Verify date formats are consistent
- Ensure sufficient data overlap for correlation analysis

## Future Enhancements

- **Real-time Data**: Integration with live news feeds
- **Machine Learning**: Sentiment-based prediction models
- **Multi-asset Analysis**: Portfolio-level sentiment analysis
- **API Integration**: Real-time stock data and news APIs
- **Web Interface**: Interactive dashboard for analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with data usage terms and regulations.

## Contact

For questions or issues, please open an issue in the repository or contact the development team.

---

**Note**: This project demonstrates the integration of sentiment analysis with quantitative financial analysis. Results should not be considered as financial advice. Always conduct thorough research before making investment decisions.