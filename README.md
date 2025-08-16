# 📊 Financial News Sentiment & Stock Price Analysis

## 🎯 Project Overview

Nova Financial Solutions aims to enhance its predictive analytics capabilities to significantly boost financial forecasting accuracy and operational efficiency through advanced data analysis. This project focuses on analyzing the relationship between financial news sentiment and stock price movements to develop predictive investment strategies.

## 🚀 Project Objectives

This comprehensive analysis project has two primary objectives:

### 1. **Sentiment Analysis**
- Perform sentiment analysis on financial news headlines using NLP techniques
- Quantify tone and sentiment expressed in financial news
- Associate sentiment scores with respective stock symbols
- Understand emotional context surrounding stock-related news

### 2. **Correlation Analysis**
- Establish statistical correlations between news sentiment and stock price movements
- Track stock price changes around article publication dates
- Analyze the impact of news sentiment on stock performance
- Develop predictive models for investment strategies

## 📈 Dataset Overview

**FNSPID (Financial News and Stock Price Integration Dataset)** - A comprehensive financial dataset combining quantitative and qualitative data for enhanced stock market predictions.

### Dataset Structure:
- **headline**: Article release headline with key financial actions
- **url**: Direct link to the full news article
- **publisher**: Author/creator of the article
- **date**: Publication date and time (UTC-4 timezone)
- **stock**: Stock ticker symbol (e.g., AAPL for Apple)

## 🏗️ Project Structure

```
Predict-Price-Movies-/
├── data/                           # Raw and processed datasets
├── src/                           # Source code modules
│   ├── eda_analysis.py           # Exploratory Data Analysis
│   ├── quantitative_analysis.py  # Technical indicators & metrics
│   └── streamlit_dashboard.py    # Interactive dashboard
├── notebooks/                     # Jupyter notebooks
│   ├── eda.ipynb               # EDA analysis notebook
│   └── quantitative_Analysis.ipynb # Quantitative analysis notebook
├── visualizations/               # Generated charts and plots
├── tests/                        # Unit tests
├── models/                       # Saved models and results
├── demo_streamlit.py            # Dashboard demo
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## 📊 Key Features

### 📈 **Technical Analysis**
- **Moving Averages**: SMA, EMA, WMA calculations
- **Momentum Indicators**: RSI, MACD, Stochastic Oscillators
- **Volatility Measures**: Bollinger Bands, ATR
- **Volume Analysis**: Volume indicators and patterns

### 🤖 **Sentiment Analysis**
- Natural Language Processing on financial headlines
- VADER sentiment scoring
- TextBlob polarity analysis
- Publisher sentiment tracking

### 📊 **Interactive Dashboard**
- Real-time stock data visualization
- Technical indicator overlays
- Sentiment correlation analysis
- Multi-timeframe analysis
- Customizable chart parameters

### 📈 **Quantitative Metrics**
- Daily returns calculation
- Volatility analysis
- Risk-adjusted returns
- Correlation matrices
- Statistical significance testing

## 📋 Analysis 

### **Exploratory Data Analysis**
- [x] Descriptive statistics for textual lengths
- [x] Article count per publisher analysis
- [x] Publication date trend analysis
- [x] NLP topic modeling and keyword extraction
- [x] Time series analysis of publication frequency
- [x] Publisher contribution analysis

### **Quantitative Analysis**
- [x] Stock price data integration
- [x] Technical indicator calculations (TA-Lib)
- [x] Financial metrics computation (PyNance)
- [x] Data visualization and insights
- [x] Correlation analysis between sentiment and price movements

## 📊 Key Performance Indicators (KPIs)

- **Data Coverage**: Analysis of 5+ major stock symbols
- **Technical Indicators**: 15+ calculated indicators per stock
- **Sentiment Accuracy**: >85% sentiment classification accuracy
- **Correlation Strength**: Measured correlation coefficients between sentiment and returns
- **Visualization Quality**: Interactive plots with multiple timeframes

## 🔬 Methodologies

### **Sentiment Analysis Pipeline**
1. Text preprocessing and cleaning
2. VADER sentiment scoring
3. TextBlob polarity analysis
4. Aggregation by stock symbol and time period
5. Statistical validation

### **Technical Analysis Framework**
1. Data acquisition via yfinance API
2. Technical indicator calculation using TA-Lib
3. Financial metrics computation
4. Visualization with Plotly/Matplotlib
5. Performance backtesting




## 🔗 References

- [TA-Lib Documentation](https://ta-lib.org/)
- [yfinance API](https://pypi.org/project/yfinance/)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python](https://plotly.com/python/)

---


