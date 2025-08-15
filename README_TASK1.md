# Task 1: Exploratory Data Analysis (EDA)

## Overview
This task performs comprehensive Exploratory Data Analysis on news data for the Movie Price Prediction project. The analysis covers all required areas specified in the task requirements.

## Requirements Completed

### ✅ Descriptive Statistics
- Basic statistics for textual lengths (headline length, word count)
- Count of articles per publisher to identify most active publishers
- Analysis of publication dates and trends over time

### ✅ Text Analysis (Topic Modeling)
- Natural language processing to identify common keywords and phrases
- Extraction of financial terms and company-specific terminology
- Topic identification from headlines

### ✅ Time Series Analysis
- Publication frequency variation over time
- Identification of spikes in article publications
- Analysis of publishing times (hour of day, day of week patterns)

### ✅ Publisher Analysis
- Analysis of most active publishers
- Publisher diversity and content analysis
- Email domain analysis (if applicable)

## Files Created

1. **`task1_eda_analysis.py`** - Main EDA analysis script
2. **`requirements_task1.txt`** - Dependencies for Task 1
3. **`README_TASK1.md`** - This documentation file

## Generated Output Files

When you run the analysis, the following files will be created:

1. **`news_eda_visualizations.png`** - Comprehensive visualization dashboard
2. **`monthly_trends.png`** - Monthly publication trends
3. **`top_publishers.png`** - Top publishers analysis
4. **`news_eda_report.md`** - Detailed analysis report

## How to Run

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements_task1.txt
```

### Run the Analysis
```bash
python task1_eda_analysis.py
```

## Analysis Features

### 1. Descriptive Statistics
- Total article count and date ranges
- Headline length statistics (mean, median, min, max)
- Word count analysis
- Top publishers and stocks by coverage

### 2. Text Analysis
- Keyword extraction and frequency analysis
- Financial terminology identification
- Company and industry term analysis
- Stop word filtering for meaningful insights

### 3. Time Series Analysis
- Daily publication patterns
- Day of week analysis
- Hour of day patterns
- Monthly trends and seasonal patterns
- Spike detection for unusual activity

### 4. Publisher Analysis
- Publisher activity rankings
- Content diversity analysis
- Email domain analysis (if applicable)
- Publisher-specific statistics

### 5. Visualizations
- Comprehensive dashboard with 6 key charts
- Time series plots
- Distribution histograms
- Bar charts for rankings
- Publication pattern analysis

## Data Source
The analysis uses the `raw_analyst_ratings.csv` file located in `data/raw_analyst/` directory.

## Expected Output
The script will provide:
- Console output with detailed statistics
- Multiple visualization files
- A comprehensive markdown report
- Detailed analysis of all required areas

## Commit Strategy
Remember to commit your work at least three times a day with descriptive commit messages as required by the task.

## Next Steps
After completing this EDA analysis, you can:
1. Use the insights for feature engineering
2. Identify patterns for trading strategies
3. Understand news sentiment timing
4. Optimize data collection strategies
