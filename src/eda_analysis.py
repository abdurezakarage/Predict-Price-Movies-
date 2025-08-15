
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
from collections import Counter
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NewsEDA:
    def __init__(self, data_path: str):
        """Initialize the EDA analysis with the news data path."""
        self.data_path = data_path
        self.news_data = None
        self.stock_data = {}
        
        # Create visualizations directory
        self.viz_dir = "../visualizations"
        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)
            print(f"Created visualizations directory: {self.viz_dir}")
        
        # Create processed data directory for further analysis
        self.processed_data_dir = "../data/processed_data"
        if not os.path.exists(self.processed_data_dir):
            os.makedirs(self.processed_data_dir)
            print(f"Created processed data directory: {self.processed_data_dir}")
        
    def load_data(self):
        """Load the news data and perform initial cleaning."""
        print("Loading news data...")
        try:
            self.news_data = pd.read_csv(self.data_path)
            print(f"Successfully loaded {len(self.news_data)} news articles")
            
            # Basic cleaning
            self.news_data['date'] = pd.to_datetime(self.news_data['date'], errors='coerce')
            self.news_data = self.news_data.dropna(subset=['date'])
            
            # Extract text length features
            self.news_data['headline_length'] = self.news_data['headline'].str.len()
            self.news_data['headline_word_count'] = self.news_data['headline'].str.split().str.len()
            
            # Extract time features
            self.news_data['hour'] = self.news_data['date'].dt.hour
            self.news_data['day_of_week'] = self.news_data['date'].dt.day_name()
            self.news_data['month'] = self.news_data['date'].dt.month
            self.news_data['year'] = self.news_data['date'].dt.year
            
            print("Data loaded and cleaned successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def descriptive_statistics(self):
        """Perform descriptive statistics analysis."""
        print("\n" + "="*50)
        print("DESCRIPTIVE STATISTICS")
        print("="*50)
        
        # Basic info
        print(f"Total articles: {len(self.news_data):,}")
        print(f"Date range: {self.news_data['date'].min()} to {self.news_data['date'].max()}")
        print(f"Unique publishers: {self.news_data['publisher'].nunique()}")
        print(f"Unique stocks: {self.news_data['stock'].nunique()}")
        
        # Headline length statistics
        print(f"\nHeadline Length Statistics:")
        print(f"  Mean length: {self.news_data['headline_length'].mean():.1f} characters")
        print(f"  Median length: {self.news_data['headline_length'].median():.1f} characters")
        print(f"  Min length: {self.news_data['headline_length'].min()} characters")
        print(f"  Max length: {self.news_data['headline_length'].max()} characters")
        
        # Word count statistics
        print(f"\nHeadline Word Count Statistics:")
        print(f"  Mean words: {self.news_data['headline_word_count'].mean():.1f} words")
        print(f"  Median words: {self.news_data['headline_word_count'].median():.1f} words")
        
        # Articles per publisher
        publisher_counts = self.news_data['publisher'].value_counts()
        print(f"\nTop 10 Most Active Publishers:")
        for i, (publisher, count) in enumerate(publisher_counts.head(10).items(), 1):
            print(f"  {i:2d}. {publisher}: {count:,} articles")
        
        # Articles per stock
        stock_counts = self.news_data['stock'].value_counts()
        print(f"\nTop 10 Most Covered Stocks:")
        for i, (stock, count) in enumerate(stock_counts.head(10).items(), 1):
            print(f"  {i:2d}. {stock}: {count:,} articles")
    
    def text_analysis(self):
        """Perform text analysis and topic modeling."""
        print("\n" + "="*50)
        print("TEXT ANALYSIS & TOPIC MODELING")
        print("="*50)
        
        # Extract common keywords
        all_headlines = ' '.join(self.news_data['headline'].astype(str))
        
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_headlines.lower())
        
        # Remove common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'that', 'with', 'this', 'from', 'they', 'have', 'will', 'your', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were', 'what', 'into', 'more', 'only', 'other', 'some', 'then', 'these', 'those', 'upon', 'would', 'about', 'after', 'again', 'against', 'could', 'every', 'first', 'found', 'great', 'house', 'large', 'might', 'never', 'often', 'place', 'right', 'small', 'sound', 'still', 'their', 'there', 'under', 'water', 'where', 'while', 'world', 'years', 'three', 'never', 'become', 'before', 'between', 'country', 'during', 'family', 'however', 'important', 'interest', 'little', 'money', 'mother', 'number', 'people', 'picture', 'public', 'school', 'should', 'system', 'through', 'together', 'without', 'always', 'another', 'because', 'company', 'country', 'different', 'example', 'government', 'important', 'interest', 'letter', 'nothing', 'question', 'something', 'sometimes', 'through', 'together', 'without', 'always', 'another', 'because', 'company', 'country', 'different', 'example', 'government', 'important', 'interest', 'letter', 'nothing', 'question', 'something', 'sometimes'}
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Get most common words
        word_counts = Counter(filtered_words)
        print("Top 20 Most Common Keywords:")
        for i, (word, count) in enumerate(word_counts.most_common(20), 1):
            print(f"  {i:2d}. {word}: {count:,} occurrences")
        
        # Extract financial terms
        financial_terms = ['stock', 'price', 'target', 'rating', 'analyst', 'earnings', 'revenue', 'profit', 'loss', 'market', 'trading', 'investment', 'dividend', 'growth', 'decline', 'increase', 'decrease', 'high', 'low', 'volume', 'market', 'cap', 'valuation', 'pe', 'ratio', 'beta', 'alpha', 'return', 'risk', 'portfolio', 'fund', 'etf', 'option', 'futures', 'derivative', 'hedge', 'arbitrage', 'technical', 'fundamental', 'chart', 'pattern', 'support', 'resistance', 'breakout', 'breakdown', 'trend', 'momentum', 'volatility', 'correlation', 'correlation', 'correlation']
        
        financial_keywords = {}
        for term in financial_terms:
            count = sum(1 for headline in self.news_data['headline'] if term.lower() in headline.lower())
            if count > 0:
                financial_keywords[term] = count
        
        print(f"\nFinancial Terms Analysis:")
        for term, count in sorted(financial_keywords.items(), key=lambda x: x[1], reverse=True):
            print(f"  {term}: {count:,} articles")
        
        # Extract company-specific terms
        company_terms = ['apple', 'microsoft', 'google', 'amazon', 'tesla', 'meta', 'facebook', 'netflix', 'nvidia', 'amd', 'intel', 'oracle', 'salesforce', 'adobe', 'paypal', 'visa', 'mastercard', 'jpmorgan', 'goldman', 'morgan', 'bank', 'financial', 'tech', 'technology', 'software', 'hardware', 'semiconductor', 'automotive', 'electric', 'vehicle', 'ev', 'ai', 'artificial', 'intelligence', 'machine', 'learning', 'cloud', 'saas', 'ecommerce', 'retail', 'healthcare', 'biotech', 'pharmaceutical', 'energy', 'oil', 'gas', 'renewable', 'solar', 'wind', 'battery', 'lithium', 'crypto', 'bitcoin', 'ethereum', 'blockchain', 'defi', 'nft']
        
        company_keywords = {}
        for term in company_terms:
            count = sum(1 for headline in self.news_data['headline'] if term.lower() in headline.lower())
            if count > 0:
                company_keywords[term] = count
        
        print(f"\nCompany/Industry Terms Analysis:")
        for term, count in sorted(company_keywords.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"  {term}: {count:,} articles")
    
    def time_series_analysis(self):
        """Perform time series analysis."""
        print("\n" + "="*50)
        print("TIME SERIES ANALYSIS")
        print("="*50)
        
        # Publication frequency over time
        daily_counts = self.news_data.groupby(self.news_data['date'].dt.date).size()
        
        print(f"Publication Frequency Analysis:")
        print(f"  Average articles per day: {daily_counts.mean():.1f}")
        print(f"  Median articles per day: {daily_counts.median():.1f}")
        print(f"  Min articles per day: {daily_counts.min()}")
        print(f"  Max articles per day: {daily_counts.max()}")
        
        # Day of week analysis
        dow_counts = self.news_data['day_of_week'].value_counts()
        print(f"\nArticles by Day of Week:")
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            if day in dow_counts:
                count = dow_counts[day]
                print(f"  {day}: {count:,} articles")
        
        # Hour of day analysis
        hour_counts = self.news_data['hour'].value_counts().sort_index()
        print(f"\nArticles by Hour of Day:")
        for hour in range(24):
            if hour in hour_counts:
                count = hour_counts[hour]
                print(f"  {hour:02d}:00 - {hour:02d}:59: {count:,} articles")
        
        # Monthly trends
        monthly_counts = self.news_data.groupby([self.news_data['date'].dt.year, self.news_data['date'].dt.month]).size()
        print(f"\nMonthly Publication Trends:")
        for (year, month), count in monthly_counts.tail(12).items():
            month_name = datetime(year, month, 1).strftime('%B %Y')
            print(f"  {month_name}: {count:,} articles")
        
        # Identify spikes (days with unusually high publication counts)
        mean_daily = daily_counts.mean()
        std_daily = daily_counts.std()
        spike_threshold = mean_daily + 2 * std_daily
        
        spike_days = daily_counts[daily_counts > spike_threshold]
        if not spike_days.empty:
            print(f"\nDays with Unusually High Publication Counts (> {spike_threshold:.1f}):")
            for date, count in spike_days.items():
                print(f"  {date}: {count} articles")
    
    def publisher_analysis(self):
        """Perform publisher analysis."""
        print("\n" + "="*50)
        print("PUBLISHER ANALYSIS")
        print("="*50)
        
        # Publisher statistics
        publisher_stats = self.news_data.groupby('publisher').agg({
            'headline': 'count',
            'headline_length': ['mean', 'std'],
            'headline_word_count': ['mean', 'std'],
            'stock': 'nunique'
        }).round(2)
        
        publisher_stats.columns = ['article_count', 'avg_headline_length', 'std_headline_length', 
                                 'avg_word_count', 'std_word_count', 'unique_stocks']
        publisher_stats = publisher_stats.sort_values('article_count', ascending=False)
        
        print("Publisher Statistics (Top 15):")
        print(publisher_stats.head(15))
        
        # Publisher diversity analysis
        print(f"\nPublisher Diversity Analysis:")
        print(f"  Total publishers: {self.news_data['publisher'].nunique()}")
        print(f"  Publishers with >100 articles: {(publisher_stats['article_count'] > 100).sum()}")
        print(f"  Publishers with >1000 articles: {(publisher_stats['article_count'] > 1000).sum()}")
        
        # Email domain analysis (if publishers are email addresses)
        email_publishers = self.news_data[self.news_data['publisher'].str.contains('@', na=False)]
        if not email_publishers.empty:
            print(f"\nEmail-based Publishers Analysis:")
            print(f"  Articles from email publishers: {len(email_publishers)}")
            
            # Extract domains
            domains = email_publishers['publisher'].str.extract(r'@([^@]+)')[0]
            domain_counts = domains.value_counts()
            print(f"  Top email domains:")
            for domain, count in domain_counts.head(10).items():
                print(f"    {domain}: {count} articles")
        
        # Publisher content analysis
        print(f"\nPublisher Content Analysis:")
        for publisher in publisher_stats.head(5).index:
            pub_data = self.news_data[self.news_data['publisher'] == publisher]
            print(f"\n  {publisher}:")
            print(f"    Total articles: {len(pub_data):,}")
            print(f"    Average headline length: {pub_data['headline_length'].mean():.1f} characters")
            print(f"    Covers {pub_data['stock'].nunique()} unique stocks")
            print(f"    Date range: {pub_data['date'].min()} to {pub_data['date'].max()}")
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        # Set up the plotting area
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('News Data EDA Visualizations', fontsize=16, fontweight='bold')
        
        # 1. Headline length distribution
        axes[0, 0].hist(self.news_data['headline_length'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Headline Length Distribution')
        axes[0, 0].set_xlabel('Headline Length (characters)')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Articles per day over time
        daily_counts = self.news_data.groupby(self.news_data['date'].dt.date).size()
        axes[0, 1].plot(daily_counts.index, daily_counts.values, linewidth=1, alpha=0.7)
        axes[0, 1].set_title('Articles Published per Day Over Time')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Number of Articles')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Articles by day of week
        dow_counts = self.news_data['day_of_week'].value_counts()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_counts_ordered = dow_counts.reindex(dow_order)
        axes[0, 2].bar(range(len(dow_counts_ordered)), dow_counts_ordered.values, color='lightcoral')
        axes[0, 2].set_title('Articles by Day of Week')
        axes[0, 2].set_xlabel('Day of Week')
        axes[0, 2].set_ylabel('Number of Articles')
        axes[0, 2].set_xticks(range(len(dow_counts_ordered)))
        axes[0, 2].set_xticklabels(dow_counts_ordered.index, rotation=45)
        
        # 4. Articles by hour of day
        hour_counts = self.news_data['hour'].value_counts().sort_index()
        axes[1, 0].bar(hour_counts.index, hour_counts.values, color='lightgreen')
        axes[1, 0].set_title('Articles by Hour of Day')
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Number of Articles')
        
        # 5. Top publishers
        top_publishers = self.news_data['publisher'].value_counts().head(10)
        axes[1, 1].barh(range(len(top_publishers)), top_publishers.values, color='gold')
        axes[1, 1].set_title('Top 10 Publishers by Article Count')
        axes[1, 1].set_xlabel('Number of Articles')
        axes[1, 1].set_yticks(range(len(top_publishers)))
        axes[1, 1].set_yticklabels(top_publishers.index)
        
        # 6. Top stocks covered
        top_stocks = self.news_data['stock'].value_counts().head(10)
        axes[1, 2].barh(range(len(top_stocks)), top_stocks.values, color='lightblue')
        axes[1, 2].set_title('Top 10 Stocks by Article Count')
        axes[1, 2].set_xlabel('Number of Articles')
        axes[1, 2].set_yticks(range(len(top_stocks)))
        axes[1, 2].set_yticklabels(top_stocks.index)
        
        plt.tight_layout()
        viz_path = os.path.join(self.viz_dir, 'news_eda_visualizations.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"Visualizations saved as '{viz_path}'")
        
        # Create additional detailed plots
        self._create_detailed_plots()
    
    def _create_detailed_plots(self):
        """Create additional detailed plots."""
        # Monthly trends
        monthly_data = self.news_data.groupby([self.news_data['date'].dt.year, self.news_data['date'].dt.month]).size()
        monthly_data.index = [f"{year}-{month:02d}" for year, month in monthly_data.index]
        
        plt.figure(figsize=(15, 6))
        monthly_data.plot(kind='line', marker='o', linewidth=2, markersize=6)
        plt.title('Monthly Publication Trends', fontsize=14, fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        viz_path = os.path.join(self.viz_dir, 'monthly_trends.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        
        # Publisher content analysis
        top_publishers = self.news_data['publisher'].value_counts().head(15)
        plt.figure(figsize=(12, 8))
        top_publishers.plot(kind='barh', color='coral')
        plt.title('Top 15 Publishers by Article Count', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Articles')
        plt.ylabel('Publisher')
        plt.tight_layout()
        viz_path = os.path.join(self.viz_dir, 'top_publishers.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        
        print("Additional detailed plots saved")
    
    def save_processed_data(self):
        """Save cleaned and processed data for further analysis."""
        print("\n" + "="*50)
        print("SAVING PROCESSED DATA FOR FURTHER ANALYSIS")
        print("="*50)
        
        # Save the main cleaned dataset
        main_data_path = os.path.join(self.processed_data_dir, "cleaned_news_data.csv")
        self.news_data.to_csv(main_data_path, index=False)
        print(f"Main cleaned dataset saved: {main_data_path}")
        
        # Save publisher statistics
        publisher_stats = self.news_data.groupby('publisher').agg({
            'headline': 'count',
            'headline_length': ['mean', 'std'],
            'headline_word_count': ['mean', 'std'],
            'stock': 'nunique'
        }).round(2)
        publisher_stats.columns = ['article_count', 'avg_headline_length', 'std_headline_length', 
                                 'avg_word_count', 'std_word_count', 'unique_stocks']
        publisher_stats = publisher_stats.sort_values('article_count', ascending=False)
        
        publisher_stats_path = os.path.join(self.processed_data_dir, "publisher_statistics.csv")
        publisher_stats.to_csv(publisher_stats_path)
        print(f"Publisher statistics saved: {publisher_stats_path}")
        
        # Save daily publication counts
        daily_counts = self.news_data.groupby(self.news_data['date'].dt.date).size()
        daily_counts_df = pd.DataFrame({
            'date': daily_counts.index,
            'article_count': daily_counts.values
        })
        daily_counts_path = os.path.join(self.processed_data_dir, "daily_publication_counts.csv")
        daily_counts_df.to_csv(daily_counts_path, index=False)
        print(f"Daily publication counts saved: {daily_counts_path}")
        
        # Save stock coverage statistics
        stock_counts = self.news_data['stock'].value_counts()
        stock_stats_df = pd.DataFrame({
            'stock': stock_counts.index,
            'article_count': stock_counts.values
        })
        stock_stats_path = os.path.join(self.processed_data_dir, "stock_coverage_statistics.csv")
        stock_stats_df.to_csv(stock_stats_path, index=False)
        print(f"Stock coverage statistics saved: {stock_stats_path}")
        
        # Save keyword analysis results
        all_headlines = ' '.join(self.news_data['headline'].astype(str))
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_headlines.lower())
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'that', 'with', 'this', 'from', 'they', 'have', 'will', 'your', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were', 'what', 'into', 'more', 'only', 'other', 'some', 'then', 'these', 'those', 'upon', 'would', 'about', 'after', 'again', 'against', 'could', 'every', 'first', 'found', 'great', 'house', 'large', 'might', 'never', 'often', 'place', 'right', 'small', 'sound', 'still', 'their', 'there', 'under', 'water', 'where', 'while', 'world', 'years', 'three', 'never', 'become', 'before', 'between', 'country', 'during', 'family', 'however', 'important', 'interest', 'little', 'money', 'mother', 'number', 'people', 'picture', 'public', 'school', 'should', 'system', 'through', 'together', 'without', 'always', 'another', 'because', 'company', 'country', 'different', 'example', 'government', 'important', 'interest', 'letter', 'nothing', 'question', 'something', 'sometimes', 'through', 'together', 'without', 'always', 'another', 'because', 'company', 'country', 'different', 'example', 'government', 'important', 'interest', 'letter', 'nothing', 'question', 'something', 'sometimes'}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        word_counts = Counter(filtered_words)
        
        keyword_df = pd.DataFrame(word_counts.most_common(100), columns=['keyword', 'frequency'])
        keyword_path = os.path.join(self.processed_data_dir, "top_keywords.csv")
        keyword_df.to_csv(keyword_path, index=False)
        print(f"Top keywords saved: {keyword_path}")
        
        # Save financial terms analysis
        financial_terms = ['stock', 'price', 'target', 'rating', 'analyst', 'earnings', 'revenue', 'profit', 'loss', 'market', 'trading', 'investment', 'dividend', 'growth', 'decline', 'increase', 'decrease', 'high', 'low', 'volume', 'market', 'cap', 'valuation', 'pe', 'ratio', 'beta', 'alpha', 'return', 'risk', 'portfolio', 'fund', 'etf', 'option', 'futures', 'derivative', 'hedge', 'arbitrage', 'technical', 'fundamental', 'chart', 'pattern', 'support', 'resistance', 'breakout', 'breakdown', 'trend', 'momentum', 'volatility', 'correlation']
        
        financial_keywords = {}
        for term in financial_terms:
            count = sum(1 for headline in self.news_data['headline'] if term.lower() in headline.lower())
            if count > 0:
                financial_keywords[term] = count
        
        financial_df = pd.DataFrame(list(financial_keywords.items()), columns=['term', 'article_count'])
        financial_df = financial_df.sort_values('article_count', ascending=False)
        financial_path = os.path.join(self.processed_data_dir, "financial_terms_analysis.csv")
        financial_df.to_csv(financial_path, index=False)
        print(f"Financial terms analysis saved: {financial_path}")
        
        # Save time-based features
        time_features = self.news_data[['date', 'hour', 'day_of_week', 'month', 'year']].copy()
        time_features_path = os.path.join(self.processed_data_dir, "time_features.csv")
        time_features.to_csv(time_features_path, index=False)
        print(f"Time features saved: {time_features_path}")
        
        # Create a summary metadata file
        metadata = {
            'total_articles': len(self.news_data),
            'date_range_start': str(self.news_data['date'].min()),
            'date_range_end': str(self.news_data['date'].max()),
            'unique_publishers': self.news_data['publisher'].nunique(),
            'unique_stocks': self.news_data['stock'].nunique(),
            'avg_headline_length': float(self.news_data['headline_length'].mean()),
            'avg_word_count': float(self.news_data['headline_word_count'].mean()),
            'data_processing_timestamp': datetime.now().isoformat()
        }
        
        metadata_df = pd.DataFrame(list(metadata.items()), columns=['metric', 'value'])
        metadata_path = os.path.join(self.processed_data_dir, "dataset_metadata.csv")
        metadata_df.to_csv(metadata_path, index=False)
        print(f"Dataset metadata saved: {metadata_path}")
        
        print(f"\nAll processed data saved in: {self.processed_data_dir}/")
    
    def generate_report(self):
        """Generate a comprehensive EDA report."""
        print("\n" + "="*50)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*50)
        
        report = []
        report.append("# News Data EDA Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data source: {self.data_path}")
        report.append("")
        
        # Summary statistics
        report.append("## Summary Statistics")
        report.append(f"- Total articles: {len(self.news_data):,}")
        report.append(f"- Date range: {self.news_data['date'].min()} to {self.news_data['date'].max()}")
        report.append(f"- Unique publishers: {self.news_data['publisher'].nunique()}")
        report.append(f"- Unique stocks: {self.news_data['stock'].nunique()}")
        report.append(f"- Average headline length: {self.news_data['headline_length'].mean():.1f} characters")
        report.append("")
        
        # Top publishers
        top_publishers = self.news_data['publisher'].value_counts().head(10)
        report.append("## Top 10 Publishers")
        for i, (publisher, count) in enumerate(top_publishers.items(), 1):
            report.append(f"{i}. {publisher}: {count:,} articles")
        report.append("")
        
        # Top stocks
        top_stocks = self.news_data['stock'].value_counts().head(10)
        report.append("## Top 10 Most Covered Stocks")
        for i, (stock, count) in enumerate(top_stocks.items(), 1):
            report.append(f"{i}. {stock}: {count:,} articles")
        report.append("")
        
        # Time analysis
        daily_counts = self.news_data.groupby(self.news_data['date'].dt.date).size()
        report.append("## Time Analysis")
        report.append(f"- Average articles per day: {daily_counts.mean():.1f}")
        report.append(f"- Busiest day: {daily_counts.idxmax()} with {daily_counts.max()} articles")
        report.append(f"- Quietest day: {daily_counts.idxmin()} with {daily_counts.min()} articles")
        report.append("")
        
        # Save report
        with open('news_eda_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("Comprehensive report saved as 'news_eda_report.md'")
    
    def run_complete_analysis(self):
        """Run the complete EDA analysis."""
        print("Starting comprehensive EDA analysis...")
        
        if not self.load_data():
            return
        
        # Run all analyses
        self.descriptive_statistics()
        self.text_analysis()
        self.time_series_analysis()
        self.publisher_analysis()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        self.generate_report()
        
        # Save processed data for further analysis
        self.save_processed_data()
        
        print("\n" + "="*50)
        print("EDA ANALYSIS COMPLETE!")
        print("="*50)
        print("Generated files:")
        print(f"- {self.viz_dir}/news_eda_visualizations.png")
        print(f"- {self.viz_dir}/monthly_trends.png")
        print(f"- {self.viz_dir}/top_publishers.png")
        print("- news_eda_report.md")
        print(f"- {self.processed_data_dir}/ (processed data for further analysis)")

def main():
    """Main function to run the EDA analysis."""
    # Initialize the EDA analysis
    eda = NewsEDA("data/raw_analyst/raw_analyst_ratings.csv")
    
    # Run the complete analysis
    eda.run_complete_analysis()

if __name__ == "__main__":
    main()
