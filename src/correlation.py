import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsStockCorrelation:
    def __init__(self):
        """Initialize the NewsStockCorrelation class."""
        self.news_data = None
        self.stock_data = None
        self.aligned_data = None
        self.correlation_results = None

    def debug_data_info(self) -> None:
        """
        Debug method to inspect data types and structure.
        """
        print("=== DEBUG INFO ===")
        if self.news_data is not None:
            print(f"News data shape: {self.news_data.shape}")
            print(f"News data columns: {self.news_data.columns.tolist()}")
            print(f"News data sample (first 3 rows):")
            print(self.news_data.head(3))
            if 'date' in self.news_data.columns:
                print(f"News date dtype: {self.news_data['date'].dtype}")
                print(f"News date sample: {self.news_data['date'].head(3).tolist()}")
                print(f"News date range: {self.news_data['date'].min()} to {self.news_data['date'].max()}")
        else:
            print("News data: None")
            
        if self.stock_data is not None:
            print(f"Stock data shape: {self.stock_data.shape}")
            print(f"Stock data columns: {self.stock_data.columns.tolist()}")
            print(f"Stock data sample (first 3 rows):")
            print(self.stock_data.head(3))
            if 'date' in self.stock_data.columns:
                print(f"Stock date dtype: {self.stock_data['date'].dtype}")
                print(f"Stock date sample: {self.stock_data['date'].head(3).tolist()}")
                print(f"Stock date range: {self.stock_data['date'].min()} to {self.stock_data['date'].max()}")
        else:
            print("Stock data: None")
            
        if self.aligned_data is not None:
            print(f"Aligned data shape: {self.aligned_data.shape}")
            print(f"Aligned data columns: {self.aligned_data.columns.tolist()}")
            if not self.aligned_data.empty:
                print(f"Aligned data date range: {self.aligned_data['date'].min()} to {self.aligned_data['date'].max()}")
                print(f"Sentiment score stats: min={self.aligned_data['sentiment_score'].min():.4f}, max={self.aligned_data['sentiment_score'].max():.4f}, mean={self.aligned_data['sentiment_score'].mean():.4f}")
                print(f"Daily return stats: min={self.aligned_data['daily_return'].min():.4f}, max={self.aligned_data['daily_return'].max():.4f}, mean={self.aligned_data['daily_return'].mean():.4f}")
                print(f"NaN counts - sentiment_score: {self.aligned_data['sentiment_score'].isna().sum()}, daily_return: {self.aligned_data['daily_return'].isna().sum()}")
        else:
            print("Aligned data: None")
        print("==================")

    def debug_step_by_step(self) -> None:
        """
        Debug method to check each step of the data preparation process.
        """
        print("=== STEP-BY-STEP DEBUG ===")
        
        # Step 1: Check if data is loaded
        print("1. DATA LOADING:")
        if self.news_data is not None and not self.news_data.empty:
            print("   ✓ News data loaded")
            print(f"   - Shape: {self.news_data.shape}")
            print(f"   - Columns: {self.news_data.columns.tolist()}")
        else:
            print("   ✗ News data not loaded or empty")
            
        if self.stock_data is not None and not self.stock_data.empty:
            print("   ✓ Stock data loaded")
            print(f"   - Shape: {self.stock_data.shape}")
            print(f"   - Columns: {self.stock_data.columns.tolist()}")
        else:
            print("   ✗ Stock data not loaded or empty")
        
        # Step 2: Check date normalization
        print("\n2. DATE NORMALIZATION:")
        if self.news_data is not None and 'date' in self.news_data.columns:
            print("   ✓ News dates normalized")
            print(f"   - Date range: {self.news_data['date'].min()} to {self.news_data['date'].max()}")
            print(f"   - Date type: {self.news_data['date'].dtype}")
        else:
            print("   ✗ News dates not normalized")
            
        if self.stock_data is not None and 'date' in self.stock_data.columns:
            print("   ✓ Stock dates normalized")
            print(f"   - Date range: {self.stock_data['date'].min()} to {self.stock_data['date'].max()}")
            print(f"   - Date type: {self.stock_data['date'].dtype}")
        else:
            print("   ✗ Stock dates not normalized")
        
        # Step 3: Check sentiment analysis
        print("\n3. SENTIMENT ANALYSIS:")
        if self.news_data is not None and 'sentiment_score' in self.news_data.columns:
            print("   ✓ Sentiment analysis completed")
            print(f"   - Sentiment range: {self.news_data['sentiment_score'].min():.4f} to {self.news_data['sentiment_score'].max():.4f}")
        else:
            print("   ✗ Sentiment analysis not completed")
        
        # Step 4: Check stock returns
        print("\n4. STOCK RETURNS:")
        if self.stock_data is not None and 'daily_return' in self.stock_data.columns:
            print("   ✓ Stock returns calculated")
            print(f"   - Returns range: {self.stock_data['daily_return'].min():.4f} to {self.stock_data['daily_return'].max():.4f}")
        else:
            print("   ✗ Stock returns not calculated")
        
        # Step 5: Check alignment
        print("\n5. DATASET ALIGNMENT:")
        if self.aligned_data is not None and not self.aligned_data.empty:
            print("   ✓ Datasets aligned")
            print(f"   - Final shape: {self.aligned_data.shape}")
            print(f"   - Date range: {self.aligned_data['date'].min()} to {self.aligned_data['date'].max()}")
        else:
            print("   ✗ Datasets not aligned")
            
            # Check for potential alignment issues
            if (self.news_data is not None and 'date' in self.news_data.columns and 
                self.stock_data is not None and 'date' in self.stock_data.columns):
                
                # Handle both datetime and date types
                if pd.api.types.is_datetime64_any_dtype(self.news_data['date']):
                    news_dates = set(self.news_data['date'].dt.date)
                else:
                    news_dates = set(self.news_data['date'])
                
                if pd.api.types.is_datetime64_any_dtype(self.stock_data['date']):
                    stock_dates = set(self.stock_data['date'].dt.date)
                else:
                    stock_dates = set(self.stock_data['date'])
                
                overlapping_dates = news_dates.intersection(stock_dates)
                
                print(f"   - News dates: {len(news_dates)}")
                print(f"   - Stock dates: {len(stock_dates)}")
                print(f"   - Overlapping dates: {len(overlapping_dates)}")
                
                if len(overlapping_dates) == 0:
                    print("   ⚠️  NO OVERLAPPING DATES - This is the problem!")
                else:
                    print(f"   - Sample overlapping: {list(overlapping_dates)[:3]}")
        
        print("========================")

    def test_date_column_detection(self) -> None:
        """
        Test method to verify date column detection works correctly.
        """
        print("=== TESTING DATE COLUMN DETECTION ===")
        
        if self.news_data is not None:
            print("Testing news data:")
            try:
                news_date_col = self._find_date_column(self.news_data, "news data")
                print(f"   ✓ Found date column: '{news_date_col}'")
            except Exception as e:
                print(f"   ✗ Error: {e}")
        
        if self.stock_data is not None:
            print("Testing stock data:")
            try:
                stock_date_col = self._find_date_column(self.stock_data, "stock data")
                print(f"   ✓ Found date column: '{stock_date_col}'")
            except Exception as e:
                print(f"   ✗ Error: {e}")
        
        print("===================================")

    def check_date_overlap(self) -> None:
        """
        Check if there are overlapping dates between news and stock data.
        """
        print("=== CHECKING DATE OVERLAP ===")
        
        if self.news_data is None or self.stock_data is None:
            print("   ✗ Data not loaded yet")
            return
        
        if 'date' not in self.news_data.columns or 'date' not in self.stock_data.columns:
            print("   ✗ Dates not normalized yet")
            return
        
        # Check for overlapping dates
        # Handle both datetime and date types
        if pd.api.types.is_datetime64_any_dtype(self.news_data['date']):
            news_dates = set(self.news_data['date'].dt.date)
        else:
            news_dates = set(self.news_data['date'])
        
        if pd.api.types.is_datetime64_any_dtype(self.stock_data['date']):
            stock_dates = set(self.stock_data['date'].dt.date)
        else:
            stock_dates = set(self.stock_data['date'])
        
        overlapping_dates = news_dates.intersection(stock_dates)
        
        print(f"   News dates count: {len(news_dates)}")
        print(f"   Stock dates count: {len(stock_dates)}")
        print(f"   Overlapping dates count: {len(overlapping_dates)}")
        
        if len(overlapping_dates) == 0:
            print("   ⚠️  NO OVERLAPPING DATES - This is the problem!")
            print("   Please check if your news and stock data cover the same time period.")
            
            # Show sample dates from both datasets
            print(f"   Sample news dates: {list(news_dates)[:5]}")
            print(f"   Sample stock dates: {list(stock_dates)[:5]}")
            
            # Check date ranges
            print(f"   News date range: {self.news_data['date'].min()} to {self.news_data['date'].max()}")
            print(f"   Stock date range: {self.stock_data['date'].min()} to {self.stock_data['date'].max()}")
        else:
            print(f"   ✓ Found overlapping dates")
            print(f"   Sample overlapping: {list(overlapping_dates)[:5]}")
        
        print("=============================")

    def test_merge_operation(self) -> None:
        """
        Test the merge operation step by step to identify issues.
        """
        print("=== TESTING MERGE OPERATION ===")
        
        if self.news_data is None or self.stock_data is None:
            print("   ✗ Data not loaded yet")
            return
        
        if 'date' not in self.news_data.columns or 'date' not in self.stock_data.columns:
            print("   ✗ Dates not normalized yet")
            return
        
        if 'sentiment_score' not in self.news_data.columns:
            print("   ✗ Sentiment analysis not completed yet")
            return
        
        if 'daily_return' not in self.stock_data.columns:
            print("   ✗ Stock returns not calculated yet")
            return
        
        print("   ✓ All prerequisites met")
        print(f"   News data shape: {self.news_data.shape}")
        print(f"   Stock data shape: {self.stock_data.shape}")
        
        # Test the merge operation
        try:
            print("   Testing merge operation...")
            test_merge = pd.merge(
                self.news_data,
                self.stock_data[['date', 'daily_return']],
                on='date',
                how='inner'
            )
            print(f"   ✓ Merge successful! Result shape: {test_merge.shape}")
            print(f"   Result columns: {test_merge.columns.tolist()}")
            
            if not test_merge.empty:
                print(f"   Sample data:")
                print(test_merge.head(2))
            else:
                print("   ⚠️  Merge resulted in empty dataset!")
                
        except Exception as e:
            print(f"   ✗ Merge failed with error: {e}")
        
        print("==============================")

    def load_data(self, news_df: pd.DataFrame, stock_df: pd.DataFrame) -> None:
        """
        Load news and stock data into the class.
        
        Args:
            news_df (pd.DataFrame): News dataset with date and headlines
            stock_df (pd.DataFrame): Stock price dataset with date and prices
        """
        # Validate input DataFrames
        if news_df is None or news_df.empty:
            raise ValueError("News DataFrame is empty or None")
        if stock_df is None or stock_df.empty:
            raise ValueError("Stock DataFrame is empty or None")
        
        # Debug: Show what columns we have
        logger.info(f"News data columns: {news_df.columns.tolist()}")
        logger.info(f"Stock data columns: {stock_df.columns.tolist()}")
        
        # Check for required columns
        if not any(col.lower() == 'date' for col in news_df.columns):
            raise ValueError("News DataFrame must contain a 'date' column (case-insensitive)")
        if not any(col.lower() == 'date' for col in stock_df.columns):
            raise ValueError("Stock DataFrame must contain a 'date' column (case-insensitive)")
        
        self.news_data = news_df.copy()
        self.stock_data = stock_df.copy()
        logger.info("Data loaded successfully")

    def _find_date_column(self, df: pd.DataFrame, dataset_name: str) -> str:
        """
        Helper method to find date column in a DataFrame (case-insensitive).
        
        Args:
            df (pd.DataFrame): DataFrame to search
            dataset_name (str): Name of the dataset for error messages
            
        Returns:
            str: Name of the date column
            
        Raises:
            ValueError: If no date column is found
        """
        # First try exact match for common variations
        exact_matches = ['date', 'Date', 'DATE']
        for exact in exact_matches:
            if exact in df.columns:
                return exact
        
        # Then try case-insensitive match
        for col in df.columns:
            if col.lower() == 'date':
                return col
        
        # If still not found, show available columns for debugging
        available_cols = df.columns.tolist()
        logger.error(f"Available columns in {dataset_name}: {available_cols}")
        raise ValueError(f"No date column found in {dataset_name}. Available columns: {available_cols}")

    def _find_price_column(self, df: pd.DataFrame, price_type: str, dataset_name: str) -> str:
        """
        Helper method to find price column in a DataFrame (case-insensitive).
        
        Args:
            df (pd.DataFrame): DataFrame to search
            price_type (str): Type of price column (e.g., 'close', 'open', 'high', 'low')
            dataset_name (str): Name of the dataset for error messages
            
        Returns:
            str: Name of the price column
            
        Raises:
            ValueError: If no price column is found
        """
        for col in df.columns:
            if col.lower() == price_type.lower():
                return col
        
        raise ValueError(f"No {price_type} column found in {dataset_name}")

    def _find_headline_column(self, df: pd.DataFrame, dataset_name: str) -> str:
        """
        Helper method to find headline column in a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to search
            dataset_name (str): Name of the dataset for error messages
            
        Returns:
            str: Name of the headline column
            
        Raises:
            ValueError: If no headline column is found
        """
        # Common headline column names
        headline_keywords = ['headline', 'title', 'text', 'content', 'news', 'article']
        
        # First try exact matches
        for keyword in headline_keywords:
            if keyword in df.columns:
                return keyword
        
        # Then try case-insensitive matches
        for col in df.columns:
            if any(keyword in col.lower() for keyword in headline_keywords):
                return col
        
        # If still not found, show available columns for debugging
        available_cols = df.columns.tolist()
        logger.error(f"Available columns in {dataset_name}: {available_cols}")
        raise ValueError(f"No headline column found in {dataset_name}. Available columns: {available_cols}")

    def normalize_dates(self) -> None:
        """
        Normalize and align dates between news and stock datasets.
        Ensures both datasets use the same date format and are properly aligned.
        """
        try:
            # Find date columns (case-insensitive)
            news_date_col = self._find_date_column(self.news_data, "news data")
            stock_date_col = self._find_date_column(self.stock_data, "stock data")
            
            logger.info(f"Found date columns: news='{news_date_col}', stock='{stock_date_col}'")
            
            # Convert dates to datetime and handle timezone differences
            self.news_data['date'] = pd.to_datetime(self.news_data[news_date_col]).dt.tz_localize(None)
            self.stock_data['date'] = pd.to_datetime(self.stock_data[stock_date_col]).dt.tz_localize(None)
            
            # Keep as datetime but normalize to date-only for consistent merging
            # This ensures we can still use .dt accessors later
            self.news_data['date'] = pd.to_datetime(self.news_data['date'].dt.date)
            self.stock_data['date'] = pd.to_datetime(self.stock_data['date'].dt.date)
            
            # Sort both datasets by date
            self.news_data = self.news_data.sort_values('date')
            self.stock_data = self.stock_data.sort_values('date')
            
            logger.info("Dates normalized successfully")
            logger.info(f"News date range: {self.news_data['date'].min()} to {self.news_data['date'].max()}")
            logger.info(f"Stock date range: {self.stock_data['date'].min()} to {self.stock_data['date'].max()}")
        except Exception as e:
            logger.error(f"Error normalizing dates: {str(e)}")
            raise

    def calculate_sentiment(self, text: str) -> float:
        """
        Calculate sentiment score for a given text using TextBlob.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Sentiment polarity score (-1 to 1)
        """
        try:
            return TextBlob(str(text)).sentiment.polarity
        except Exception as e:
            logger.warning(f"Error calculating sentiment for text: {str(e)}")
            return 0.0

    def analyze_sentiment(self, headline_column: str = 'headline') -> None:
        """
        Perform sentiment analysis on news headlines.
        
        Args:
            headline_column (str): Name of the column containing headlines
        """
        try:
            # First, let's check what columns are actually available
            if headline_column not in self.news_data.columns:
                logger.info(f"Column '{headline_column}' not found, attempting to auto-detect headline column...")
                headline_column = self._find_headline_column(self.news_data, "news data")
                logger.info(f"Auto-detected headline column: '{headline_column}'")
            
            # Check if the column contains text data
            if self.news_data[headline_column].dtype == 'object':
                # Convert to string and handle NaN values
                text_data = self.news_data[headline_column].astype(str).replace('nan', '')
                self.news_data['sentiment_score'] = text_data.apply(self.calculate_sentiment)
            else:
                raise ValueError(f"Column '{headline_column}' does not contain text data. Data type: {self.news_data[headline_column].dtype}")
            
            # Aggregate sentiments by date (mean of all articles per day)
            daily_sentiment = self.news_data.groupby('date')['sentiment_score'].mean().reset_index()
            self.news_data = daily_sentiment
            
            logger.info("Sentiment analysis completed successfully")
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            raise

    def calculate_stock_returns(self, price_column: str = 'close') -> None:
        """
        Calculate daily stock returns.
        
        Args:
            price_column (str): Name of the column containing stock prices
        """
        try:
            # Find the actual price column name (case-insensitive)
            actual_price_column = self._find_price_column(self.stock_data, price_column, "stock data")
            
            self.stock_data['daily_return'] = self.stock_data[actual_price_column].pct_change()
            self.stock_data = self.stock_data.dropna()  # Remove first row with NaN return
            logger.info("Stock returns calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating stock returns: {str(e)}")
            raise

    def align_datasets(self) -> None:
        """
        Align news sentiment and stock returns data by date.
        """
        try:
            # Find date columns (case-insensitive)
            news_date_col = self._find_date_column(self.news_data, "news data")
            
            # Use the normalized 'date' column that was created in normalize_dates
            if 'date' not in self.stock_data.columns:
                raise ValueError("Stock data must be normalized first. Call normalize_dates() before align_datasets().")
            
            # Debug: Show data before merge
            logger.info(f"Before merge - News data shape: {self.news_data.shape}")
            logger.info(f"Before merge - Stock data shape: {self.stock_data.shape}")
            logger.info(f"News data columns: {self.news_data.columns.tolist()}")
            logger.info(f"Stock data columns: {self.stock_data.columns.tolist()}")
            
            if 'date' in self.news_data.columns:
                logger.info(f"News date range: {self.news_data['date'].min()} to {self.news_data['date'].max()}")
                logger.info(f"News date dtype: {self.news_data['date'].dtype}")
                logger.info(f"Sample news dates: {self.news_data['date'].head(3).tolist()}")
            
            if 'date' in self.stock_data.columns:
                logger.info(f"Stock date range: {self.stock_data['date'].min()} to {self.stock_data['date'].max()}")
                logger.info(f"Stock date dtype: {self.stock_data['date'].dtype}")
                logger.info(f"Sample stock dates: {self.stock_data['date'].head(3).tolist()}")
            
            # Check for overlapping dates
            if 'date' in self.news_data.columns and 'date' in self.stock_data.columns:
                # Handle both datetime and date types
                if pd.api.types.is_datetime64_any_dtype(self.news_data['date']):
                    news_dates = set(self.news_data['date'].dt.date)
                else:
                    news_dates = set(self.news_data['date'])
                
                if pd.api.types.is_datetime64_any_dtype(self.stock_data['date']):
                    stock_dates = set(self.stock_data['date'].dt.date)
                else:
                    stock_dates = set(self.stock_data['date'])
                
                overlapping_dates = news_dates.intersection(stock_dates)
                
                logger.info(f"News dates count: {len(news_dates)}")
                logger.info(f"Stock dates count: {len(stock_dates)}")
                logger.info(f"Overlapping dates count: {len(overlapping_dates)}")
                
                if len(overlapping_dates) == 0:
                    logger.error("No overlapping dates found! This will result in empty alignment.")
                    logger.error("Please check if your news and stock data cover the same time period.")
                else:
                    logger.info(f"Sample overlapping dates: {list(overlapping_dates)[:5]}")
            
            # Debug: Show exact data being merged
            logger.info("=== MERGE DEBUG ===")
            logger.info(f"News data to merge:")
            logger.info(f"  - Shape: {self.news_data.shape}")
            logger.info(f"  - Columns: {self.news_data.columns.tolist()}")
            logger.info(f"  - Sample dates: {self.news_data['date'].head(3).tolist()}")
            logger.info(f"  - Date dtype: {self.news_data['date'].dtype}")
            
            logger.info(f"Stock data to merge:")
            logger.info(f"  - Shape: {self.stock_data.shape}")
            logger.info(f"  - Columns: {self.stock_data.columns.tolist()}")
            logger.info(f"  - Sample dates: {self.stock_data['date'].head(3).tolist()}")
            logger.info(f"  - Date dtype: {self.stock_data['date'].dtype}")
            
            # Check for any NaN values in key columns
            if 'sentiment_score' in self.news_data.columns:
                nan_sentiment = self.news_data['sentiment_score'].isna().sum()
                logger.info(f"  - NaN sentiment scores: {nan_sentiment}")
            
            if 'daily_return' in self.stock_data.columns:
                nan_returns = self.stock_data['daily_return'].isna().sum()
                logger.info(f"  - NaN daily returns: {nan_returns}")
            
            # Merge datasets on date
            logger.info("Attempting merge...")
            self.aligned_data = pd.merge(
                self.news_data,
                self.stock_data[['date', 'daily_return']],
                on='date',
                how='inner'
            )
            logger.info(f"Merge completed. Result shape: {self.aligned_data.shape}")
            
            if not self.aligned_data.empty:
                logger.info(f"Merge successful! Final columns: {self.aligned_data.columns.tolist()}")
                logger.info(f"Sample merged data:")
                logger.info(self.aligned_data.head(3))
            else:
                logger.error("Merge resulted in empty dataset!")
                logger.error("This suggests a problem with the merge operation itself.")
            
            # Additional validation
            if self.aligned_data.empty:
                logger.warning("Alignment resulted in empty dataset. This might indicate:")
                logger.warning("1. No overlapping dates between news and stock data")
                logger.warning("2. Date format mismatches")
                logger.warning("3. Missing or invalid data in key columns")
                
                # Show sample dates from both datasets for debugging
                if not self.news_data.empty and not self.stock_data.empty:
                    logger.info(f"Sample news dates: {self.news_data['date'].head(5).tolist()}")
                    logger.info(f"Sample stock dates: {self.stock_data['date'].head(5).tolist()}")
        except Exception as e:
            logger.error(f"Error aligning datasets: {str(e)}")
            raise

    def calculate_correlation(self) -> Dict[str, float]:
        """
        Calculate correlation between news sentiment and stock returns.
        Automatically prepares data if not already prepared.
        """
        try:
            # Check if aligned data exists and has valid data
            if self.aligned_data is None or self.aligned_data.empty:
                # Try to automatically prepare the data
                logger.info("No aligned data available. Attempting to prepare data automatically...")
                
                # Check if we have the basic data loaded
                if self.news_data is None or self.stock_data is None:
                    raise ValueError("No data loaded. Please call load_data() first with your news and stock datasets.")
                
                # Check if sentiment analysis has been done
                if 'sentiment_score' not in self.news_data.columns:
                    logger.info("Performing sentiment analysis...")
                    self.analyze_sentiment()
                
                # Check if stock returns have been calculated
                if 'daily_return' not in self.stock_data.columns:
                    logger.info("Calculating stock returns...")
                    self.calculate_stock_returns()
                
                # Check if dates have been normalized
                if 'date' not in self.stock_data.columns:
                    logger.info("Normalizing dates...")
                    self.normalize_dates()
                
                # Now align the datasets
                logger.info("Aligning datasets...")
                self.align_datasets()
                
                if self.aligned_data is None or self.aligned_data.empty:
                    raise ValueError("Failed to align datasets. Please check your data quality and date formats.")
                
                logger.info("Data preparation completed successfully!")
            
            # Check for required columns
            required_cols = ['sentiment_score', 'daily_return']
            missing_cols = [col for col in required_cols if col not in self.aligned_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Remove rows with NaN values in key columns
            clean_data = self.aligned_data.dropna(subset=['sentiment_score', 'daily_return'])
            
            if clean_data.empty:
                raise ValueError("No valid data points after removing NaN values. Check your data quality.")
            
            if len(clean_data) < 2:
                raise ValueError("Need at least 2 data points to calculate correlation.")
            
            logger.info(f"Calculating correlation with {len(clean_data)} valid data points")
            
            # Calculate Pearson correlation
            correlation = clean_data['sentiment_score'].corr(clean_data['daily_return'])
            
            # Calculate lagged correlations (t+1, t+2)
            clean_data['next_day_return'] = clean_data['daily_return'].shift(-1)
            clean_data['two_day_return'] = clean_data['daily_return'].shift(-2)
            
            lag_1_corr = clean_data['sentiment_score'].corr(clean_data['next_day_return'])
            lag_2_corr = clean_data['sentiment_score'].corr(clean_data['two_day_return'])
            
            self.correlation_results = {
                'same_day_correlation': correlation,
                'next_day_correlation': lag_1_corr,
                'two_day_correlation': lag_2_corr
            }
            
            logger.info("Correlation analysis completed successfully")
            return self.correlation_results
        except Exception as e:
            logger.error(f"Error calculating correlation: {str(e)}")
            raise

    def plot_correlation(self, save_path: str = None) -> None:
        """
        Plot correlation between news sentiment and stock returns.
        Automatically prepares data if not already prepared.
        """
        try:
            # Check if aligned data exists and has valid data
            if self.aligned_data is None or self.aligned_data.empty:
                # Try to automatically prepare the data
                logger.info("No aligned data available. Attempting to prepare data automatically...")
                
                # Check if we have the basic data loaded
                if self.news_data is None or self.stock_data is None:
                    raise ValueError("No data loaded. Please call load_data() first with your news and stock datasets.")
                
                # Check if sentiment analysis has been done
                if 'sentiment_score' not in self.news_data.columns:
                    logger.info("Performing sentiment analysis...")
                    self.analyze_sentiment()
                
                # Check if stock returns have been calculated
                if 'daily_return' not in self.stock_data.columns:
                    logger.info("Calculating stock returns...")
                    self.calculate_stock_returns()
                
                # Check if dates have been normalized
                if 'date' not in self.stock_data.columns:
                    logger.info("Normalizing dates...")
                    self.normalize_dates()
                
                # Now align the datasets
                logger.info("Aligning datasets...")
                self.align_datasets()
                
                if self.aligned_data is None or self.aligned_data.empty:
                    raise ValueError("Failed to align datasets. Please check your data quality and date formats.")
                
                logger.info("Data preparation completed successfully!")
            
            # Check for required columns
            required_cols = ['sentiment_score', 'daily_return', 'date']
            missing_cols = [col for col in required_cols if col not in self.aligned_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Remove rows with NaN values in key columns
            clean_data = self.aligned_data.dropna(subset=['sentiment_score', 'daily_return'])
            
            if clean_data.empty:
                raise ValueError("No valid data points after removing NaN values. Check your data quality.")
            
            if len(clean_data) < 2:
                raise ValueError("Need at least 2 data points to create plots.")
            
            logger.info(f"Plotting correlation with {len(clean_data)} valid data points")
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Scatter plot of sentiment vs returns
            sns.scatterplot(
                data=clean_data,
                x='sentiment_score',
                y='daily_return',
                ax=ax1
            )
            ax1.set_title('Sentiment Score vs Daily Returns')
            ax1.set_xlabel('Sentiment Score')
            ax1.set_ylabel('Daily Return')
            
            # Add trend line only if we have enough data points
            if len(clean_data) >= 2:
                z = np.polyfit(clean_data['sentiment_score'], clean_data['daily_return'], 1)
                p = np.poly1d(z)
                ax1.plot(clean_data['sentiment_score'], p(clean_data['sentiment_score']), "r--", alpha=0.8)
            
            # Plot 2: Time series of sentiment and returns
            ax2.plot(clean_data['date'], clean_data['sentiment_score'], label='Sentiment Score')
            ax2.plot(clean_data['date'], clean_data['daily_return'], label='Daily Return')
            ax2.set_title('Sentiment Score and Daily Returns Over Time')
            ax2.set_xlabel('Date')
            ax2.legend()
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating correlation plot: {str(e)}")
            raise

    def get_summary_statistics(self) -> Dict[str, float]:
        """
        Calculate summary statistics for the analysis.
        
        Returns:
            Dict[str, float]: Dictionary containing summary statistics
        """
        try:
            # Check if aligned data exists and has valid data
            if self.aligned_data is None or self.aligned_data.empty:
                raise ValueError("No aligned data available. Please run align_datasets() first.")
            
            # Remove rows with NaN values in key columns
            clean_data = self.aligned_data.dropna(subset=['sentiment_score', 'daily_return'])
            
            if clean_data.empty:
                raise ValueError("No valid data points after removing NaN values. Check your data quality.")
            
            stats = {
                'avg_sentiment': clean_data['sentiment_score'].mean(),
                'std_sentiment': clean_data['sentiment_score'].std(),
                'avg_return': clean_data['daily_return'].mean(),
                'std_return': clean_data['daily_return'].std(),
                'total_days': len(clean_data),
                'positive_sentiment_days': (clean_data['sentiment_score'] > 0).sum(),
                'negative_sentiment_days': (clean_data['sentiment_score'] < 0).sum()
            }
            return stats
        except Exception as e:
            logger.error(f"Error calculating summary statistics: {str(e)}")
            raise

    def diagnose_data_issues(self) -> Dict[str, str]:
        """
        Diagnose common data issues and provide suggestions for fixing them.
        
        Returns:
            Dict[str, str]: Dictionary containing issue descriptions and suggestions
        """
        issues = {}
        
        try:
            if self.news_data is None or self.news_data.empty:
                issues['news_data'] = "News data is missing or empty. Please load news data first."
                return issues
                
            if self.stock_data is None or self.stock_data.empty:
                issues['stock_data'] = "Stock data is missing or empty. Please load stock data first."
                return issues
            
            # Check for date column issues
            if 'date' not in self.news_data.columns:
                issues['news_date_column'] = "News data missing 'date' column. Check column names."
            if 'date' not in self.stock_data.columns:
                issues['stock_date_column'] = "Stock data missing 'date' column. Check column names."
            
            # Check for data type issues
            if 'date' in self.news_data.columns:
                if not pd.api.types.is_datetime64_any_dtype(self.news_data['date']):
                    issues['news_date_type'] = "News date column is not datetime type. Run normalize_dates()."
            
            if 'date' in self.stock_data.columns:
                if not pd.api.types.is_datetime64_any_dtype(self.stock_data['date']):
                    issues['stock_date_type'] = "Stock date column is not datetime type. Run normalize_dates()."
            
            # Check for sentiment analysis issues
            if 'sentiment_score' not in self.news_data.columns:
                issues['sentiment_missing'] = "Sentiment scores not calculated. Run analyze_sentiment()."
            
            # Check for stock return issues
            if 'daily_return' not in self.stock_data.columns:
                issues['returns_missing'] = "Daily returns not calculated. Run calculate_stock_returns()."
            
            # Check alignment issues
            if self.aligned_data is not None:
                if self.aligned_data.empty:
                    issues['alignment_empty'] = "Data alignment resulted in empty dataset. Check date ranges and formats."
                else:
                    # Check for NaN values
                    nan_sentiment = self.aligned_data['sentiment_score'].isna().sum()
                    nan_returns = self.aligned_data['daily_return'].isna().sum()
                    
                    if nan_sentiment > 0:
                        issues['nan_sentiment'] = f"Found {nan_sentiment} NaN values in sentiment scores."
                    if nan_returns > 0:
                        issues['nan_returns'] = f"Found {nan_returns} NaN values in daily returns."
            
            if not issues:
                issues['status'] = "No issues detected. Data appears to be ready for analysis."
            
            return issues
            
        except Exception as e:
            issues['error'] = f"Error during diagnosis: {str(e)}"
            return issues

    def get_workflow_guide(self) -> str:
        """
        Get a step-by-step guide for the proper workflow to use this class.
        
        Returns:
            str: Workflow guide
        """
        guide = """
        === WORKFLOW GUIDE ===
        
        To use this class properly, follow these steps in order:
        
        1. LOAD DATA:
           analyzer = NewsStockCorrelation()
           analyzer.load_data(news_df, stock_df)
           
        2. NORMALIZE DATES:
           analyzer.normalize_dates()
           
        3. ANALYZE SENTIMENT:
           analyzer.analyze_sentiment('headline')  # or your headline column name
           
        4. CALCULATE STOCK RETURNS:
           analyzer.calculate_stock_returns('close')  # or your price column name
           
        5. ALIGN DATASETS:
           analyzer.align_datasets()
           
        6. DIAGNOSE ISSUES (optional but recommended):
           issues = analyzer.diagnose_data_issues()
           print(issues)
           
        7. ANALYZE CORRELATION:
           correlation = analyzer.calculate_correlation()
           
        8. CREATE PLOTS:
           analyzer.plot_correlation()
           
        9. GET SUMMARY STATISTICS:
           stats = analyzer.get_summary_statistics()
           
        === TROUBLESHOOTING ===
        
        If you encounter errors:
        - Use analyzer.debug_data_info() to inspect your data
        - Use analyzer.diagnose_data_issues() to identify problems
        - Ensure your data has overlapping date ranges
        - Check that column names match expected formats
        
        === DATA REQUIREMENTS ===
        
        News data must contain:
        - A date column (will be converted to datetime)
        - A text column for sentiment analysis (e.g., 'headline')
        
        Stock data must contain:
        - A date column (will be converted to datetime)
        - A price column (e.g., 'close', 'open', 'high', 'low')
        
        ========================
        """
        return guide

    def run_full_analysis(self, news_df: pd.DataFrame, stock_df: pd.DataFrame, 
                          headline_column: str = 'headline', price_column: str = 'close') -> Dict[str, any]:
        """
        Run the complete analysis workflow automatically.
        This is the easiest way to use the class - just provide your data and get results!
        
        Args:
            news_df (pd.DataFrame): News dataset with date and headlines
            stock_df (pd.DataFrame): Stock price dataset with date and prices
            headline_column (str): Name of the column containing headlines
            price_column (str): Name of the column containing stock prices
            
        Returns:
            Dict[str, any]: Complete analysis results including correlations and summary statistics
        """
        try:
            logger.info("Starting full analysis workflow...")
            
            # Step 1: Load data
            self.load_data(news_df, stock_df)
            logger.info("✓ Data loaded successfully")
            
            # Step 2: Normalize dates
            self.normalize_dates()
            logger.info("✓ Dates normalized successfully")
            
            # Step 3: Analyze sentiment
            self.analyze_sentiment(headline_column)
            logger.info("✓ Sentiment analysis completed")
            
            # Step 4: Calculate stock returns
            self.calculate_stock_returns(price_column)
            logger.info("✓ Stock returns calculated")
            
            # Step 5: Align datasets
            self.align_datasets()
            logger.info("✓ Datasets aligned")
            
            # Step 6: Calculate correlations
            correlations = self.calculate_correlation()
            logger.info("✓ Correlations calculated")
            
            # Step 7: Get summary statistics
            summary_stats = self.get_summary_statistics()
            logger.info("✓ Summary statistics generated")
            
            # Step 8: Create plots
            self.plot_correlation()
            logger.info("✓ Plots created")
            
            # Return comprehensive results
            results = {
                'correlations': correlations,
                'summary_statistics': summary_stats,
                'data_shape': self.aligned_data.shape if self.aligned_data is not None else None,
                'date_range': {
                    'start': self.aligned_data['date'].min() if self.aligned_data is not None else None,
                    'end': self.aligned_data['date'].max() if self.aligned_data is not None else None
                } if self.aligned_data is not None else None
            }
            
            logger.info("🎉 Full analysis completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Error in full analysis: {str(e)}")
            raise
