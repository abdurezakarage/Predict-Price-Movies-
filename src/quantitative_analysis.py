

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import talib
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class QuantitativeAnalyzer:
  
    
    def __init__(self, symbol: str, period: str = "1y", interval: str = "1d"):
   
        self.symbol = symbol.upper()
        self.period = period
        self.interval = interval
        self.stock_data = None
        self.technical_indicators = {}
        self.financial_metrics = {}
        
        # Create directories for outputs
        self.viz_dir = "../visualizations"
        self.data_dir = "../data"
        self.models_dir = "../models"
        
        for directory in [self.viz_dir, self.data_dir, self.models_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
    
    def load_stock_data(self) -> bool:
      
        try:
            logger.info(f"Loading stock data for {self.symbol}...")
            
            # Download stock data
            ticker = yf.Ticker(self.symbol)
            self.stock_data = ticker.history(period=self.period, interval=self.interval)
            
            if self.stock_data.empty:
                logger.error(f"No data found for symbol {self.symbol}")
                return False
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in self.stock_data.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Clean data
            self.stock_data = self.stock_data.dropna()
            
            # Add date index if not present
            if not isinstance(self.stock_data.index, pd.DatetimeIndex):
                self.stock_data.index = pd.to_datetime(self.stock_data.index)
            
            logger.info(f"Successfully loaded {len(self.stock_data)} data points for {self.symbol}")
            logger.info(f"Date range: {self.stock_data.index.min()} to {self.stock_data.index.max()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading stock data: {e}")
            return False
    
    def calculate_technical_indicators(self) -> Dict:
     
        if self.stock_data is None:
            logger.error("Stock data not loaded. Please load data first.")
            return {}
        
        logger.info("Calculating technical indicators...")
        
        try:
            # Extract OHLCV data and ensure they are float64 for TA-Lib
            logger.debug(f"Original data types: Open={self.stock_data['Open'].dtype}, Close={self.stock_data['Close'].dtype}")
            
            open_prices = self.stock_data['Open'].astype(np.float64).values
            high_prices = self.stock_data['High'].astype(np.float64).values
            low_prices = self.stock_data['Low'].astype(np.float64).values
            close_prices = self.stock_data['Close'].astype(np.float64).values
            volume = self.stock_data['Volume'].astype(np.float64).values
            
            logger.debug(f"Converted data types: open_prices={open_prices.dtype}, close_prices={close_prices.dtype}")
            logger.debug(f"Array shapes: open={open_prices.shape}, close={close_prices.shape}")
            
            # Initialize indicators dictionary
            indicators = {}
            
            # 1. Moving Averages
            indicators['SMA_20'] = talib.SMA(close_prices, timeperiod=20)
            indicators['SMA_50'] = talib.SMA(close_prices, timeperiod=50)
            indicators['SMA_200'] = talib.SMA(close_prices, timeperiod=200)
            indicators['EMA_12'] = talib.EMA(close_prices, timeperiod=12)
            indicators['EMA_26'] = talib.EMA(close_prices, timeperiod=26)
            
            # 2. Momentum Indicators
            indicators['RSI'] = talib.RSI(close_prices, timeperiod=14)
            indicators['STOCH_K'], indicators['STOCH_D'] = talib.STOCH(high_prices, low_prices, close_prices)
            indicators['WILLR'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
            indicators['CCI'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
            
            # 3. Trend Indicators
            indicators['MACD'], indicators['MACD_SIGNAL'], indicators['MACD_HIST'] = talib.MACD(close_prices)
            indicators['ADX'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
            indicators['AROON_UP'], indicators['AROON_DOWN'] = talib.AROON(high_prices, low_prices, timeperiod=14)
            
            # 4. Volatility Indicators
            indicators['BBANDS_UPPER'], indicators['BBANDS_MIDDLE'], indicators['BBANDS_LOWER'] = talib.BBANDS(close_prices)
            indicators['ATR'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            indicators['NATR'] = talib.NATR(high_prices, low_prices, close_prices, timeperiod=14)
            
            # 5. Volume Indicators
            indicators['OBV'] = talib.OBV(close_prices, volume)
            indicators['AD'] = talib.AD(high_prices, low_prices, close_prices, volume)
            indicators['ADOSC'] = talib.ADOSC(high_prices, low_prices, close_prices, volume)
            
            # 6. Price Pattern Recognition
            indicators['DOJI'] = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
            indicators['HAMMER'] = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
            indicators['ENGULFING'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
            
            # 7. Additional Custom Indicators
            # Price Rate of Change
            indicators['ROC'] = talib.ROC(close_prices, timeperiod=10)
            
            # Money Flow Index
            indicators['MFI'] = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=14)
            
            # Ultimate Oscillator
            indicators['ULTOSC'] = talib.ULTOSC(high_prices, low_prices, close_prices)
            
            # Store indicators
            self.technical_indicators = indicators
            
            logger.info(f"Successfully calculated {len(indicators)} technical indicators")
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    def calculate_financial_metrics(self) -> Dict:
      
        if self.stock_data is None:
            logger.error("Stock data not loaded. Please load data first.")
            return {}
        
        logger.info("Calculating financial metrics...")
        
        try:
            metrics = {}
            
            # Price-based metrics
            close_prices = self.stock_data['Close']
            high_prices = self.stock_data['High']
            low_prices = self.stock_data['Low']
            volume = self.stock_data['Volume']
            
            # 1. Price Statistics
            metrics['price_mean'] = close_prices.mean()
            metrics['price_std'] = close_prices.std()
            metrics['price_min'] = close_prices.min()
            metrics['price_max'] = close_prices.max()
            metrics['price_range'] = close_prices.max() - close_prices.min()
            
            # 2. Returns and Volatility
            returns = close_prices.pct_change().dropna()
            metrics['daily_return_mean'] = returns.mean()
            metrics['daily_return_std'] = returns.std()
            metrics['annualized_volatility'] = returns.std() * np.sqrt(252)  # Assuming 252 trading days
            
            # 3. Risk Metrics
            metrics['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
            metrics['max_drawdown'] = self._calculate_max_drawdown(close_prices)
            metrics['var_95'] = np.percentile(returns, 5)  # 95% VaR
            
            # 4. Volume Analysis
            metrics['volume_mean'] = volume.mean()
            metrics['volume_std'] = volume.std()
            metrics['price_volume_correlation'] = close_prices.corr(volume)
            
            # 5. Technical Ratios
            metrics['high_low_ratio'] = (high_prices / low_prices).mean()
            metrics['close_open_ratio'] = (close_prices / self.stock_data['Open']).mean()
            
            # 6. Trend Strength
            sma_20 = talib.SMA(close_prices.values, timeperiod=20)
            sma_50 = talib.SMA(close_prices.values, timeperiod=50)
            metrics['trend_strength'] = np.mean((sma_20 - sma_50) / sma_50) if len(sma_50) > 0 else 0
            
            # 7. Momentum Metrics
            metrics['momentum_5d'] = (close_prices.iloc[-1] / close_prices.iloc[-6] - 1) if len(close_prices) >= 6 else 0
            metrics['momentum_20d'] = (close_prices.iloc[-1] / close_prices.iloc[-21] - 1) if len(close_prices) >= 21 else 0
            
            # Store metrics
            self.financial_metrics = metrics
            
            logger.info(f"Successfully calculated {len(metrics)} financial metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating financial metrics: {e}")
            return {}
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown from peak."""
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def create_comprehensive_visualizations(self) -> None:
       
        if self.stock_data is None or not self.technical_indicators:
            logger.error("Data not available for visualization. Please load data and calculate indicators first.")
            return
        
        logger.info("Creating comprehensive visualizations...")
        
        try:
            # Create subplots for different analysis sections
            fig = make_subplots(
                rows=4, cols=2,
                subplot_titles=(
                    'Price Chart with Moving Averages',
                    'Volume Analysis',
                    'RSI and Stochastic Oscillators',
                    'MACD Analysis',
                    'Bollinger Bands',
                    'Price Patterns',
                    'Financial Metrics Summary',
                    'Risk-Return Analysis'
                ),
                specs=[
                    [{"secondary_y": True}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}]
                ],
                vertical_spacing=0.08,
                horizontal_spacing=0.1
            )
            
            # 1. Price Chart with Moving Averages
            fig.add_trace(
                go.Scatter(x=self.stock_data.index, y=self.stock_data['Close'], 
                          name='Close Price', line=dict(color='blue')),
                row=1, col=1, secondary_y=False
            )
            
            # Add moving averages
            if 'SMA_20' in self.technical_indicators:
                fig.add_trace(
                    go.Scatter(x=self.stock_data.index, y=self.technical_indicators['SMA_20'],
                              name='SMA 20', line=dict(color='orange', dash='dash')),
                    row=1, col=1, secondary_y=False
                )
            
            if 'SMA_50' in self.technical_indicators:
                fig.add_trace(
                    go.Scatter(x=self.stock_data.index, y=self.technical_indicators['SMA_50'],
                              name='SMA 50', line=dict(color='red', dash='dash')),
                    row=1, col=1, secondary_y=False
                )
            
            # Add volume as secondary y-axis
            fig.add_trace(
                go.Bar(x=self.stock_data.index, y=self.stock_data['Volume'],
                      name='Volume', opacity=0.3, yaxis='y2'),
                row=1, col=1, secondary_y=True
            )
            
            # 2. Volume Analysis
            fig.add_trace(
                go.Bar(x=self.stock_data.index, y=self.stock_data['Volume'],
                      name='Volume', marker_color='lightblue'),
                row=1, col=2
            )
            
            # 3. RSI and Stochastic
            if 'RSI' in self.technical_indicators:
                fig.add_trace(
                    go.Scatter(x=self.stock_data.index, y=self.technical_indicators['RSI'],
                              name='RSI', line=dict(color='purple')),
                    row=2, col=1
                )
                # Add RSI overbought/oversold lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            if 'STOCH_K' in self.technical_indicators:
                fig.add_trace(
                    go.Scatter(x=self.stock_data.index, y=self.technical_indicators['STOCH_K'],
                              name='Stoch %K', line=dict(color='blue')),
                    row=2, col=2
                )
                if 'STOCH_D' in self.technical_indicators:
                    fig.add_trace(
                        go.Scatter(x=self.stock_data.index, y=self.technical_indicators['STOCH_D'],
                                  name='Stoch %D', line=dict(color='red')),
                        row=2, col=2
                    )
            
            # 4. MACD
            if 'MACD' in self.technical_indicators:
                fig.add_trace(
                    go.Scatter(x=self.stock_data.index, y=self.technical_indicators['MACD'],
                              name='MACD', line=dict(color='blue')),
                    row=3, col=1
                )
                if 'MACD_SIGNAL' in self.technical_indicators:
                    fig.add_trace(
                        go.Scatter(x=self.stock_data.index, y=self.technical_indicators['MACD_SIGNAL'],
                                  name='MACD Signal', line=dict(color='red')),
                        row=3, col=1
                    )
                if 'MACD_HIST' in self.technical_indicators:
                    fig.add_trace(
                        go.Bar(x=self.stock_data.index, y=self.technical_indicators['MACD_HIST'],
                              name='MACD Histogram', marker_color='gray'),
                        row=3, col=1
                    )
            
            # 5. Bollinger Bands
            if 'BBANDS_UPPER' in self.technical_indicators:
                fig.add_trace(
                    go.Scatter(x=self.stock_data.index, y=self.technical_indicators['BBANDS_UPPER'],
                              name='BB Upper', line=dict(color='gray', dash='dash')),
                    row=3, col=2
                )
                fig.add_trace(
                    go.Scatter(x=self.stock_data.index, y=self.technical_indicators['BBANDS_LOWER'],
                              name='BB Lower', line=dict(color='gray', dash='dash')),
                    row=3, col=2
                )
                fig.add_trace(
                    go.Scatter(x=self.stock_data.index, y=self.stock_data['Close'],
                              name='Price', line=dict(color='blue')),
                    row=3, col=2
                )
            
            # 6. Price Patterns (Candlestick)
            fig.add_trace(
                go.Candlestick(
                    x=self.stock_data.index,
                    open=self.stock_data['Open'],
                    high=self.stock_data['High'],
                    low=self.stock_data['Low'],
                    close=self.stock_data['Close'],
                    name='OHLC'
                ),
                row=4, col=1
            )
            
            # 7. Financial Metrics Summary
            if self.financial_metrics:
                metrics_names = list(self.financial_metrics.keys())[:8]  # Show first 8 metrics
                metrics_values = [self.financial_metrics[name] for name in metrics_names]
                
                fig.add_trace(
                    go.Bar(x=metrics_names, y=metrics_values,
                          name='Financial Metrics', marker_color='lightgreen'),
                    row=4, col=2
                )
            
            # Update layout
            fig.update_layout(
                title=f'Comprehensive Technical Analysis for {self.symbol}',
                height=1200,
                showlegend=True,
                template='plotly_white'
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Date", row=4, col=1)
            fig.update_xaxes(title_text="Date", row=4, col=2)
            
            # Save the comprehensive chart
            chart_path = os.path.join(self.viz_dir, f'{self.symbol}_comprehensive_analysis.html')
            fig.write_html(chart_path)
            
            # Also save as PNG for static viewing
            png_path = os.path.join(self.viz_dir, f'{self.symbol}_comprehensive_analysis.png')
            fig.write_image(png_path, width=1600, height=1200)
            
            logger.info(f"Comprehensive visualizations saved: {chart_path}")
            logger.info(f"PNG version saved: {png_path}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def generate_analysis_report(self) -> str:
      
        if not self.technical_indicators or not self.financial_metrics:
            logger.error("Analysis data not available. Please run analysis first.")
            return ""
        
        logger.info("Generating analysis report...")
        
        try:
            report_path = os.path.join(self.data_dir, f'{self.symbol}_analysis_report.md')
            
            with open(report_path, 'w') as f:
                f.write(f"# Technical Analysis Report for {self.symbol}\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## Executive Summary\n")
                f.write(f"Analysis period: {self.period}\n")
                f.write(f"Data interval: {self.interval}\n")
                f.write(f"Total data points: {len(self.stock_data)}\n")
                f.write(f"Date range: {self.stock_data.index.min()} to {self.stock_data.index.max()}\n\n")
                
                f.write("## Key Financial Metrics\n")
                for metric, value in self.financial_metrics.items():
                    if isinstance(value, float):
                        f.write(f"- **{metric}**: {value:.4f}\n")
                    else:
                        f.write(f"- **{metric}**: {value}\n")
                f.write("\n")
                
                f.write("## Technical Indicators Summary\n")
                f.write(f"Total indicators calculated: {len(self.technical_indicators)}\n\n")
                
                # Group indicators by category
                categories = {
                    'Moving Averages': ['SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26'],
                    'Momentum': ['RSI', 'STOCH_K', 'STOCH_D', 'WILLR', 'CCI'],
                    'Trend': ['MACD', 'MACD_SIGNAL', 'MACD_HIST', 'ADX', 'AROON_UP', 'AROON_DOWN'],
                    'Volatility': ['BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER', 'ATR', 'NATR'],
                    'Volume': ['OBV', 'AD', 'ADOSC'],
                    'Patterns': ['DOJI', 'HAMMER', 'ENGULFING']
                }
                
                for category, indicators in categories.items():
                    f.write(f"### {category}\n")
                    for indicator in indicators:
                        if indicator in self.technical_indicators:
                            f.write(f"- {indicator}: Available\n")
                    f.write("\n")
                
                f.write("## Recommendations\n")
                f.write("Based on the technical analysis:\n")
                
                # Generate basic recommendations
                close_price = self.stock_data['Close'].iloc[-1]
                sma_20 = self.technical_indicators.get('SMA_20', [0])[-1] if 'SMA_20' in self.technical_indicators else 0
                sma_50 = self.technical_indicators.get('SMA_50', [0])[-1] if 'SMA_50' in self.technical_indicators else 0
                rsi = self.technical_indicators.get('RSI', [50])[-1] if 'RSI' in self.technical_indicators else 50
                
                if close_price > sma_20 > sma_50:
                    f.write("- **Trend**: Bullish (price above moving averages)\n")
                elif close_price < sma_20 < sma_50:
                    f.write("- **Trend**: Bearish (price below moving averages)\n")
                else:
                    f.write("- **Trend**: Mixed signals\n")
                
                if rsi > 70:
                    f.write("- **RSI**: Overbought conditions (>70)\n")
                elif rsi < 30:
                    f.write("- **RSI**: Oversold conditions (<30)\n")
                else:
                    f.write("- **RSI**: Neutral conditions\n")
                
                f.write("\n## Data Quality Assessment\n")
                f.write(f"- Missing values: {self.stock_data.isnull().sum().sum()}\n")
                f.write(f"- Data completeness: {((len(self.stock_data) - self.stock_data.isnull().sum().sum()) / (len(self.stock_data) * len(self.stock_data.columns)) * 100):.2f}%\n")
                
            logger.info(f"Analysis report generated: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return ""
    
    def save_processed_data(self) -> None:
        """Save all processed data and indicators for further analysis."""
        if self.stock_data is None:
            logger.error("No data to save. Please load data first.")
            return
        
        logger.info("Saving processed data...")
        
        try:
            # Save stock data with indicators
            data_with_indicators = self.stock_data.copy()
            
            # Add technical indicators
            for indicator_name, indicator_values in self.technical_indicators.items():
                if len(indicator_values) == len(data_with_indicators):
                    data_with_indicators[indicator_name] = indicator_values
            
            # Save to CSV
            csv_path = os.path.join(self.data_dir, f'{self.symbol}_data_with_indicators.csv')
            data_with_indicators.to_csv(csv_path)
            logger.info(f"Data with indicators saved: {csv_path}")
            
            # Save financial metrics
            if self.financial_metrics:
                metrics_df = pd.DataFrame(list(self.financial_metrics.items()), 
                                        columns=['Metric', 'Value'])
                metrics_path = os.path.join(self.data_dir, f'{self.symbol}_financial_metrics.csv')
                metrics_df.to_csv(metrics_path, index=False)
                logger.info(f"Financial metrics saved: {metrics_path}")
            
            # Save metadata
            metadata = {
                'symbol': self.symbol,
                'period': self.period,
                'interval': self.interval,
                'data_points': len(self.stock_data),
                'date_range_start': str(self.stock_data.index.min()),
                'date_range_end': str(self.stock_data.index.max()),
                'indicators_calculated': len(self.technical_indicators),
                'metrics_calculated': len(self.financial_metrics),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            metadata_df = pd.DataFrame(list(metadata.items()), columns=['Field', 'Value'])
            metadata_path = os.path.join(self.data_dir, f'{self.symbol}_metadata.csv')
            metadata_df.to_csv(metadata_path, index=False)
            logger.info(f"Metadata saved: {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
    
    def run_complete_analysis(self) -> bool:
        """
        Run the complete quantitative analysis pipeline.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Starting complete quantitative analysis for {self.symbol}...")
        
        try:
            # 1. Load data
            if not self.load_stock_data():
                return False
            
            # 2. Calculate technical indicators
            if not self.calculate_technical_indicators():
                logger.warning("Some technical indicators could not be calculated")
            
            # 3. Calculate financial metrics
            if not self.calculate_financial_metrics():
                logger.warning("Some financial metrics could not be calculated")
            
            # 4. Create visualizations
            self.create_comprehensive_visualizations()
            
            # 5. Generate report
            report_path = self.generate_analysis_report()
            
            # 6. Save processed data
            self.save_processed_data()
            
            logger.info("Complete quantitative analysis finished successfully!")
            logger.info(f"Generated files:")
            logger.info(f"- Visualizations: {self.viz_dir}/")
            logger.info(f"- Data files: {self.data_dir}/")
            logger.info(f"- Analysis report: {report_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {e}")
            return False

def main():
    """Main function to demonstrate the quantitative analysis."""
    # Example usage
    symbols = ['AAPL', 'MSFT', 'GOOGL']  # Example stocks
    
    for symbol in symbols:
        logger.info(f"\n{'='*50}")
        logger.info(f"Analyzing {symbol}")
        logger.info(f"{'='*50}")
        
        analyzer = QuantitativeAnalyzer(symbol, period="1y", interval="1d")
        success = analyzer.run_complete_analysis()
        
        if success:
            logger.info(f"Analysis completed successfully for {symbol}")
        else:
            logger.error(f"Analysis failed for {symbol}")

if __name__ == "__main__":
    main()
