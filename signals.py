import requests
import pandas as pd
import os, json, time, logging
from datetime import datetime, timedelta, timezone
import feedparser
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Twelve Data API configuration
TWELVE_DATA_BASE_URL = "https://api.twelvedata.com"
API_TIMEOUT = 15
RATE_LIMIT_DELAY = 1.0  # Delay between API calls in seconds
MAX_RETRIES = 3

class APIError(Exception):
    """Custom exception for API errors"""
    pass

class TwelveDataClient:
    """Handles all Twelve Data API interactions with robust error handling"""
    
    def __init__(self):
        self.api_key = self._get_api_key()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TradingBot/1.0',
            'Accept': 'application/json'
        })
    
    def _get_api_key(self) -> str:
        """Get Twelve Data API key from config or environment with validation"""
        # Try config.json first
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    api_key = config.get('TWELVE_DATA_API_KEY')
                    if api_key and api_key != 'YOUR_TWELVE_DATA_API_KEY_HERE':
                        return api_key
        except Exception as e:
            logger.warning(f"Could not read config file: {e}")
        
        # Fallback to environment variable
        api_key = os.getenv('TWELVE_DATA_API_KEY')
        if not api_key:
            raise ValueError('TWELVE_DATA_API_KEY not found in config.json or environment variables')
        
        return api_key
    
    def make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make request to Twelve Data API with comprehensive error handling and retry logic"""
        if params is None:
            params = {}
        
        params['apikey'] = self.api_key
        url = f"{TWELVE_DATA_BASE_URL}/{endpoint}"
        
        last_exception = None
        
        for attempt in range(MAX_RETRIES):
            try:
                logger.debug(f"Twelve Data API request attempt {attempt + 1}: {endpoint}")
                
                response = self.session.get(url, params=params, timeout=API_TIMEOUT)
                response.raise_for_status()
                
                data = response.json()
                
                # Check for Twelve Data API specific errors
                if isinstance(data, dict):
                    if data.get('status') == 'error':
                        error_msg = data.get('message', 'Unknown error')
                        raise APIError(f"Twelve Data API Error: {error_msg}")
                    
                    if data.get('code') == 429:
                        # Rate limit - wait and retry
                        wait_time = 5 * (attempt + 1)
                        logger.warning(f"Twelve Data rate limit hit, waiting {wait_time}s before retry")
                        time.sleep(wait_time)
                        continue
                
                return data
                
            except requests.exceptions.Timeout:
                last_exception = APIError("Request timeout - Twelve Data API server may be slow")
                logger.warning(f"Request timeout on attempt {attempt + 1}")
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait_time = 5 * (attempt + 1)
                    logger.warning(f"Twelve Data HTTP 429 rate limit, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                last_exception = APIError(f"HTTP error {e.response.status_code}: {e}")
                
            except requests.exceptions.RequestException as e:
                last_exception = APIError(f"Network error: {str(e)}")
                
            except json.JSONDecodeError:
                last_exception = APIError("Invalid JSON response from Twelve Data API")
                
            except Exception as e:
                last_exception = APIError(f"Unexpected error: {str(e)}")
            
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Request failed, retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        # If all retries failed
        logger.error(f"All {MAX_RETRIES} attempts failed for Twelve Data endpoint: {endpoint}")
        raise last_exception

# Global Twelve Data client instance
twelve_data_client = TwelveDataClient()

def get_api_key():
    """Get Twelve Data API key from config or environment"""
    return twelve_data_client.api_key

def make_api_request(endpoint: str, params: Dict = None) -> Dict:
    """Make a request to Twelve Data API with error handling"""
    return twelve_data_client.make_request(endpoint, params)

def fetch_price(symbol: str) -> float:
    """Fetch current price using Twelve Data real-time price endpoint"""
    try:
        data = make_api_request('price', {'symbol': symbol})
        
        if 'price' not in data:
            raise ValueError(f'No price data for {symbol}')
        
        price = float(data['price'])
        logger.debug(f"Successfully fetched price for {symbol}: ${price:.2f}")
        return price
        
    except Exception as e:
        logger.error(f"Failed to fetch price for {symbol}: {e}")
        raise ValueError(f'No data for {symbol}')

def fetch_daily_data(symbol: str, days: int = 100) -> pd.DataFrame:
    """Fetch daily historical data from Twelve Data"""
    params = {
        'symbol': symbol,
        'interval': '1day',
        'outputsize': min(days, 5000),  # Max 5000 for free tier
        'format': 'JSON'
    }
    
    try:
        data = make_api_request('time_series', params)
        
        # Parse time series data
        if 'values' not in data or not data['values']:
            raise ValueError(f'No daily data for {symbol}')
        
        # Convert to DataFrame
        df = pd.DataFrame(data['values'])
        
        if df.empty:
            raise ValueError(f'Empty dataset returned for {symbol}')
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.sort_index()
        
        # Rename and convert columns
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Convert to numeric with proper error handling
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        
        # Remove rows with invalid data
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        if df.empty:
            raise ValueError(f'No valid data after processing for {symbol}')
        
        logger.debug(f"Successfully fetched {len(df)} days of data for {symbol}")
        return df
    
    except Exception as e:
        logger.error(f"Failed to fetch daily data for {symbol}: {e}")
        raise ValueError(f'Insufficient data for {symbol}')

def fetch_sma_data(symbol: str, window: int, days: int = 100) -> pd.Series:
    """Fetch SMA data directly from Twelve Data (more efficient)"""
    params = {
        'symbol': symbol,
        'interval': '1day',
        'time_period': window,
        'series_type': 'close',
        'outputsize': min(days, 5000),
        'format': 'JSON'
    }
    
    try:
        data = make_api_request('sma', params)
        
        if 'values' not in data or not data['values']:
            raise ValueError(f'No SMA data for {symbol}')
        
        # Convert to Series
        sma_data = []
        dates = []
        for item in data['values']:
            try:
                dates.append(pd.to_datetime(item['datetime']))
                sma_data.append(float(item['sma']))
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping invalid SMA data point: {e}")
                continue
        
        if not sma_data:
            raise ValueError(f'No valid SMA data for {symbol}')
        
        sma_series = pd.Series(sma_data, index=dates)
        return sma_series.sort_index()
    
    except Exception as e:
        logger.warning(f"Failed to fetch SMA data for {symbol}: {e}")
        return None

def fetch_company_overview(symbol: str) -> Dict:
    """Fetch company profile from Twelve Data"""
    try:
        data = make_api_request('profile', {'symbol': symbol})
        
        if not data or 'name' not in data:
            logger.warning(f"Limited profile data for {symbol}")
            return {
                'name': symbol,
                'symbol': symbol,
                'sector': 'N/A',
                'industry': 'N/A',
                'market_capitalization': 'N/A',
                'pe_ratio': 'N/A',
                'employees': 'N/A'
            }
        
        # Format market cap
        market_cap = data.get('market_capitalization', 'N/A')
        market_cap_str = 'N/A'
        if market_cap != 'N/A' and market_cap is not None:
            try:
                mc_value = float(market_cap)
                if mc_value >= 1e12:
                    market_cap_str = f"${mc_value/1e12:.2f}T"
                elif mc_value >= 1e9:
                    market_cap_str = f"${mc_value/1e9:.2f}B"
                elif mc_value >= 1e6:
                    market_cap_str = f"${mc_value/1e6:.2f}M"
                else:
                    market_cap_str = f"${mc_value:,.0f}"
            except (ValueError, TypeError):
                market_cap_str = str(market_cap)
        
        return {
            'name': data.get('name', symbol),
            'symbol': symbol,
            'sector': data.get('sector', 'N/A'),
            'industry': data.get('industry', 'N/A'),
            'market_capitalization': market_cap_str,
            'pe_ratio': data.get('pe_ratio', 'N/A'),
            'employees': data.get('employees', 'N/A')
        }
    
    except Exception as e:
        logger.warning(f"Failed to fetch profile for {symbol}: {e}")
        return {
            'name': symbol,
            'symbol': symbol,
            'sector': 'N/A',
            'industry': 'N/A',
            'market_capitalization': 'N/A',
            'pe_ratio': 'N/A',
            'employees': 'N/A'
        }

def fetch_news(symbol: str) -> List[Dict]:
    """Fetch news for a given symbol with multiple fallback sources"""
    news_items = []
    
    # Method 1: Try Twelve Data news endpoint
    try:
        logging.info(f"Trying Twelve Data news for {symbol}")
        data = make_api_request('news', {'symbol': symbol, 'limit': 5})
        
        if 'feed' in data and data['feed']:
            for item in data['feed'][:5]:
                title = item.get('title', '')
                url = item.get('url', '')
                published = item.get('published_utc', '')
                
                if title and url:
                    # Convert time format
                    pub_time = int(time.time())
                    try:
                        if published:
                            dt = datetime.fromisoformat(published.replace('Z', '+00:00'))
                            pub_time = int(dt.timestamp())
                    except ValueError:
                        pass
                    
                    news_items.append({
                        'title': title.strip(),
                        'link': url.strip(),
                        'published': pub_time,
                        'source': 'twelve_data'
                    })
            
            if news_items:
                logging.info(f"Found {len(news_items)} news items from Twelve Data for {symbol}")
                return news_items
                
    except Exception as e:
        logging.warning(f'Twelve Data news failed for {symbol}: {e}')
    
    # Method 2: Try Yahoo RSS feed as fallback
    try:
        logging.info(f"Trying Yahoo RSS feed for {symbol}")
        rss_url = f"https://finance.yahoo.com/rss/headline?s={symbol}"
        
        feed = feedparser.parse(rss_url)
        
        if hasattr(feed, 'entries') and feed.entries:
            for entry in feed.entries[:5]:
                title = entry.get('title', '')
                link = entry.get('link', '')
                
                if title and link:
                    # Parse publication date
                    pub_time = int(time.time())
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        try:
                            pub_time = int(time.mktime(entry.published_parsed))
                        except (TypeError, OverflowError):
                            pass
                    
                    news_items.append({
                        'title': title.strip(),
                        'link': link.strip(),
                        'published': pub_time,
                        'source': 'yahoo_rss'
                    })
        
        if news_items:
            logging.info(f"Found {len(news_items)} news items from Yahoo RSS for {symbol}")
            return news_items
            
    except Exception as e:
        logging.warning(f'Yahoo RSS feed failed for {symbol}: {e}')
    
    # Method 3: Fallback news items
    try:
        logging.info(f"Using fallback news for {symbol}")
        
        # Get company name from profile
        try:
            profile = fetch_company_overview(symbol)
            company_name = profile.get('name', symbol)
        except Exception:
            company_name = symbol
        
        fallback_items = [
            {
                'title': f'Latest {company_name} ({symbol}) Financial News',
                'link': f'https://finance.yahoo.com/quote/{symbol}/news',
                'published': int(time.time()),
                'source': 'fallback_yahoo'
            },
            {
                'title': f'{company_name} Stock Analysis and Updates',
                'link': f'https://finance.yahoo.com/quote/{symbol}',
                'published': int(time.time()),
                'source': 'fallback_quote'
            },
            {
                'title': f'Recent {symbol} Market Activity',
                'link': f'https://marketwatch.com/investing/stock/{symbol.lower()}',
                'published': int(time.time()),
                'source': 'fallback_marketwatch'
            }
        ]
        
        news_items.extend(fallback_items)
        logging.info(f"Using fallback news items for {symbol}")
        return news_items
        
    except Exception as e:
        logging.warning(f'Fallback news generation failed for {symbol}: {e}')
    
    # If all methods fail, return empty list
    logging.warning(f'All news sources failed for {symbol}')
    return []

def fetch_summary(symbol: str) -> Dict:
    """Generate a comprehensive daily summary for a stock using Twelve Data"""
    try:
        # Get current quote data
        quote_data = make_api_request('quote', {'symbol': symbol})
        
        if not quote_data:
            raise ValueError(f'No quote data for {symbol}')
        
        # Extract quote information
        current_price = float(quote_data.get('close', 0))
        previous_close = float(quote_data.get('previous_close', 0))
        change = current_price - previous_close if previous_close > 0 else 0
        change_percent = (change / previous_close * 100) if previous_close > 0 else 0
        
        # Get day's high/low and volume
        day_high = float(quote_data.get('high', current_price))
        day_low = float(quote_data.get('low', current_price))
        current_volume = int(float(quote_data.get('volume', 0)))
        
        # Get average volume from historical data
        try:
            daily_data = fetch_daily_data(symbol, 30)
            if not daily_data.empty:
                avg_volume = int(daily_data['Volume'].tail(20).mean())
            else:
                avg_volume = current_volume
        except Exception:
            avg_volume = current_volume
        
        # Get company profile
        profile = fetch_company_overview(symbol)
        company_name = profile.get('name', symbol)
        market_cap = profile.get('market_capitalization', 'N/A')
        pe_ratio = profile.get('pe_ratio', 'N/A')
        
        return {
            'symbol': symbol,
            'company_name': company_name,
            'current_price': current_price,
            'price_change': change,
            'price_change_pct': change_percent,
            'day_high': day_high,
            'day_low': day_low,
            'volume': current_volume,
            'avg_volume': avg_volume,
            'market_cap': market_cap,
            'pe_ratio': pe_ratio if pe_ratio not in ['N/A', 'None', '-', None] else 'N/A',
            'timestamp': int(time.time())
        }
        
    except Exception as e:
        logging.error(f'Failed to fetch summary for {symbol}: {e}')
        raise ValueError(f'Could not fetch summary for {symbol}: {str(e)}')

def generate_signal(symbol: str, short_window: int = 5, long_window: int = 20) -> Dict:
    """Generate trading signal based on SMA crossover using Twelve Data"""
    try:
        # Method 1: Try to get SMA data directly (more efficient)
        short_sma = fetch_sma_data(symbol, short_window, long_window + 10)
        long_sma = fetch_sma_data(symbol, long_window, long_window + 10)
        
        if short_sma is not None and long_sma is not None and len(short_sma) >= 2 and len(long_sma) >= 2:
            # Get current price
            current_price = fetch_price(symbol)
            
            # Take last two values to detect crossover
            S_prev, S_last = short_sma.iloc[-2], short_sma.iloc[-1]
            L_prev, L_last = long_sma.iloc[-2], long_sma.iloc[-1]
            
            # Determine signal
            signal = 'HOLD'
            if S_last > L_last and S_prev <= L_prev:
                signal = 'BUY'
            elif S_last < L_last and S_prev >= L_prev:
                signal = 'SELL'
            
            return {
                'symbol': symbol,
                'price': current_price,
                'short_sma': float(S_last),
                'long_sma': float(L_last),
                'signal': signal,
                'timestamp': int(time.time())
            }
        
        # Method 2: Fallback to historical data and calculate SMA manually
        logging.info(f"Using fallback method for {symbol}")
        hist = fetch_daily_data(symbol, long_window + 10)
        
        if hist.empty or len(hist) < long_window + 1:
            raise ValueError(f'Insufficient data for {symbol}')
        
        # Calculate SMAs manually
        close = hist['Close'].dropna()
        short_sma_calc = close.rolling(window=short_window).mean()
        long_sma_calc = close.rolling(window=long_window).mean()
        
        # Take last two values to detect crossover
        S_prev, S_last = short_sma_calc.iloc[-2], short_sma_calc.iloc[-1]
        L_prev, L_last = long_sma_calc.iloc[-2], long_sma_calc.iloc[-1]
        price = float(close.iloc[-1])
        
        # Determine signal
        signal = 'HOLD'
        if S_last > L_last and S_prev <= L_prev:
            signal = 'BUY'
        elif S_last < L_last and S_prev >= L_prev:
            signal = 'SELL'
        
        return {
            'symbol': symbol,
            'price': price,
            'short_sma': float(S_last),
            'long_sma': float(L_last),
            'signal': signal,
            'timestamp': int(time.time())
        }
    
    except Exception as e:
        logging.error(f'Failed to generate signal for {symbol}: {e}')
        raise ValueError(f'Insufficient data for {symbol}')

class ManualSignalHandler:
    """Handles manual signal creation process"""
    
    def __init__(self):
        self.pending_signals = {}  # user_id -> signal_data
        self.signal_steps = ['symbol', 'signal', 'timeframe', 'risk']
        self.data_dir = self._ensure_data_directory()
        
    def _ensure_data_directory(self):
        """Ensure data directory exists"""
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        return data_dir
        
    def start_manual_signal(self, user_id: str) -> str:
        """Start manual signal creation process"""
        self.pending_signals[user_id] = {
            'step': 0,  # Current step index
            'data': {}  # Signal data being collected
        }
        
        return """🎯 <b>MANUAL SIGNAL CREATOR</b>

📝 <b>Step 1/4:</b> Enter stock symbol
<i>Example: AAPL, TSLA, GOOGL</i>

Type the stock symbol you want to create a signal for:"""

    def process_input(self, user_id: str, input_text: str) -> str:
        """Process user input for manual signal creation"""
        if user_id not in self.pending_signals:
            return "❌ No active manual signal process. Use /manual to start."
        
        signal_data = self.pending_signals[user_id]
        current_step = signal_data['step']
        
        try:
            if current_step == 0:  # Symbol input
                return self._process_symbol(user_id, input_text.upper().strip())
            elif current_step == 1:  # Signal type (BUY/SELL)
                return self._process_signal_type(user_id, input_text.upper().strip())
            elif current_step == 2:  # Timeframe
                return self._process_timeframe(user_id, input_text.upper().strip())
            elif current_step == 3:  # Risk percentage
                return self._process_risk(user_id, input_text.strip())
        except Exception as e:
            logger.error(f"Error processing manual signal input: {e}")
            return "❌ An error occurred processing your input. Please try again."
        
        return "❌ Invalid step in manual signal process."

    def _process_symbol(self, user_id: str, symbol: str) -> str:
        """Process symbol input with validation"""
        if not symbol or len(symbol) > 10 or not symbol.isalpha():
            return "❌ Invalid symbol. Please enter a valid stock symbol (e.g., AAPL):"
        
        # Validate symbol by checking if we can get price
        try:
            price = fetch_price(symbol)
        except Exception:
            return f"❌ Could not validate symbol {symbol}. Please enter a valid stock symbol:"
        
        self.pending_signals[user_id]['data']['symbol'] = symbol
        self.pending_signals[user_id]['step'] = 1
        
        return f"""✅ Symbol: <b>{symbol}</b> (${price:.2f})

📝 <b>Step 2/4:</b> Choose signal type
<i>Select BUY or SELL</i>

Type your signal:
• <code>BUY</code> - Long position
• <code>SELL</code> - Short position"""

    def _process_signal_type(self, user_id: str, signal_type: str) -> str:
        """Process signal type input with validation"""
        if signal_type not in ['BUY', 'SELL']:
            return """❌ Invalid signal type. Please choose:
• <code>BUY</code> - Long position  
• <code>SELL</code> - Short position"""
        
        self.pending_signals[user_id]['data']['signal'] = signal_type
        self.pending_signals[user_id]['step'] = 2
        
        return f"""✅ Signal: <b>{signal_type}</b>

📝 <b>Step 3/4:</b> Set timeframe
<i>Choose your trading timeframe</i>

Type timeframe:
• <code>1H</code> - 1 Hour
• <code>4H</code> - 4 Hours  
• <code>1D</code> - 1 Day
• <code>1W</code> - 1 Week"""

    def _process_timeframe(self, user_id: str, timeframe: str) -> str:
        """Process timeframe input with validation"""
        valid_timeframes = ['1H', '4H', '1D', '1W', '2H', '8H', '3D']
        
        if timeframe not in valid_timeframes:
            return """❌ Invalid timeframe. Please choose:
• <code>1H</code> - 1 Hour
• <code>4H</code> - 4 Hours  
• <code>1D</code> - 1 Day
• <code>1W</code> - 1 Week"""
        
        self.pending_signals[user_id]['data']['timeframe'] = timeframe
        self.pending_signals[user_id]['step'] = 3
        
        return f"""✅ Timeframe: <b>{timeframe}</b>

📝 <b>Step 4/4:</b> Set risk percentage
<i>Risk percentage for stop loss calculation</i>

Type risk percentage (1-10):
<i>Example: 2.5 (for 2.5% risk)</i>"""

    def _process_risk(self, user_id: str, risk_input: str) -> str:
        """Process risk percentage and complete signal"""
        try:
            risk_pct = float(risk_input)
            if risk_pct <= 0 or risk_pct > 10:
                return "❌ Risk percentage must be between 0.1 and 10. Please enter a valid risk percentage:"
        except ValueError:
            return "❌ Invalid risk percentage. Please enter a number (e.g., 2.5):"
        
        # Complete the signal
        signal_data = self.pending_signals[user_id]['data']
        signal_data['risk_pct'] = risk_pct
        
        # Calculate levels
        try:
            current_price = fetch_price(signal_data['symbol'])
            signal_data['entry_price'] = current_price
            
            # Calculate stop loss and target based on signal type
            if signal_data['signal'] == 'BUY':
                stop_loss = current_price * (1 - risk_pct / 100)
                target = current_price + 2 * (current_price - stop_loss)  # 1:2 R/R
            else:  # SELL
                stop_loss = current_price * (1 + risk_pct / 100)
                target = current_price - 2 * (stop_loss - current_price)  # 1:2 R/R
            
            signal_data['stop_loss'] = stop_loss
            signal_data['target'] = target
            signal_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        except Exception as e:
            logger.error(f"Error calculating signal levels: {e}")
            return f"❌ Error calculating signal levels: {str(e)}"
        
        # Save to history
        self._save_manual_signal(signal_data)
        
        # Create final message
        response = self._create_signal_message(signal_data)
        
        # Clean up
        del self.pending_signals[user_id]
        
        return response

    def _create_signal_message(self, signal_data: Dict) -> str:
        """Create formatted signal message"""
        symbol = signal_data['symbol']
        signal_type = signal_data['signal']
        timeframe = signal_data['timeframe']
        entry_price = signal_data['entry_price']
        stop_loss = signal_data['stop_loss']
        target = signal_data['target']
        risk_pct = signal_data['risk_pct']
        
        # Calculate R/R ratio
        if signal_type == 'BUY':
            risk_amount = entry_price - stop_loss
            reward_amount = target - entry_price
        else:
            risk_amount = stop_loss - entry_price
            reward_amount = entry_price - target
        
        rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        signal_emoji = "🟢" if signal_type == 'BUY' else "🔴"
        
        return f"""🎯 <b>MANUAL SIGNAL CREATED</b>

{signal_emoji} <b>{symbol}</b> - <b>{signal_type}</b> ({timeframe})

💰 <b>Entry:</b> ${entry_price:.2f}
🛑 <b>Stop Loss:</b> ${stop_loss:.2f}
🎯 <b>Target:</b> ${target:.2f}

📊 <b>Risk:</b> {risk_pct}%
📈 <b>R/R Ratio:</b> 1:{rr_ratio:.1f}

⏰ <i>{signal_data['timestamp']}</i>"""

    def _save_manual_signal(self, signal_data: Dict):
        """Save manual signal to history file with error handling"""
        try:
            history_file = os.path.join(self.data_dir, 'manual_signals.json')
            
            history = []
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        history = json.loads(content)
            
            history.append(signal_data)
            
            # Keep only last 100 signals to prevent file bloat
            if len(history) > 100:
                history = history[-100:]
            
            # Write with atomic operation
            temp_file = history_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            # Atomic move
            if os.name == 'nt':  # Windows
                if os.path.exists(history_file):
                    os.remove(history_file)
                os.rename(temp_file, history_file)
            else:  # Unix-like
                os.rename(temp_file, history_file)
                
            logger.info(f"Saved manual signal for {signal_data['symbol']}")
                
        except Exception as e:
            logger.error(f"Failed to save manual signal: {e}")
            # Clean up temp file if it exists
            temp_file = os.path.join(self.data_dir, 'manual_signals.json.tmp')
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

    def cancel_signal(self, user_id: str) -> str:
        """Cancel active manual signal process"""
        if user_id in self.pending_signals:
            del self.pending_signals[user_id]
            return "❌ Manual signal creation cancelled."
        return "No active manual signal process to cancel."

    def get_signal_history(self, limit: int = 10) -> str:
        """Get manual signals history with proper error handling"""
        try:
            history_file = os.path.join(self.data_dir, 'manual_signals.json')
            
            if not os.path.exists(history_file):
                return "📝 No manual signals history found."
            
            with open(history_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    return "📝 No manual signals in history."
                
                history = json.loads(content)
            
            if not history:
                return "📝 No manual signals in history."
            
            # Get recent signals
            recent = history[-limit:] if len(history) > limit else history
            recent.reverse()  # Show newest first
            
            response = f"📋 <b>Manual Signals History</b> (Last {len(recent)})\n\n"
            
            for i, signal in enumerate(recent, 1):
                signal_emoji = "🟢" if signal['signal'] == 'BUY' else "🔴"
                response += f"{i}. {signal_emoji} <b>{signal['symbol']}</b> - {signal['signal']} ({signal['timeframe']})\n"
                response += f"   💰 ${signal['entry_price']:.2f} | 🛑 ${signal['stop_loss']:.2f} | 🎯 ${signal['target']:.2f}\n"
                response += f"   ⏰ {signal['timestamp']}\n\n"
            
            return response
            
        except json.JSONDecodeError:
            logger.error("Invalid JSON in manual signals history file")
            return "❌ Error reading signal history (corrupted file)."
        except Exception as e:
            logger.error(f"Failed to get signal history: {e}")
            return "❌ Error retrieving signal history."

class SignalChecker:
    """Production-ready signal checker with enhanced error handling and monitoring"""
    
    def __init__(self, config, bot):
        self.config = config
        self.bot = bot
        self.interval = int(config.get('CHECK_INTERVAL_SECONDS', 300))
        self.short = int(config.get('SHORT_WINDOW', 5))
        self.long = int(config.get('LONG_WINDOW', 20))
        self._running = False
        self.data_dir = self._ensure_data_directory()
        
        # Initialize manual signal handler
        self.manual_handler = ManualSignalHandler()
        
        # Statistics tracking
        self.stats = {
            'total_checks': 0,
            'successful_checks': 0,
            'failed_checks': 0,
            'last_run': None
        }
    
    def _ensure_data_directory(self):
        """Ensure data directory exists"""
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        return data_dir

    def handle_manual_command(self, user_id: str) -> str:
        """Handle /manual command"""
        return self.manual_handler.start_manual_signal(user_id)

    def handle_manual_input(self, user_id: str, input_text: str) -> str:
        """Handle manual signal input"""
        return self.manual_handler.process_input(user_id, input_text)

    def handle_cancel_manual(self, user_id: str) -> str:
        """Handle /cancel command"""
        return self.manual_handler.cancel_signal(user_id)

    def get_manual_signals_history(self, limit: int = 10) -> str:
        """Get manual signals history"""
        return self.manual_handler.get_signal_history(limit)

    def _load_symbols(self):
        """Load symbols with enhanced error handling and validation"""
        symbols_file = os.path.join(self.data_dir, 'symbols.json')
        logger.debug(f"Loading symbols from: {symbols_file}")
        
        if not os.path.exists(symbols_file):
            logger.info("Symbols file not found, creating sample file")
            
            try:
                sample_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
                with open(symbols_file, 'w') as f:
                    json.dump(sample_symbols, f, indent=2)
                logger.info(f"Created sample symbols file with: {sample_symbols}")
                return sample_symbols
            except Exception as e:
                logger.error(f"Failed to create sample symbols file: {e}")
                return []
        
        try:
            with open(symbols_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    logger.warning("Empty symbols file")
                    return []
                
                symbols = json.loads(content)
                
                # Validate symbols
                if not isinstance(symbols, list):
                    logger.error("Symbols file must contain a list")
                    return []
                
                # Filter out invalid symbols
                valid_symbols = []
                for symbol in symbols:
                    if isinstance(symbol, str) and symbol.strip() and len(symbol.strip()) <= 10:
                        valid_symbols.append(symbol.strip().upper())
                    else:
                        logger.warning(f"Skipping invalid symbol: {symbol}")
                
                logger.info(f"Successfully loaded {len(valid_symbols)} valid symbols")
                return valid_symbols
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON in symbols file")
            return []
        except Exception as e:
            logger.error(f"Error reading symbols file: {e}")
            return []

    def _load_alerts(self):
        """Load alerts with enhanced error handling and validation"""
        alerts_file = os.path.join(self.data_dir, 'alerts.json')
        logger.debug(f"Loading alerts from: {alerts_file}")
        
        if not os.path.exists(alerts_file):
            logger.info("Alerts file not found (this is optional)")
            return {}
        
        try:
            with open(alerts_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    return {}
                
                alerts = json.loads(content)
                
                # Validate alerts format
                if not isinstance(alerts, dict):
                    logger.error("Alerts file must contain a dictionary")
                    return {}
                
                # Validate and clean alerts
                valid_alerts = {}
                for symbol, threshold in alerts.items():
                    try:
                        if isinstance(symbol, str) and symbol.strip():
                            threshold_float = float(threshold)
                            if 0 < threshold_float <= 100:  # Reasonable threshold range
                                valid_alerts[symbol.strip().upper()] = threshold_float
                            else:
                                logger.warning(f"Skipping invalid threshold for {symbol}: {threshold}")
                        else:
                            logger.warning(f"Skipping invalid symbol in alerts: {symbol}")
                    except (ValueError, TypeError):
                        logger.warning(f"Skipping invalid alert entry: {symbol}={threshold}")
                
                logger.info(f"Successfully loaded {len(valid_alerts)} valid alerts")
                return valid_alerts
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON in alerts file")
            return {}
        except Exception as e:
            logger.error(f"Error reading alerts file: {e}")
            return {}

    def start_periodic_checks(self):
        """Start periodic signal checks with enhanced monitoring"""
        self._running = True
        logger.info(f'Starting signal checks every {self.interval} seconds.')
        
        try:
            while self._running:
                try:
                    self.run_once()
                    self.stats['last_run'] = datetime.now().isoformat()
                except Exception as e:
                    logger.exception('Error during signal check cycle')
                    self.stats['failed_checks'] += 1
                    
                    # Send error notification to bot
                    try:
                        error_msg = f"🚨 Signal checker error: {str(e)[:100]}"
                        self.bot.post_message_from_thread(error_msg)
                    except Exception:
                        pass
                
                if self._running:  # Check if still running before sleeping
                    time.sleep(self.interval)
                    
        except KeyboardInterrupt:
            logger.info("Signal checker stopped by user")
        finally:
            self._running = False

    def stop(self):
        """Stop signal checking"""
        self._running = False
        logger.info("Signal checker stop requested")

    def run_once(self):
        """Enhanced single check run with comprehensive error handling"""
        start_time = time.time()
        logger.info("Starting signal check cycle")
        
        # Update statistics
        self.stats['total_checks'] += 1
        
        # Load symbols and alerts
        symbols = self._load_symbols()
        logger.info(f"Loaded {len(symbols)} symbols for checking")
        
        if not symbols:
            error_msg = '⚠️ No symbols to check. Add symbols with /addsymbol command.'
            logger.warning("No symbols available for checking")
            try:
                self.bot.post_message_from_thread(error_msg)
            except Exception as e:
                logger.error(f"Failed to send 'no symbols' message: {e}")
            return
        
        alerts = self._load_alerts()
        logger.info(f"Loaded {len(alerts)} price alerts")
        
        successful_checks = 0
        failed_checks = 0
        
        for i, symbol in enumerate(symbols):
            logger.debug(f"Processing symbol {i+1}/{len(symbols)}: {symbol}")
            
            try:
                # Rate limiting - respect API limits
                if i > 0:
                    time.sleep(RATE_LIMIT_DELAY)
                
                # Generate trading signal
                logger.debug(f"Generating signal for {symbol}")
                signal_result = generate_signal(symbol, short_window=self.short, long_window=self.long)
                logger.debug(f"Signal result for {symbol}: {signal_result}")
                
                # Check price alerts
                alert_text = self._check_price_alert(symbol, alerts)
                
                # Format and send message
                message_text = self._format_signal_message(signal_result, alert_text)
                
                # Send message via bot
                self.bot.post_message_from_thread(message_text)
                successful_checks += 1
                logger.debug(f"Successfully processed {symbol}")
                
            except Exception as e:
                failed_checks += 1
                error_msg = f'❌ Error checking {symbol}: {str(e)[:100]}'
                logger.error(f"Error processing {symbol}: {e}")
                
                # Send individual error message
                try:
                    self.bot.post_message_from_thread(error_msg)
                except Exception as bot_error:
                    logger.error(f"Failed to send error message for {symbol}: {bot_error}")
        
        # Update statistics
        self.stats['successful_checks'] += successful_checks
        self.stats['failed_checks'] += failed_checks
        
        # Send summary
        duration = time.time() - start_time
        summary = self._create_summary_message(successful_checks, failed_checks, len(symbols), duration)
        
        try:
            self.bot.post_message_from_thread(summary)
        except Exception as e:
            logger.error(f"Failed to send summary message: {e}")
        
        logger.info(f"Signal check cycle completed: {successful_checks}/{len(symbols)} successful in {duration:.1f}s")

    def _check_price_alert(self, symbol: str, alerts: Dict) -> str:
        """Check price alerts for a symbol"""
        if symbol not in alerts:
            return ''
        
        try:
            alert_threshold = alerts[symbol]
            logger.debug(f"Checking alert for {symbol} (threshold: {alert_threshold}%)")
            
            quote_data = make_api_request('quote', {'symbol': symbol})
            
            if not quote_data:
                return ''
            
            current_price = float(quote_data.get('close', 0))
            prev_close = float(quote_data.get('previous_close', 0))
            
            if prev_close <= 0:
                return ''
            
            change_percent = (current_price - prev_close) / prev_close * 100.0
            logger.debug(f"Price change for {symbol}: {change_percent:.2f}%")
            
            if abs(change_percent) >= alert_threshold:
                direction = "📈" if change_percent > 0 else "📉"
                return f"\n\n🚨 <b>PRICE ALERT</b> {direction}\n{symbol} moved {change_percent:+.2f}% (threshold: {alert_threshold}%)"
            
        except Exception as e:
            logger.warning(f'Alert check failed for {symbol}: {e}')
        
        return ''

    def _format_signal_message(self, signal_result: Dict, alert_text: str) -> str:
        """Format signal message with consistent styling"""
        signal_emoji_map = {
            'BUY': '🟢',
            'SELL': '🔴', 
            'HOLD': '🟡'
        }
        
        signal_emoji = signal_emoji_map.get(signal_result['signal'], '🟡')
        
        message = (
            f"<b>{signal_result['symbol']}</b> {signal_emoji} <b>{signal_result['signal']}</b>\n"
            f"💰 Price: ${signal_result['price']:.2f}\n"
            f"📊 Short SMA({self.short}): ${signal_result['short_sma']:.2f}\n"
            f"📊 Long SMA({self.long}): ${signal_result['long_sma']:.2f}"
            f"{alert_text}"
        )
        
        return message

    def _create_summary_message(self, successful: int, failed: int, total: int, duration: float) -> str:
        """Create comprehensive summary message"""
        success_rate = (successful / total * 100) if total > 0 else 0
        
        summary = f"📊 <b>Check Summary</b>\n"
        summary += f"✅ Successful: {successful}/{total} ({success_rate:.1f}%)\n"
        
        if failed > 0:
            summary += f"❌ Failed: {failed}\n"
        
        summary += f"⏱️ Duration: {duration:.1f}s\n"
        summary += f"🔄 Total checks run: {self.stats['total_checks']}"
        
        return summary

    def get_statistics(self) -> Dict:
        """Get signal checker statistics"""
        return self.stats.copy()