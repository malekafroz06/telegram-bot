import requests
import pandas as pd
import os, json, time, logging, asyncio
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
API_TIMEOUT = 10
RATE_LIMIT_DELAY = 1.0
MAX_RETRIES = 2

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
        """Get Twelve Data API key from config or environment"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    api_key = config.get('TWELVE_DATA_API_KEY')
                    if api_key and api_key != 'YOUR_TWELVE_DATA_API_KEY_HERE':
                        return api_key
        except Exception:
            pass
        
        api_key = os.getenv('TWELVE_DATA_API_KEY')
        if not api_key:
            raise ValueError('TWELVE_DATA_API_KEY not found in config.json or environment variables')
        
        return api_key
    
    def make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make request to Twelve Data API with error handling"""
        if params is None:
            params = {}
        
        params['apikey'] = self.api_key
        url = f"{TWELVE_DATA_BASE_URL}/{endpoint}"
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(url, params=params, timeout=API_TIMEOUT)
                response.raise_for_status()
                
                data = response.json()
                
                if isinstance(data, dict):
                    if data.get('status') == 'error':
                        error_msg = data.get('message', 'Unknown error')
                        raise APIError(f"Twelve Data API Error: {error_msg}")
                    
                    if data.get('code') == 429:
                        wait_time = 3 * (attempt + 1)
                        time.sleep(wait_time)
                        continue
                
                return data
                
            except requests.exceptions.Timeout:
                if attempt == MAX_RETRIES - 1:
                    raise APIError("Request timeout")
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait_time = 3 * (attempt + 1)
                    time.sleep(wait_time)
                    continue
                raise APIError(f"HTTP error {e.response.status_code}")
                
            except requests.exceptions.RequestException as e:
                if attempt == MAX_RETRIES - 1:
                    raise APIError(f"Network error: {str(e)}")
                
            except json.JSONDecodeError:
                raise APIError("Invalid JSON response")
                
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
        
        raise APIError("Max retries exceeded")

# Global Twelve Data client instance
twelve_data_client = TwelveDataClient()

def get_api_key():
    """Get Twelve Data API key"""
    return twelve_data_client.api_key

def make_api_request(endpoint: str, params: Dict = None) -> Dict:
    """Make a request to Twelve Data API"""
    return twelve_data_client.make_request(endpoint, params)

def fetch_price(symbol: str) -> float:
    """Fetch current price using Twelve Data"""
    try:
        data = make_api_request('price', {'symbol': symbol})
        
        if 'price' not in data:
            raise ValueError(f'No price data for {symbol}')
        
        return float(data['price'])
        
    except Exception as e:
        raise ValueError(f'No data for {symbol}')

def fetch_daily_data(symbol: str, days: int = 100) -> pd.DataFrame:
    """Fetch daily historical data from Twelve Data"""
    params = {
        'symbol': symbol,
        'interval': '1day',
        'outputsize': min(days, 5000),
        'format': 'JSON'
    }
    
    try:
        data = make_api_request('time_series', params)
        
        if 'values' not in data or not data['values']:
            raise ValueError(f'No daily data for {symbol}')
        
        df = pd.DataFrame(data['values'])
        
        if df.empty:
            raise ValueError(f'Empty dataset returned for {symbol}')
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.sort_index()
        
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        if df.empty:
            raise ValueError(f'No valid data after processing for {symbol}')
        
        return df
    
    except Exception as e:
        raise ValueError(f'Insufficient data for {symbol}')

def fetch_sma_data(symbol: str, window: int, days: int = 100) -> pd.Series:
    """Fetch SMA data directly from Twelve Data"""
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
        
        sma_data = []
        dates = []
        for item in data['values']:
            try:
                dates.append(pd.to_datetime(item['datetime']))
                sma_data.append(float(item['sma']))
            except (KeyError, ValueError):
                continue
        
        if not sma_data:
            raise ValueError(f'No valid SMA data for {symbol}')
        
        sma_series = pd.Series(sma_data, index=dates)
        return sma_series.sort_index()
    
    except Exception:
        return None

def fetch_company_overview(symbol: str) -> Dict:
    """Fetch company profile from Twelve Data"""
    try:
        data = make_api_request('profile', {'symbol': symbol})
        
        if not data or 'name' not in data:
            return {
                'name': symbol,
                'symbol': symbol,
                'sector': 'N/A',
                'industry': 'N/A',
                'market_capitalization': 'N/A',
                'pe_ratio': 'N/A',
                'employees': 'N/A'
            }
        
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
    
    except Exception:
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
    """Fetch news for a given symbol"""
    news_items = []
    
    # Method 1: Try Twelve Data news endpoint
    try:
        data = make_api_request('news', {'symbol': symbol, 'limit': 5})
        
        if 'feed' in data and data['feed']:
            for item in data['feed'][:5]:
                title = item.get('title', '')
                url = item.get('url', '')
                published = item.get('published_utc', '')
                
                if title and url:
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
                return news_items
                
    except Exception:
        pass
    
    # Method 2: Try Yahoo RSS feed as fallback
    try:
        rss_url = f"https://finance.yahoo.com/rss/headline?s={symbol}"
        feed = feedparser.parse(rss_url)
        
        if hasattr(feed, 'entries') and feed.entries:
            for entry in feed.entries[:5]:
                title = entry.get('title', '')
                link = entry.get('link', '')
                
                if title and link:
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
            return news_items
            
    except Exception:
        pass
    
    # Method 3: Fallback news items
    try:
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
        return news_items
        
    except Exception:
        pass
    
    return []

def fetch_summary(symbol: str) -> Dict:
    """Generate a comprehensive daily summary for a stock using Twelve Data"""
    try:
        quote_data = make_api_request('quote', {'symbol': symbol})
        
        if not quote_data:
            raise ValueError(f'No quote data for {symbol}')
        
        current_price = float(quote_data.get('close', 0))
        previous_close = float(quote_data.get('previous_close', 0))
        change = current_price - previous_close if previous_close > 0 else 0
        change_percent = (change / previous_close * 100) if previous_close > 0 else 0
        
        day_high = float(quote_data.get('high', current_price))
        day_low = float(quote_data.get('low', current_price))
        current_volume = int(float(quote_data.get('volume', 0)))
        
        try:
            daily_data = fetch_daily_data(symbol, 30)
            if not daily_data.empty:
                avg_volume = int(daily_data['Volume'].tail(20).mean())
            else:
                avg_volume = current_volume
        except Exception:
            avg_volume = current_volume
        
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
        raise ValueError(f'Could not fetch summary for {symbol}: {str(e)}')

def generate_signal(symbol: str, short_window: int = 5, long_window: int = 20) -> Dict:
    """Generate trading signal based on SMA crossover using Twelve Data"""
    try:
        # Method 1: Try to get SMA data directly
        short_sma = fetch_sma_data(symbol, short_window, long_window + 10)
        long_sma = fetch_sma_data(symbol, long_window, long_window + 10)
        
        if short_sma is not None and long_sma is not None and len(short_sma) >= 2 and len(long_sma) >= 2:
            current_price = fetch_price(symbol)
            
            S_prev, S_last = short_sma.iloc[-2], short_sma.iloc[-1]
            L_prev, L_last = long_sma.iloc[-2], long_sma.iloc[-1]
            
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
        
        # Method 2: Fallback to historical data
        hist = fetch_daily_data(symbol, long_window + 10)
        
        if hist.empty or len(hist) < long_window + 1:
            raise ValueError(f'Insufficient data for {symbol}')
        
        close = hist['Close'].dropna()
        short_sma_calc = close.rolling(window=short_window).mean()
        long_sma_calc = close.rolling(window=long_window).mean()
        
        S_prev, S_last = short_sma_calc.iloc[-2], short_sma_calc.iloc[-1]
        L_prev, L_last = long_sma_calc.iloc[-2], long_sma_calc.iloc[-1]
        price = float(close.iloc[-1])
        
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
    
    except Exception:
        raise ValueError(f'Insufficient data for {symbol}')

class CustomSignalHandler:
    """ULTRA-OPTIMIZED custom signal handler with instant sticker and message sending"""
    
    def __init__(self, bot):
        self.bot = bot
        self.pending_signals = {}
        self.signal_steps = ['pair', 'time', 'risk', 'direction', 'result', 'mtg_direction', 'final_result', 'screenshot']
        self.data_dir = self._ensure_data_directory()
        self.broker_link = "https://qxbroker.com/en?lid=1227277"
        
        # Use sticker names for instant sending
        self.sticker_names = {
            'UP': 'up',
            'DOWN': 'down', 
            'WIN': 'win',
            'MTGUP': 'mtg_up',
            'MTGDOWN': 'mtg_down'
        }

    def _ensure_data_directory(self):
        """Ensure data directory exists"""
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        return data_dir
        
    def start_custom_signal(self, user_id: str) -> str:
        """Start custom signal creation process"""
        self.pending_signals[user_id] = {
            'step': 0,
            'data': {}
        }
        
        return """🎯 <b>CUSTOM SIGNAL CREATOR</b>

📝 <b>Step 1/8:</b> Enter PAIR name
<i>Example: EUR JPY OTC</i>

Type the currency pair you want to trade:"""

    def process_input(self, user_id: str, input_text: str = None, callback_data: str = None) -> str:
        """Process user input for custom signal creation"""
        if user_id not in self.pending_signals:
            return "❌ No active custom signal process. Use /custom_signal to start."
        
        signal_data = self.pending_signals[user_id]
        current_step = signal_data['step']
        
        try:
            if current_step == 0:  # Pair input
                if input_text:
                    return self._process_pair(user_id, input_text.strip())
                else:
                    return "❌ Please enter a currency pair."
                    
            elif current_step == 1:  # Time input
                if input_text:
                    return self._process_time(user_id, input_text.strip())
                else:
                    return "❌ Please enter a timeframe."
                    
            elif current_step == 2:  # Risk input
                if input_text:
                    return self._process_risk(user_id, input_text.strip())
                else:
                    return "❌ Please enter a risk amount."
                    
            elif current_step == 3:  # Direction
                if callback_data in ['UP', 'DOWN']:
                    return self._process_direction(user_id, callback_data)
                else:
                    return "❌ Please select UP or DOWN using the buttons above."
                    
            elif current_step == 4:  # First Result
                if callback_data in ['WIN', 'LOSS']:
                    return self._process_first_result(user_id, callback_data)
                else:
                    return "❌ Please select WIN or LOSS using the buttons above."
                    
            elif current_step == 5:  # MTG Direction
                if callback_data in ['MTGUP', 'MTGDOWN']:
                    return self._process_mtg_direction(user_id, callback_data)
                else:
                    return "❌ Please select MTGUP or MTGDOWN using the buttons above."
                    
            elif current_step == 6:  # Final Result
                if callback_data in ['WIN', 'LOSS']:
                    return self._process_final_result(user_id, callback_data)
                else:
                    return "❌ Please select WIN or LOSS using the buttons above."
                    
            elif current_step == 7:  # Screenshot or new signal
                if callback_data in ['SCREENSHOT', 'NEW_SIGNAL']:
                    return self._process_final_action(user_id, callback_data, input_text)
                else:
                    return "❌ Please choose an action using the buttons above."
                    
        except Exception:
            return "❌ An error occurred processing your input. Please try again or use /cancel to restart."
        
        return "❌ Invalid step in custom signal process."

    def _post_to_channel_instant(self, message: str):
        """Post message to channel INSTANTLY without blocking - fire and forget"""
        if self.bot.main_loop and not self.bot.main_loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self.bot.post_message(message),
                self.bot.main_loop
            )

    def _send_sticker_instantly(self, direction_or_result: str):
        """Send sticker instantly using optimized method - fire and forget"""
        sticker_name = self.sticker_names.get(direction_or_result)
        
        if sticker_name:
            # Fire and forget - don't wait
            self.bot.send_sticker_to_channel_instant(sticker_name)

    def _process_pair(self, user_id: str, pair: str) -> str:
        """Process pair input with validation"""
        if not pair or len(pair) > 50:
            return "❌ Invalid pair. Please enter a valid currency pair (e.g., EUR JPY OTC):"
        
        self.pending_signals[user_id]['data']['pair'] = pair.upper()
        self.pending_signals[user_id]['step'] = 1
        
        return f"""✅ Pair: <b>{pair.upper()}</b>

📝 <b>Step 2/8:</b> Enter Time
<i>Example: 1 MINUTE</i>

Type the timeframe for your trade:"""

    def _process_time(self, user_id: str, time_input: str) -> str:
        """Process time input with validation"""
        if not time_input or len(time_input) > 20:
            return "❌ Invalid timeframe. Please enter a valid timeframe (e.g., 1 MINUTE):"
        
        self.pending_signals[user_id]['data']['time'] = time_input.upper()
        self.pending_signals[user_id]['step'] = 2
        
        return f"""✅ Time: <b>{time_input.upper()}</b>

📝 <b>Step 3/8:</b> Enter My Risk
<i>Example: Rs 200</i>

Type your risk amount:"""

    def _process_risk(self, user_id: str, risk_input: str) -> str:
        """Process risk input and show broker link - OPTIMIZED for instant posting"""
        if not risk_input or len(risk_input) > 20:
            return "❌ Invalid risk amount. Please enter a valid amount (e.g., Rs 200):"
        
        self.pending_signals[user_id]['data']['risk'] = risk_input
        
        # Create and send the signal message with broker link to channel - INSTANT FIRE AND FORGET
        signal_message = self._create_signal_message(self.pending_signals[user_id]['data'])
        self._post_to_channel_instant(signal_message)
        
        self.pending_signals[user_id]['step'] = 3
        
        return f"""✅ Risk: <b>{risk_input}</b>

📊 Signal posted to channel instantly! ⚡

📝 <b>Step 4/8:</b> Enter signal direction
Choose your direction:"""

    def _process_direction(self, user_id: str, direction: str) -> str:
        """Process direction selection and send sticker INSTANTLY"""
        if direction not in ['UP', 'DOWN']:
            return "❌ Please select UP or DOWN direction."
        
        self.pending_signals[user_id]['data']['direction'] = direction
        
        # Send sticker INSTANTLY - fire and forget
        self._send_sticker_instantly(direction)
        
        # Move to next step immediately
        self.pending_signals[user_id]['step'] = 4
        
        return f"""✅ Direction: <b>{direction}</b> 🎯

{direction} sticker sent instantly! ⚡

📝 <b>Step 5/8:</b> Enter Signal Result
Choose the result:"""

    def _process_first_result(self, user_id: str, result: str) -> str:
        """Process first result selection - INSTANT POSTING"""
        if result not in ['WIN', 'LOSS']:
            return "❌ Please select WIN or LOSS."
        
        self.pending_signals[user_id]['data']['first_result'] = result
        
        if result == 'WIN':
            # Send WIN result message and sticker INSTANTLY - fire and forget
            result_message = f"📊 <b>RESULT: {result}</b> 🎉"
            self._post_to_channel_instant(result_message)
            self._send_sticker_instantly('WIN')
            
            # Skip MTG steps
            self.pending_signals[user_id]['step'] = 7
            
            return f"""✅ Result: <b>{result}</b> 🎉

WIN result and sticker posted instantly! ⚡

📝 <b>Step 8/8:</b> Final Action
Choose your next action:"""
        
        else:  # LOSS - go to MTG
            self.pending_signals[user_id]['step'] = 5
            
            return f"""✅ First Result: <b>{result}</b> 💔

Now choose MTG direction for recovery:

📝 <b>Step 6/8:</b> Select MTG Direction
Choose MTG direction:"""

    def _process_mtg_direction(self, user_id: str, mtg_direction: str) -> str:
        """Process MTG direction selection and send MTG sticker INSTANTLY"""
        if mtg_direction not in ['MTGUP', 'MTGDOWN']:
            return "❌ Please select MTGUP or MTGDOWN."
        
        self.pending_signals[user_id]['data']['mtg_direction'] = mtg_direction
        
        # Send MTG sticker INSTANTLY - fire and forget
        self._send_sticker_instantly(mtg_direction)
        
        # Move to final result step
        self.pending_signals[user_id]['step'] = 6
        
        return f"""✅ MTG Direction: <b>{mtg_direction}</b> 🎯

MTG {mtg_direction} sticker sent instantly! ⚡

📝 <b>Step 7/8:</b> Final Result
Choose the final result:"""

    def _process_final_result(self, user_id: str, final_result: str) -> str:
        """Process final result after MTG - INSTANT POSTING"""
        if final_result not in ['WIN', 'LOSS']:
            return "❌ Please select WIN or LOSS."
        
        self.pending_signals[user_id]['data']['final_result'] = final_result
        self.pending_signals[user_id]['step'] = 7
        
        # Send final result to channel INSTANTLY - fire and forget
        if final_result == 'WIN':
            result_message = f"📊 <b>FINAL RESULT: {final_result}</b> 🎉"
            self._post_to_channel_instant(result_message)
            self._send_sticker_instantly('WIN')
            status_message = "Final WIN result and sticker posted instantly! ⚡"
        else:  # LOSS
            result_message = f"📊 <b>FINAL RESULT: {final_result}</b> 💔"
            self._post_to_channel_instant(result_message)
            status_message = "Final LOSS result posted to channel instantly! ⚡"
        
        return f"""✅ Final Result: <b>{final_result}</b> {'🎉' if final_result == 'WIN' else '💔'}

{status_message}

📝 <b>Step 8/8:</b> Final Action
Choose your next action:"""

    def _process_final_action(self, user_id: str, action: str, screenshot_text: str = None) -> str:
        """Process final action"""
        if action == 'NEW_SIGNAL':
            # Clean up current signal and start new one
            del self.pending_signals[user_id]
            return self.start_custom_signal(user_id)
        
        elif action == 'SCREENSHOT':
            # Note: Screenshot will be handled by photo upload handler in bot.py
            # Just save the completed signal here
            self._save_custom_signal(self.pending_signals[user_id]['data'])
            
            return """📸 <b>Ready for Screenshot Upload</b>

Please send your screenshot now.
It will be posted to the channel instantly! ⚡"""
        
        return "❌ Please choose a valid action."

    def _create_signal_message(self, signal_data: Dict) -> str:
        """Create formatted signal message with broker link"""
        pair = signal_data.get('pair', 'N/A')
        time = signal_data.get('time', 'N/A')
        risk = signal_data.get('risk', 'N/A')
        
        message = f"""✅✅✅✅✅✅✅✅✅
🎯 <b>PAIR</b> ⚡ <b>{pair}</b>
⏱️ <b>TIMER</b> 📊 <b>{time}</b>
💰 <b>MY RISK</b> ⚡ <b>{risk}</b>⭐
🖥️ <a href="{self.broker_link}">CLICK TO OPEN BROKER</a>🖥️"""

        return message

    def _save_custom_signal(self, signal_data: Dict):
        """Save custom signal to history file"""
        try:
            signal_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            history_file = os.path.join(self.data_dir, 'custom_signals.json')
            
            history = []
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        history = json.loads(content)
            
            history.append(signal_data)
            
            # Keep only last 100 signals
            if len(history) > 100:
                history = history[-100:]
            
            # Write with atomic operation
            temp_file = history_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            # Atomic move
            if os.name == 'nt':
                if os.path.exists(history_file):
                    os.remove(history_file)
                os.rename(temp_file, history_file)
            else:
                os.rename(temp_file, history_file)
                
        except Exception:
            temp_file = os.path.join(self.data_dir, 'custom_signals.json.tmp')
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

    def cancel_signal(self, user_id: str) -> str:
        """Cancel active custom signal process"""
        if user_id in self.pending_signals:
            del self.pending_signals[user_id]
            return "❌ Custom signal creation cancelled."
        return "No active custom signal process to cancel."

    def get_signal_history(self, limit: int = 10) -> str:
        """Get custom signals history"""
        try:
            history_file = os.path.join(self.data_dir, 'custom_signals.json')
            
            if not os.path.exists(history_file):
                return "📝 No custom signals history found."
            
            with open(history_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    return "📝 No custom signals in history."
                
                history = json.loads(content)
            
            if not history:
                return "📝 No custom signals in history."
            
            recent = history[-limit:] if len(history) > limit else history
            recent.reverse()
            
            response = f"📋 <b>Custom Signals History</b> (Last {len(recent)})\n\n"
            
            for i, signal in enumerate(recent, 1):
                direction_emoji = "🔼" if signal.get('direction') == 'UP' else "🔽" if signal.get('direction') == 'DOWN' else "❓"
                
                first_result = signal.get('first_result', signal.get('result', 'N/A'))
                final_result = signal.get('final_result')
                mtg_direction = signal.get('mtg_direction')
                
                if first_result == 'WIN':
                    result_display = "🎉 WIN"
                elif first_result == 'LOSS' and final_result:
                    mtg_emoji = "🔼" if mtg_direction == 'MTGUP' else "🔽" if mtg_direction == 'MTGDOWN' else ""
                    final_emoji = "🎉" if final_result == 'WIN' else "💔"
                    result_display = f"💔 LOSS → {mtg_emoji} {mtg_direction} → {final_emoji} {final_result}"
                else:
                    result_display = f"{'🎉' if first_result == 'WIN' else '💔' if first_result == 'LOSS' else '❓'} {first_result}"
                
                response += f"{i}. {direction_emoji} <b>{signal.get('pair', 'N/A')}</b> - {signal.get('time', 'N/A')}\n"
                response += f"   💰 {signal.get('risk', 'N/A')} | {result_display}\n"
                response += f"   ⏰ {signal.get('timestamp', 'N/A')}\n\n"
            
            return response
            
        except json.JSONDecodeError:
            return "❌ Error reading signal history (corrupted file)."
        except Exception:
            return "❌ Error retrieving signal history."

    def get_user_current_step(self, user_id: str) -> int:
        """Get current step for user"""
        if user_id in self.pending_signals:
            return self.pending_signals[user_id]['step']
        return -1

    def is_user_in_process(self, user_id: str) -> bool:
        """Check if user has active signal process"""
        return user_id in self.pending_signals


class SignalChecker:
    """Production-ready signal checker"""
    
    def __init__(self, config, bot):
        self.config = config
        self.bot = bot
        self.interval = int(config.get('CHECK_INTERVAL_SECONDS', 300))
        self.short = int(config.get('SHORT_WINDOW', 5))
        self.long = int(config.get('LONG_WINDOW', 20))
        self._running = False
        self.data_dir = self._ensure_data_directory()
        
        # Initialize custom signal handler
        self.custom_handler = CustomSignalHandler(bot)
        
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

    def _load_symbols(self):
        """Load symbols with error handling"""
        symbols_file = os.path.join(self.data_dir, 'symbols.json')
        
        if not os.path.exists(symbols_file):
            try:
                sample_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
                with open(symbols_file, 'w') as f:
                    json.dump(sample_symbols, f, indent=2)
                return sample_symbols
            except Exception:
                return []
        
        try:
            with open(symbols_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    return []
                
                symbols = json.loads(content)
                
                if not isinstance(symbols, list):
                    return []
                
                valid_symbols = []
                for symbol in symbols:
                    if isinstance(symbol, str) and symbol.strip() and len(symbol.strip()) <= 10:
                        valid_symbols.append(symbol.strip().upper())
                
                return valid_symbols
                
        except Exception:
            return []

    def _load_alerts(self):
        """Load alerts with error handling"""
        alerts_file = os.path.join(self.data_dir, 'alerts.json')
        
        if not os.path.exists(alerts_file):
            return {}
        
        try:
            with open(alerts_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    return {}
                
                alerts = json.loads(content)
                
                if not isinstance(alerts, dict):
                    return {}
                
                valid_alerts = {}
                for symbol, threshold in alerts.items():
                    try:
                        if isinstance(symbol, str) and symbol.strip():
                            threshold_float = float(threshold)
                            if 0 < threshold_float <= 100:
                                valid_alerts[symbol.strip().upper()] = threshold_float
                    except (ValueError, TypeError):
                        continue
                
                return valid_alerts
                
        except Exception:
            return {}

    def start_periodic_checks(self):
        """Start periodic signal checks"""
        self._running = True
        
        try:
            while self._running:
                try:
                    self.run_once()
                    self.stats['last_run'] = datetime.now().isoformat()
                except Exception:
                    self.stats['failed_checks'] += 1
                
                if self._running:
                    time.sleep(self.interval)
                    
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False

    def stop(self):
        """Stop signal checking"""
        self._running = False

    def run_once(self):
        """Single check run"""
        start_time = time.time()
        
        self.stats['total_checks'] += 1
        
        symbols = self._load_symbols()
        
        if not symbols:
            try:
                self.bot.post_message_from_thread('⚠️ No symbols to check. Add symbols with /addsymbol command.')
            except Exception:
                pass
            return
        
        alerts = self._load_alerts()
        
        successful_checks = 0
        failed_checks = 0
        
        for i, symbol in enumerate(symbols):
            try:
                if i > 0:
                    time.sleep(RATE_LIMIT_DELAY)
                
                signal_result = generate_signal(symbol, short_window=self.short, long_window=self.long)
                alert_text = self._check_price_alert(symbol, alerts)
                message_text = self._format_signal_message(signal_result, alert_text)
                
                self.bot.post_message_from_thread(message_text)
                successful_checks += 1
                
            except Exception:
                failed_checks += 1
                error_msg = f'❌ Error checking {symbol}'
                
                try:
                    self.bot.post_message_from_thread(error_msg)
                except Exception:
                    pass
        
        self.stats['successful_checks'] += successful_checks
        self.stats['failed_checks'] += failed_checks
        
        duration = time.time() - start_time
        summary = self._create_summary_message(successful_checks, failed_checks, len(symbols), duration)
        
        try:
            self.bot.post_message_from_thread(summary)
        except Exception:
            pass

    def _check_price_alert(self, symbol: str, alerts: Dict) -> str:
        """Check price alerts for a symbol"""
        if symbol not in alerts:
            return ''
        
        try:
            alert_threshold = alerts[symbol]
            
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