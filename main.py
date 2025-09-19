import json
import os
import logging
import signal
import sys
import time
from pathlib import Path
from datetime import datetime
import re  # <-- Added for EmojiFilter

# Import bot module (will be provided by user)
try:
    from bot import TradingBot
except ImportError:
    print("❌ Error: bot.py not found. Please ensure bot.py is in the same directory.")
    sys.exit(1)

# -------------------------------
# Emoji Filter (for Windows)
# -------------------------------
class EmojiFilter(logging.Filter):
    """Filter to remove emojis from log records (for Windows console compatibility)."""
    def filter(self, record):
        record.msg = re.sub(r'[^\x00-\x7F]+', '', str(record.msg))
        return True

# Configure comprehensive logging
def setup_logging():
    """Setup production-ready logging configuration"""
    # Create logs directory
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Setup handlers with UTF-8 encoding
    file_handler = logging.FileHandler(logs_dir / 'tradepulse.log', encoding='utf-8')
    error_handler = logging.FileHandler(logs_dir / 'tradepulse_error.log', encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    
    # Console handler with UTF-8 for Windows compatibility
    console_handler = logging.StreamHandler(sys.stdout)
    try:
        # Try to set UTF-8 encoding for console on Windows
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        console_handler.stream.encoding = 'utf-8'
    except (AttributeError, OSError):
        # Fallback: strip emojis for Windows console compatibility
        console_handler.addFilter(EmojiFilter())
    
    handlers = [file_handler, error_handler, console_handler]
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

# -------------------------------
# Rest of your code (unchanged)
# -------------------------------

class ConfigurationManager:
    """Handles configuration loading and validation"""
    REQUIRED_KEYS = [
        'TELEGRAM_TOKEN',
        'CHANNEL_ID', 
        'TWELVE_DATA_API_KEY'
    ]
    
    DEFAULT_VALUES = {
        'ADMIN_IDS': [],
        'MORNING_TIME': '09:00',
        'TIMEZONE_OFFSET': 0,
        'CHECK_INTERVAL_SECONDS': 300,
        'SHORT_WINDOW': 5,
        'LONG_WINDOW': 20,
        'MAX_SYMBOLS': 20,
        'MAX_NEWS_ITEMS': 5,
        'RATE_LIMIT_DELAY': 1.0
    }
    
    def __init__(self, config_file='config.json'):
        self.config_file = Path(config_file)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_config(self) -> dict:
        if not self.config_file.exists():
            self.logger.error(f"Configuration file {self.config_file} not found")
            self._create_sample_config()
            self._display_config_instructions()
            raise SystemExit(1)
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in configuration file: {e}")
            raise SystemExit(1)
        except Exception as e:
            self.logger.error(f"Error reading configuration file: {e}")
            raise SystemExit(1)
        
        missing_keys = [key for key in self.REQUIRED_KEYS if not config.get(key)]
        if missing_keys:
            self.logger.error(f"Missing required configuration keys: {', '.join(missing_keys)}")
            self._display_config_instructions()
            raise SystemExit(1)
        
        placeholder_keys = []
        for key in self.REQUIRED_KEYS:
            value = config.get(key, '')
            if isinstance(value, str) and ('YOUR_' in value.upper() or '_HERE' in value.upper()):
                placeholder_keys.append(key)
        
        if placeholder_keys:
            self.logger.error(f"Configuration contains placeholder values for: {', '.join(placeholder_keys)}")
            self._display_config_instructions()
            raise SystemExit(1)
        
        for key, default_value in self.DEFAULT_VALUES.items():
            config.setdefault(key, default_value)
        
        self._validate_config(config)
        self.logger.info("Configuration loaded and validated successfully")
        return config
    
    def _validate_config(self, config: dict):
        token = config.get('TELEGRAM_TOKEN', '')
        if not token or ':' not in token or len(token) < 20:
            raise ValueError("Invalid TELEGRAM_TOKEN format")
        
        channel_id = config.get('CHANNEL_ID', '')
        if not str(channel_id).startswith(('-', '@')) and not str(channel_id).isdigit():
            self.logger.warning("CHANNEL_ID should start with '-' for groups or '@' for channels")
        
        numeric_fields = {
            'CHECK_INTERVAL_SECONDS': (60, 3600),
            'SHORT_WINDOW': (1, 50),
            'LONG_WINDOW': (2, 200),
            'TIMEZONE_OFFSET': (-12, 12),
            'MAX_SYMBOLS': (1, 100),
            'MAX_NEWS_ITEMS': (1, 20),
            'RATE_LIMIT_DELAY': (0.1, 10.0)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            value = config.get(field)
            if value is not None:
                try:
                    num_value = float(value)
                    if not (min_val <= num_value <= max_val):
                        raise ValueError(f"{field} must be between {min_val} and {max_val}")
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid {field}: {e}")
        
        if config.get('SHORT_WINDOW', 0) >= config.get('LONG_WINDOW', 0):
            raise ValueError("SHORT_WINDOW must be less than LONG_WINDOW")
        
        admin_ids = config.get('ADMIN_IDS', [])
        if not isinstance(admin_ids, list):
            raise ValueError("ADMIN_IDS must be a list")
        
        for admin_id in admin_ids:
            if not isinstance(admin_id, (int, str)) or not str(admin_id).isdigit():
                raise ValueError(f"Invalid admin ID: {admin_id}")
    
    def _create_sample_config(self):
        sample_config = {
            "TWELVE_DATA_API_KEY": "YOUR_TWELVE_DATA_API_KEY_HERE",
            "TELEGRAM_TOKEN": "YOUR_TELEGRAM_BOT_TOKEN_HERE", 
            "CHANNEL_ID": "YOUR_TELEGRAM_CHANNEL_ID_HERE",
            "ADMIN_IDS": [],
            "MORNING_TIME": "09:00",
            "TIMEZONE_OFFSET": 0,
            "CHECK_INTERVAL_SECONDS": 300,
            "SHORT_WINDOW": 5,
            "LONG_WINDOW": 20,
            "MAX_SYMBOLS": 20,
            "MAX_NEWS_ITEMS": 5,
            "RATE_LIMIT_DELAY": 1.0
        }
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(sample_config, f, indent=2)
            self.logger.info(f"Created sample configuration file: {self.config_file}")
        except Exception as e:
            self.logger.error(f"Failed to create sample configuration: {e}")
    
    def _display_config_instructions(self):
        print("\n" + "="*60)
        print("📋 CONFIGURATION SETUP REQUIRED")
        print("="*60)
        print(f"Please update {self.config_file} with your API keys:")
        print()
        print("🔑 Required Configuration:")
        print("  1. TELEGRAM_TOKEN")
        print("     - Get from @BotFather on Telegram")
        print("     - Start a chat with @BotFather")
        print("     - Send /newbot and follow instructions")
        print()
        print("  2. CHANNEL_ID")
        print("     - Your Telegram channel/group ID")
        print("     - For channels: @channelname or -100XXXXXXXXX")
        print("     - For groups: -XXXXXXXXX")
        print("     - Use @userinfobot to get chat ID")
        print()
        print("  3. TWELVE_DATA_API_KEY")
        print("     - Get free API key from https://twelvedata.com/")
        print("     - Sign up for free account (800 requests/day)")
        print("     - Find API key in dashboard")
        print()
        print("⚙️  Optional Configuration:")
        print("  - ADMIN_IDS: List of user IDs for admin commands")
        print("  - MORNING_TIME: Daily report time (HH:MM format)")
        print("  - CHECK_INTERVAL_SECONDS: Signal check frequency")
        print()
        print("📝 Example config.json:")
        print("""  {
    "TWELVE_DATA_API_KEY": "abc123def456...",
    "TELEGRAM_TOKEN": "123456789:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
    "CHANNEL_ID": "@mytradingchannel",
    "ADMIN_IDS": [123456789]
  }""")
        print("="*60)

class DataDirectoryManager:
    """Manages data directory and files"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def setup_data_directory(self):
        """Create data directory with all required files"""
        try:
            self.data_dir.mkdir(exist_ok=True)
            self.logger.info(f"Data directory ready: {self.data_dir}")
            
            # Create required files
            self._create_symbols_file()
            self._create_alerts_file()
            self._create_manual_signals_file()
            
        except Exception as e:
            self.logger.error(f"Failed to setup data directory: {e}")
            raise
    
    def _create_symbols_file(self):
        """Create symbols.json if it doesn't exist"""
        symbols_file = self.data_dir / 'symbols.json'
        if not symbols_file.exists():
            sample_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
            try:
                with open(symbols_file, 'w', encoding='utf-8') as f:
                    json.dump(sample_symbols, f, indent=2)
                self.logger.info(f"Created symbols file with sample data: {sample_symbols}")
            except Exception as e:
                self.logger.error(f"Failed to create symbols file: {e}")
    
    def _create_alerts_file(self):
        """Create alerts.json if it doesn't exist"""
        alerts_file = self.data_dir / 'alerts.json'
        if not alerts_file.exists():
            try:
                with open(alerts_file, 'w', encoding='utf-8') as f:
                    json.dump({}, f, indent=2)
                self.logger.info("Created empty alerts file")
            except Exception as e:
                self.logger.error(f"Failed to create alerts file: {e}")
    
    def _create_manual_signals_file(self):
        """Create manual_signals.json if it doesn't exist"""
        signals_file = self.data_dir / 'manual_signals.json'
        if not signals_file.exists():
            try:
                with open(signals_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, indent=2)
                self.logger.info("Created empty manual signals file")
            except Exception as e:
                self.logger.error(f"Failed to create manual signals file: {e}")

class APIConnectionTester:
    """Tests API connections before starting the bot"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def test_twelve_data_connection(self, config: dict) -> bool:
        """Test Twelve Data API connection"""
        try:
            # Import here to avoid circular imports
            from signals import fetch_price
            
            self.logger.info("Testing Twelve Data API connection...")
            
            # Test with a reliable stock
            test_symbol = 'AAPL'
            price = fetch_price(test_symbol)
            
            if price > 0:
                self.logger.info(f"✅ Twelve Data API test successful! {test_symbol} price: ${price:.2f}")
                return True
            else:
                self.logger.error(f"❌ Invalid price returned: ${price}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Twelve Data API test failed: {e}")
            self._display_api_troubleshooting()
            return False
    
    def test_telegram_connection(self, config: dict) -> bool:
        """Test Telegram Bot API connection"""
        try:
            import requests
            
            token = config['TELEGRAM_TOKEN']
            url = f"https://api.telegram.org/bot{token}/getMe"
            
            self.logger.info("Testing Telegram Bot API connection...")
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('ok'):
                bot_info = data.get('result', {})
                bot_name = bot_info.get('first_name', 'Unknown')
                bot_username = bot_info.get('username', 'unknown')
                self.logger.info(f"✅ Telegram API test successful! Bot: {bot_name} (@{bot_username})")
                return True
            else:
                self.logger.error(f"❌ Telegram API error: {data.get('description', 'Unknown error')}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Telegram API connection failed: {e}")
            self._display_telegram_troubleshooting()
            return False
        except Exception as e:
            self.logger.error(f"❌ Telegram API test error: {e}")
            return False
    
    def _display_api_troubleshooting(self):
        """Display Twelve Data API troubleshooting information"""
        print("\n🔧 Twelve Data API Troubleshooting:")
        print("1. Check your API key is correct")
        print("2. Ensure you have internet connection")  
        print("3. Verify you haven't exceeded daily API limit (800 calls/day)")
        print("4. Get your free API key from: https://twelvedata.com/")
        print("5. Check API status at: https://status.twelvedata.com/")
    
    def _display_telegram_troubleshooting(self):
        """Display Telegram API troubleshooting information"""
        print("\n🔧 Telegram API Troubleshooting:")
        print("1. Check your bot token is correct")
        print("2. Ensure bot was created with @BotFather")
        print("3. Verify internet connection")
        print("4. Check Telegram API status")

class SignalHandler:
    """Handles graceful shutdown signals"""
    
    def __init__(self):
        self.shutdown_requested = False
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        if hasattr(signal, 'SIGHUP'):  # Unix only
            signal.signal(signal.SIGHUP, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully"""
        signal_names = {
            signal.SIGINT: 'SIGINT (Ctrl+C)',
            signal.SIGTERM: 'SIGTERM',
        }
        if hasattr(signal, 'SIGHUP'):
            signal_names[signal.SIGHUP] = 'SIGHUP'
        
        signal_name = signal_names.get(signum, f'Signal {signum}')
        self.logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        
        self.shutdown_requested = True
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown was requested"""
        return self.shutdown_requested

class TradePulseApplication:
    """Main application class with comprehensive error handling and monitoring"""
    
    def __init__(self):
        self.logger = None
        self.config = None
        self.bot = None
        self.signal_handler = SignalHandler()
        self.start_time = None
    
    def run(self):
        """Main application entry point"""
        try:
            # Setup logging first
            self.logger = setup_logging()
            self.start_time = datetime.now()
            
            self.logger.info("="*60)
            self.logger.info("🚀 TradePulse Bot Starting Up")
            self.logger.info("="*60)
            
            # Load and validate configuration
            config_manager = ConfigurationManager()
            self.config = config_manager.load_config()
            
            # Setup data directory
            data_manager = DataDirectoryManager()
            data_manager.setup_data_directory()
            
            # Test API connections
            api_tester = APIConnectionTester()
            
            if not api_tester.test_twelve_data_connection(self.config):
                self.logger.error("Cannot start bot without working Twelve Data API connection")
                return 1
            
            if not api_tester.test_telegram_connection(self.config):
                self.logger.error("Cannot start bot without working Telegram API connection")
                return 1
            
            # Initialize bot
            self.logger.info("Initializing TradingBot...")
            self.bot = TradingBot(self.config)
            
            # Display startup information
            self._display_startup_info()
            
            # Start bot (this will block)
            self.logger.info("Starting Telegram bot...")
            return self._run_bot()
            
        except KeyboardInterrupt:
            self.logger.info("Startup interrupted by user")
            return 1
        except SystemExit as e:
            return e.code
        except Exception as e:
            if self.logger:
                self.logger.error(f"Fatal startup error: {e}")
                self.logger.exception("Detailed error information:")
            else:
                print(f"❌ Fatal startup error: {e}")
            return 1
    
    def _run_bot(self):
        """Run the bot with comprehensive error handling"""
        try:
            # Start bot in a way that allows graceful shutdown
            self.bot.start()
            return 0
            
        except KeyboardInterrupt:
            self.logger.info("Bot interrupted by user")
            return 0
        except Exception as e:
            self.logger.error(f"Bot runtime error: {e}")
            self.logger.exception("Detailed error information:")
            return 1
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Cleanup resources on shutdown"""
        try:
            self.logger.info("Starting cleanup process...")
            
            if self.bot:
                self.bot.stop()
                self.logger.info("Bot stopped successfully")
            
            # Calculate uptime
            if self.start_time:
                uptime = datetime.now() - self.start_time
                self.logger.info(f"Total uptime: {uptime}")
            
            self.logger.info("Cleanup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def _display_startup_info(self):
        """Display comprehensive startup information"""
        config = self.config
        
        self.logger.info("📊 Configuration Summary:")
        self.logger.info(f"   - API Provider: Twelve Data (800 calls/day limit)")
        self.logger.info(f"   - Morning report: {config.get('MORNING_TIME', '09:00')} (UTC{int(config.get('TIMEZONE_OFFSET', 0)):+d})")
        self.logger.info(f"   - Admin users: {len(config.get('ADMIN_IDS', []))} configured")
        self.logger.info(f"   - Target channel: {config.get('CHANNEL_ID')}")
        self.logger.info(f"   - Signal check interval: {config.get('CHECK_INTERVAL_SECONDS', 300)}s")
        self.logger.info(f"   - SMA windows: {config.get('SHORT_WINDOW', 5)}/{config.get('LONG_WINDOW', 20)}")
        self.logger.info(f"   - Max symbols: {config.get('MAX_SYMBOLS', 20)}")
        
        self.logger.info("🎯 Available Features:")
        self.logger.info("   - Automated signal generation (SMA crossover)")
        self.logger.info("   - Manual signal creation with R/R calculation")
        self.logger.info("   - Real-time price alerts")
        self.logger.info("   - Daily market summaries")
        self.logger.info("   - News aggregation from multiple sources")
        self.logger.info("   - Signal history tracking")
        
        self.logger.info("🔧 Available Commands:")
        commands = {
            "Market Data": [
                "/check SYMBOL - Get trading signal",
                "/price SYMBOL - Get current price", 
                "/summary SYMBOL - Get detailed summary",
                "/news SYMBOL - Get latest news"
            ],
            "Manual Signals": [
                "/manual - Create manual trading signal",
                "/history [limit] - View signal history", 
                "/cancel - Cancel signal creation"
            ],
            "Management": [
                "/help - Show all commands",
                "/addsymbol SYMBOL - Add to watchlist",
                "/symbols - View watchlist",
                "/setalert SYMBOL % - Set price alert",
                "/morningreport - Send report now"
            ]
        }
        
        for category, cmd_list in commands.items():
            self.logger.info(f"   {category}:")
            for cmd in cmd_list:
                self.logger.info(f"      - {cmd}")
        
        self.logger.info("✅ Bot initialization completed successfully")
        self.logger.info("📱 Bot is now running and ready to receive commands")
        self.logger.info("🛑 Press Ctrl+C to stop the bot gracefully")

def main():
    """Application entry point"""
    app = TradePulseApplication()
    return app.run()

if __name__ == '__main__':
    sys.exit(main())