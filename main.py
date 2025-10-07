import json
import os
import logging
import signal
import sys
import time
from pathlib import Path
from datetime import datetime
import re  # For EmojiFilter

# Import bot module
try:
    from bot import TradingBot
except ImportError as e:
    print(f"❌ ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# -------------------------------
# Emoji Filter (for Windows compatibility)
# -------------------------------
class EmojiFilter(logging.Filter):
    """Filter to remove emojis from log records (for Windows console compatibility)."""
    def filter(self, record):
        record.msg = re.sub(r'[^\x00-\x7F]+', '', str(record.msg))
        return True

# Configure comprehensive logging
def setup_logging():
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    file_handler = logging.FileHandler(logs_dir / 'tradepulse.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    error_handler = logging.FileHandler(logs_dir / 'tradepulse_error.log', encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    console_handler = logging.StreamHandler(sys.stdout)

    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        elif hasattr(console_handler.stream, 'encoding'):
            console_handler.stream.encoding = 'utf-8'
    except (AttributeError, OSError):
        console_handler.addFilter(EmojiFilter())
        print("Note: Console emoji filtering enabled for compatibility")

    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    handlers = [file_handler, error_handler, console_handler]
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)

    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('telegram').setLevel(logging.INFO)

    return logging.getLogger(__name__)

# -------------------------------
# Configuration Manager
# -------------------------------
class ConfigurationManager:
    REQUIRED_KEYS = ['TELEGRAM_TOKEN', 'CHANNEL_ID', 'TWELVE_DATA_API_KEY']
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
        missing_keys = [key for key in self.REQUIRED_KEYS if not config.get(key)]
        if missing_keys:
            self.logger.error(f"Missing required configuration keys: {', '.join(missing_keys)}")
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
        short_window = config.get('SHORT_WINDOW', 5)
        long_window = config.get('LONG_WINDOW', 20)
        if short_window >= long_window:
            raise ValueError("SHORT_WINDOW must be less than LONG_WINDOW")

    def _create_sample_config(self):
        sample_config = {
            "TWELVE_DATA_API_KEY": "YOUR_TWELVE_DATA_API_KEY_HERE",
            "TELEGRAM_TOKEN": "YOUR_TELEGRAM_BOT_TOKEN_HERE",
            "CHANNEL_ID": "YOUR_TELEGRAM_CHANNEL_ID_HERE",
            "ADMIN_IDS": [123456789],
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
        print("\n📋 Please update your config.json with TELEGRAM_TOKEN, CHANNEL_ID, TWELVE_DATA_API_KEY\n")

# -------------------------------
# Data Directory Manager
# -------------------------------
class DataDirectoryManager:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(self.__class__.__name__)

    def setup_data_directory(self):
        self.data_dir.mkdir(exist_ok=True)
        self._create_symbols_file()
        self._create_alerts_file()
        self._create_custom_signals_file()

    def _create_symbols_file(self):
        symbols_file = self.data_dir / 'symbols.json'
        if not symbols_file.exists():
            sample_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
            with open(symbols_file, 'w', encoding='utf-8') as f:
                json.dump(sample_symbols, f, indent=2)

    def _create_alerts_file(self):
        alerts_file = self.data_dir / 'alerts.json'
        if not alerts_file.exists():
            with open(alerts_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=2)

    def _create_custom_signals_file(self):
        signals_file = self.data_dir / 'custom_signals.json'
        if not signals_file.exists():
            with open(signals_file, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)

# -------------------------------
# Signal Handler
# -------------------------------
class SignalHandler:
    def __init__(self):
        self.shutdown_requested = False
        self.logger = logging.getLogger(self.__class__.__name__)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        self.logger.info(f"Received shutdown signal ({signum}), stopping...")
        self.shutdown_requested = True

    def is_shutdown_requested(self):
        return self.shutdown_requested

# -------------------------------
# Main Bot Application
# -------------------------------
class TradePulseApplication:
    def __init__(self):
        self.logger = None
        self.config = None
        self.bot = None
        self.signal_handler = SignalHandler()
        self.start_time = None

    def run(self):
        try:
            self.logger = setup_logging()
            self.start_time = datetime.now()
            self.logger.info("🚀 TradePulse Bot Starting Up")

            config_manager = ConfigurationManager()
            self.config = config_manager.load_config()

            data_manager = DataDirectoryManager()
            data_manager.setup_data_directory()

            self.logger.info("Initializing TradingBot...")
            self.bot = TradingBot(self.config)

            self._display_startup_info()
            self.logger.info("Starting Telegram bot (long-polling)...")
            return self._run_bot()

        except KeyboardInterrupt:
            self.logger.info("Startup interrupted by user")
            return 1
        except Exception as e:
            if self.logger:
                self.logger.error(f"Fatal startup error: {e}")
                self.logger.exception("Detailed error info:")
            else:
                print(f"❌ Fatal startup error: {e}")
            return 1

    def _run_bot(self):
        try:
            self.bot.start()
            return 0
        except Exception as e:
            self.logger.error(f"Bot runtime error: {e}")
            self.logger.exception("Detailed error info:")
            return 1
        finally:
            self._cleanup()

    def _cleanup(self):
        try:
            self.logger.info("Cleaning up...")
            if self.bot:
                self.bot.stop()
            if self.start_time:
                uptime = datetime.now() - self.start_time
                self.logger.info(f"Total uptime: {uptime}")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def _display_startup_info(self):
        self.logger.info(f"📊 Channel: {self.config.get('CHANNEL_ID')}")
        self.logger.info(f"📱 Admin users: {len(self.config.get('ADMIN_IDS', []))}")
        self.logger.info("✅ Bot initialized successfully")

# -------------------------------
# Webhook Mode for Render
# -------------------------------
def run_webhook():
    logger = setup_logging()  # Enable comprehensive logging for webhook mode
    from flask import Flask, request
    import requests  # For setting webhook

    # Get required environment variables
    TOKEN = os.environ.get("TELEGRAM_TOKEN")
    CHANNEL_ID = os.environ.get("CHANNEL_ID")
    TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY")

    # Validate
    missing = [name for name in ["TELEGRAM_TOKEN", "CHANNEL_ID", "TWELVE_DATA_API_KEY"]
               if not os.environ.get(name)]
    if missing:
        logger.error(f"❌ Missing required environment variables: {', '.join(missing)}")
        raise SystemExit(f"❌ Missing required environment variables: {', '.join(missing)}")

    app = Flask(__name__)

    config = {
        "TELEGRAM_TOKEN": TOKEN,
        "CHANNEL_ID": CHANNEL_ID,
        "TWELVE_DATA_API_KEY": TWELVE_DATA_API_KEY
    }

    # Include optional settings if needed
    config["ADMIN_IDS"] = [int(id.strip()) for id in os.environ.get("ADMIN_IDS", "").split(",") if id.strip()] if os.environ.get("ADMIN_IDS") else []
    config["CHECK_INTERVAL_SECONDS"] = int(os.environ.get("CHECK_INTERVAL_SECONDS", 300))

    logger.info("Initializing TradingBot for webhook mode...")
    bot = TradingBot(config)

    @app.route(f"/{TOKEN}", methods=["POST"])
    def webhook():
        update = request.get_json(force=True)
        try:
            logger.info(f"Received update: {update}")  # Debug log
            bot.handle_update(update)
        except Exception as e:
            logger.error(f"Error handling update: {e}")
        return "OK"

    @app.route("/", methods=["GET"])
    def index():
        return "Trading Bot is running!"

    # Set webhook automatically if running on Render
    RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL")
    if RENDER_URL:
        webhook_url = f"{RENDER_URL}/{TOKEN}"
        api_url = f"https://api.telegram.org/bot{TOKEN}/setWebhook"
        payload = {"url": webhook_url}
        try:
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                if result.get("ok"):
                    logger.info(f"✅ Webhook set successfully: {webhook_url}")
                else:
                    logger.error(f"❌ Webhook set failed: {result.get('description', 'Unknown error')}")
            else:
                logger.error(f"❌ HTTP {response.status_code} when setting webhook: {response.text}")
        except Exception as e:
            logger.error(f"❌ Error setting webhook: {e}")

    PORT = int(os.environ.get("PORT", 8443))
    logger.info(f"Starting Flask app on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)


# -------------------------------
# Entry Point
# -------------------------------
def main():
    app = TradePulseApplication()
    return app.run()

if __name__ == "__main__":
    # Check for webhook/production environment
    is_production = (
        os.environ.get("RENDER") or 
        os.environ.get("PORT") or 
        os.environ.get("PRODUCTION")
    )
    
    if is_production:
        run_webhook()
    else:
        sys.exit(main())