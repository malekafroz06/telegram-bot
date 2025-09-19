import json
import os
import logging
import asyncio
import threading
import signal
from datetime import datetime, time
from pathlib import Path
from typing import Optional, Dict, Any

from telegram import Update, Bot
from telegram.ext import (
    ApplicationBuilder, 
    CommandHandler, 
    MessageHandler, 
    ContextTypes, 
    filters
)
from telegram.error import TelegramError, NetworkError, TimedOut

from signals import (
    generate_signal, 
    fetch_price, 
    fetch_news, 
    fetch_summary, 
    fetch_company_overview,
    SignalChecker
)

# Configure logger for this module
logger = logging.getLogger(__name__)

class TradingBotError(Exception):
    """Custom exception for bot-related errors"""
    pass

class ConfigurationError(TradingBotError):
    """Raised when there are configuration issues"""
    pass

class MessageSender:
    """Handles message sending with retry logic and error handling"""
    
    def __init__(self, bot: Bot, config: Dict[str, Any]):
        self.bot = bot
        self.config = config
        self.channel_id = config.get('CHANNEL_ID')
        self.max_retries = 3
        self.retry_delay = 2.0
        
    async def send_message_with_retry(self, chat_id: str, text: str, **kwargs) -> bool:
        """Send message with retry logic and comprehensive error handling"""
        for attempt in range(self.max_retries):
            try:
                await self.bot.send_message(chat_id=chat_id, text=text, **kwargs)
                return True
                
            except TimedOut:
                logger.warning(f"Telegram timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                    
            except NetworkError as e:
                logger.warning(f"Network error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                    
            except TelegramError as e:
                if "chat not found" in str(e).lower():
                    logger.error(f"Chat not found: {chat_id}. Check CHANNEL_ID configuration.")
                    return False
                elif "bot was blocked" in str(e).lower():
                    logger.error(f"Bot was blocked by user/chat: {chat_id}")
                    return False
                else:
                    logger.error(f"Telegram API error: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        continue
                        
            except Exception as e:
                logger.error(f"Unexpected error sending message: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                    
        logger.error(f"Failed to send message after {self.max_retries} attempts")
        return False

class TimerManager:
    """Manages the morning report timer with enhanced reliability"""
    
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.config = bot_instance.config
        self.morning_time = self.config.get('MORNING_TIME', '09:00')
        self.timezone_offset = self.config.get('TIMEZONE_OFFSET', 0)
        self.timer_thread: Optional[threading.Thread] = None
        self.timer_running = False
        self.test_mode = False
        self.main_loop: Optional[asyncio.AbstractEventLoop] = None
        
    def start_timer(self, event_loop: asyncio.AbstractEventLoop):
        """Start the morning timer thread"""
        if self.timer_running:
            logger.warning("Timer already running")
            return
            
        self.main_loop = event_loop
        self.timer_running = True
        self.timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self.timer_thread.start()
        logger.info(f"Morning timer started for {self.morning_time}")
    
    def stop_timer(self):
        """Stop the morning timer"""
        self.timer_running = False
        if self.timer_thread and self.timer_thread.is_alive():
            self.timer_thread.join(timeout=5.0)
            if self.timer_thread.is_alive():
                logger.warning("Timer thread did not stop gracefully")
        logger.info("Morning timer stopped")
    
    def set_morning_time(self, time_str: str) -> bool:
        """Set new morning time with validation"""
        try:
            parsed_time = self._parse_time(time_str)
            self.morning_time = time_str
            logger.info(f"Morning time updated to {time_str}")
            return True
        except ValueError as e:
            logger.error(f"Invalid time format: {e}")
            return False
    
    def toggle_test_mode(self) -> bool:
        """Toggle test mode (reports every 2 minutes)"""
        self.test_mode = not self.test_mode
        mode_text = "enabled" if self.test_mode else "disabled"
        logger.info(f"Test mode {mode_text}")
        return self.test_mode
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive timer status"""
        return {
            'running': self.timer_running,
            'test_mode': self.test_mode,
            'morning_time': self.morning_time,
            'timezone_offset': self.timezone_offset,
            'thread_alive': self.timer_thread.is_alive() if self.timer_thread else False
        }
    
    def _timer_loop(self):
        """Main timer loop with enhanced error handling"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.timer_running:
            try:
                current_time = datetime.now()
                
                if self.test_mode:
                    # Test mode: send report every 2 minutes
                    sleep_seconds = 120
                else:
                    # Normal mode: calculate sleep time until morning report
                    target_time = self._parse_time(self.morning_time)
                    next_trigger = current_time.replace(
                        hour=target_time.hour,
                        minute=target_time.minute,
                        second=0,
                        microsecond=0
                    )
                    
                    # If past today's time, schedule for tomorrow
                    if current_time >= next_trigger:
                        next_trigger = next_trigger.replace(day=next_trigger.day + 1)
                    
                    sleep_seconds = (next_trigger - current_time).total_seconds()
                
                # Sleep with interruption checking
                if sleep_seconds > 0:
                    sleep_interval = min(sleep_seconds, 60)  # Check every minute
                    end_time = current_time.timestamp() + sleep_seconds
                    
                    while datetime.now().timestamp() < end_time and self.timer_running:
                        threading.Event().wait(min(sleep_interval, end_time - datetime.now().timestamp()))
                
                # Send report if it's time
                if self.timer_running and self._should_send_report():
                    await_result = self._schedule_morning_report()
                    if await_result:
                        consecutive_errors = 0
                    else:
                        consecutive_errors += 1
                
                # Prevent immediate re-triggering
                if self.timer_running:
                    threading.Event().wait(120)  # Wait 2 minutes
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error in timer loop (error #{consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(f"Too many consecutive timer errors ({consecutive_errors}), stopping timer")
                    self.timer_running = False
                    break
                
                # Exponential backoff for errors
                error_sleep = min(60 * (2 ** consecutive_errors), 300)  # Max 5 minutes
                threading.Event().wait(error_sleep)
    
    def _should_send_report(self) -> bool:
        """Determine if it's time to send the morning report"""
        if self.test_mode:
            return True
        
        current_time = datetime.now()
        target_time = self._parse_time(self.morning_time)
        target_time_today = current_time.replace(
            hour=target_time.hour,
            minute=target_time.minute,
            second=0,
            microsecond=0
        )
        
        # Check if within 2 minutes of target time
        return abs((current_time - target_time_today).total_seconds()) < 120
    
    def _schedule_morning_report(self) -> bool:
        """Schedule morning report in the main event loop"""
        if not self.main_loop or self.main_loop.is_closed():
            logger.error("Main event loop not available")
            return False
        
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.bot.send_morning_report(),
                self.main_loop
            )
            future.result(timeout=60)  # Wait up to 60 seconds
            return True
        except Exception as e:
            logger.error(f"Failed to schedule morning report: {e}")
            return False
    
    def _parse_time(self, time_str: str) -> time:
        """Parse time string with comprehensive validation"""
        if not isinstance(time_str, str):
            raise ValueError("Time must be a string")
        
        time_str = time_str.strip()
        if ':' not in time_str:
            raise ValueError("Time must be in HH:MM format")
        
        try:
            parts = time_str.split(':')
            if len(parts) != 2:
                raise ValueError("Time must be in HH:MM format")
            
            hour, minute = int(parts[0]), int(parts[1])
            
            if not (0 <= hour <= 23):
                raise ValueError("Hour must be between 0 and 23")
            if not (0 <= minute <= 59):
                raise ValueError("Minute must be between 0 and 59")
            
            return time(hour, minute)
            
        except (ValueError, TypeError) as e:
            if "invalid literal" in str(e):
                raise ValueError("Time must contain valid numbers")
            raise e

class FileManager:
    """Handles file operations with atomic writes and error recovery"""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def load_json_file(self, filename: str, default=None):
        """Load JSON file with error handling"""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.debug(f"File {filename} does not exist, returning default")
            return default if default is not None else []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    logger.warning(f"File {filename} is empty")
                    return default if default is not None else []
                
                return json.loads(content)
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filename}: {e}")
            # Try to create backup
            backup_path = filepath.with_suffix('.json.backup')
            try:
                filepath.rename(backup_path)
                logger.info(f"Corrupted file backed up to {backup_path}")
            except Exception:
                pass
            return default if default is not None else []
            
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")
            return default if default is not None else []
    
    def save_json_file(self, filename: str, data) -> bool:
        """Save JSON file with atomic write"""
        filepath = self.data_dir / filename
        temp_filepath = filepath.with_suffix('.json.tmp')
        
        try:
            # Write to temporary file first
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic move
            if os.name == 'nt':  # Windows
                if filepath.exists():
                    filepath.unlink()
                temp_filepath.rename(filepath)
            else:  # Unix-like systems
                temp_filepath.rename(filepath)
            
            logger.debug(f"Successfully saved {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving {filename}: {e}")
            # Cleanup temp file
            if temp_filepath.exists():
                try:
                    temp_filepath.unlink()
                except:
                    pass
            return False

class TradingBot:
    """Production-ready trading bot with comprehensive error handling and monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = self._validate_config(config)
        
        # Initialize components
        self.file_manager = FileManager()
        self.app = None
        self.message_sender = None
        self.timer_manager = None
        self.signal_checker = None
        self.main_loop = None
        
        # Initialize Telegram application
        self._init_telegram_app()
        
        # Initialize other components
        self.message_sender = MessageSender(self.app.bot, self.config)
        self.timer_manager = TimerManager(self)
        self.signal_checker = SignalChecker(self.config, self)
        
        # Register command handlers
        self._register_handlers()
        
        # Setup graceful shutdown
        self._setup_shutdown_handlers()
        
        logger.info("TradingBot initialized successfully")
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration with comprehensive checks"""
        if not isinstance(config, dict):
            raise ConfigurationError("Configuration must be a dictionary")
        
        # Check required keys
        required_keys = ['TELEGRAM_TOKEN', 'CHANNEL_ID', 'TWELVE_DATA_API_KEY']
        missing_keys = [key for key in required_keys if not config.get(key)]
        
        if missing_keys:
            raise ConfigurationError(f"Missing required configuration: {', '.join(missing_keys)}")
        
        # Validate token format
        token = config['TELEGRAM_TOKEN']
        if not isinstance(token, str) or ':' not in token or len(token) < 20:
            raise ConfigurationError("Invalid TELEGRAM_TOKEN format")
        
        # Set defaults for optional values
        defaults = {
            'ADMIN_IDS': [],
            'MORNING_TIME': '09:00',
            'TIMEZONE_OFFSET': 0,
            'CHECK_INTERVAL_SECONDS': 300,
            'SHORT_WINDOW': 5,
            'LONG_WINDOW': 20,
            'MAX_RETRIES': 3,
            'MESSAGE_TIMEOUT': 10
        }
        
        for key, default_value in defaults.items():
            config.setdefault(key, default_value)
        
        return config
    
    def _init_telegram_app(self):
        """Initialize Telegram application with proper configuration"""
        try:
            self.app = ApplicationBuilder() \
                .token(self.config['TELEGRAM_TOKEN']) \
                .connect_timeout(30) \
                .read_timeout(30) \
                .write_timeout(30) \
                .build()
            
            logger.info("Telegram application initialized")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Telegram application: {e}")
    
    def _register_handlers(self):
        """Register all command and message handlers"""
        handlers = [
            # Basic commands
            ('start', self.start_cmd),
            ('help', self.help_cmd),
            
            # Symbol management
            ('addsymbol', self.addsymbol_cmd),
            ('removesymbol', self.removesymbol_cmd),
            ('symbols', self.listsymbols_cmd),
            ('listsymbols', self.listsymbols_cmd),
            
            # Alerts
            ('setalert', self.setalert_cmd),
            
            # Market data
            ('news', self.news_cmd),
            ('summary', self.summary_cmd),
            ('check', self.check_cmd),
            ('price', self.price_cmd),
            
            # Manual signals
            ('manual', self.manual_cmd),
            ('cancel', self.cancel_manual_cmd),
            ('history', self.history_cmd),
            
            # Timer commands
            ('morningreport', self.morningreport_cmd),
            ('setmorningtime', self.setmorningtime_cmd),
            ('testtimer', self.testtimer_cmd),
            ('timerstatus', self.timerstatus_cmd),
        ]
        
        # Register command handlers
        for command, handler in handlers:
            self.app.add_handler(CommandHandler(command, handler))
        
        # Register message handler for manual signal input
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_input)
        )
        
        logger.info(f"Registered {len(handlers)} command handlers")
    
    def _setup_shutdown_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start(self):
        """Start the bot with comprehensive initialization"""
        try:
            logger.info("Starting TradingBot...")
            
            # Store main event loop
            self.main_loop = asyncio.get_event_loop()
            
            # Start timer
            self.timer_manager.start_timer(self.main_loop)
            
            # Start polling
            logger.info("Starting Telegram polling...")
            self.app.run_polling(
                poll_interval=1.0,
                timeout=30,
                bootstrap_retries=5,
                allowed_updates=None,
                drop_pending_updates=True
            )
            
        except KeyboardInterrupt:
            logger.info("Bot interrupted by user")
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            raise
        finally:
            self.stop()
    
    def stop(self):
        """Stop all bot components gracefully"""
        logger.info("Stopping TradingBot...")
        
        try:
            # Stop timer
            if self.timer_manager:
                self.timer_manager.stop_timer()
            
            # Stop signal checker
            if self.signal_checker and hasattr(self.signal_checker, 'stop'):
                self.signal_checker.stop()
            
            # Stop Telegram application
            if self.app:
                self.app.stop_running()
            
            logger.info("TradingBot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def _is_admin(self, user_id: int) -> bool:
        """Check if user is an admin with type safety"""
        try:
            admin_ids = self.config.get('ADMIN_IDS', [])
            return int(user_id) in [int(admin_id) for admin_id in admin_ids]
        except (ValueError, TypeError):
            return False
    
    async def post_message(self, text: str, **kwargs) -> bool:
        """Post message to configured channel with error handling"""
        channel_id = self.config.get('CHANNEL_ID')
        if not channel_id:
            logger.warning('CHANNEL_ID not configured, cannot post message')
            return False
        
        return await self.message_sender.send_message_with_retry(
            chat_id=channel_id,
            text=text,
            parse_mode='HTML',
            **kwargs
        )
    
    def post_message_from_thread(self, text: str) -> bool:
        """Post message from another thread safely"""
        if not self.main_loop or self.main_loop.is_closed():
            logger.error("Main event loop not available for cross-thread communication")
            return False
        
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.post_message(text),
                self.main_loop
            )
            return future.result(timeout=self.config.get('MESSAGE_TIMEOUT', 10))
        except Exception as e:
            logger.error(f"Failed to post message from thread: {e}")
            return False
    
    async def send_morning_report(self):
        """Generate and send comprehensive morning report"""
        try:
            logger.info("Generating morning report...")
            
            symbols = self.file_manager.load_json_file('symbols.json', [])
            if not symbols:
                logger.info("No symbols in watchlist for morning report")
                return
            
            # Create report header
            current_date = datetime.now().strftime('%B %d, %Y')
            report_text = f"🌅 <b>Good Morning! Daily Market Report</b>\n📅 {current_date}\n\n"
            
            # Process symbols (limit to 5 for rate limiting)
            processed_count = 0
            for symbol in symbols[:5]:
                try:
                    # Rate limiting delay
                    if processed_count > 0:
                        await asyncio.sleep(1.2)  # Slightly more than 1 second
                    
                    # Get signal and summary data
                    signal_data = generate_signal(symbol)
                    summary_data = fetch_summary(symbol)
                    
                    # Format symbol report
                    change_emoji = "🔺" if summary_data['price_change'] > 0 else "🔻" if summary_data['price_change'] < 0 else "➡️"
                    signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(signal_data['signal'], "🟡")
                    
                    report_text += (
                        f"{signal_emoji} <b>{symbol}</b> — {signal_data['signal']}\n"
                        f"💰 ${summary_data['current_price']:.2f} {change_emoji} {summary_data['price_change_pct']:+.2f}%\n"
                        f"📊 Vol: {summary_data['volume']:,}\n\n"
                    )
                    
                    processed_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {e}")
                    report_text += f"❌ <b>{symbol}</b> — Data unavailable\n\n"
            
            report_text += "📈 <i>Have a profitable trading day!</i>\n"
            report_text += "📡 <i>Data from Twelve Data API</i>"
            
            # Send report
            success = await self.post_message(report_text)
            if success:
                logger.info("Morning report sent successfully")
            else:
                logger.error("Failed to send morning report")
            
        except Exception as e:
            logger.error(f"Error generating morning report: {e}")
    
    # Command handlers with enhanced error handling
    
    async def handle_text_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text input during manual signal process"""
        try:
            user = update.effective_user
            if not self._is_admin(user.id):
                return
            
            user_id = str(user.id)
            message_text = update.message.text.strip()
            
            # Check for active manual signal process
            if (hasattr(self.signal_checker, 'manual_handler') and 
                hasattr(self.signal_checker.manual_handler, 'pending_signals') and
                user_id in self.signal_checker.manual_handler.pending_signals):
                
                # Process input
                response = self.signal_checker.handle_manual_input(user_id, message_text)
                await update.message.reply_text(response, parse_mode='HTML')
                
                # If signal completed, also send to channel
                if "MANUAL SIGNAL CREATED" in response:
                    await self.post_message(response)
                    
        except Exception as e:
            logger.error(f"Error handling text input: {e}")
            await update.message.reply_text("❌ An error occurred processing your message.")
    
    async def start_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced start command with comprehensive information"""
        welcome_text = """👋 <b>Welcome to TradePulse Bot!</b>
Your AI-powered trading assistant for signals and market analysis.

🚀 <b>Quick Start:</b>
• <code>/help</code> - See all commands
• <code>/manual</code> - Create trading signal
• <code>/check AAPL</code> - Analyze a stock
• <code>/addsymbol TSLA</code> - Add to watchlist

💡 <b>Key Features:</b>
✅ Manual signal creation with R/R ratios
✅ Signal history tracking
✅ Enhanced market analysis
✅ Automated morning reports
✅ Price alerts and notifications

📊 <b>Data Sources:</b>
• Real-time prices from Twelve Data API
• Technical analysis with SMA crossovers
• Multi-source news aggregation
• Company fundamentals and metrics

<i>Ready to start trading smarter! 🎯</i>

Use /help for complete command reference."""
        
        await update.message.reply_text(welcome_text, parse_mode='HTML')
    
    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced help command with categorized commands"""
        help_text = """🤖 <b>TradePulse Bot - Command Reference</b>

<b>📊 Trading Signals:</b>
• <code>/check SYMBOL</code> - Get trading signal analysis
• <code>/manual</code> - Create custom trading signal
• <code>/cancel</code> - Cancel manual signal process
• <code>/history [limit]</code> - View recent manual signals

<b>📈 Market Data:</b>
• <code>/price SYMBOL</code> - Get current stock price
• <code>/summary SYMBOL</code> - Detailed stock summary
• <code>/news SYMBOL</code> - Latest news and updates

<b>⚙️ Watchlist Management:</b>
• <code>/addsymbol SYMBOL</code> - Add stock to watchlist
• <code>/removesymbol SYMBOL</code> - Remove from watchlist  
• <code>/symbols</code> - View current watchlist
• <code>/setalert SYMBOL %</code> - Set price change alert

<b>⏰ Reports & Automation:</b>
• <code>/morningreport</code> - Generate daily report now
• <code>/setmorningtime HH:MM</code> - Set report schedule
• <code>/testtimer</code> - Toggle test mode (2-min reports)
• <code>/timerstatus</code> - Check timer status

<b>📋 Manual Signal Workflow:</b>
1. <code>/manual</code> → Start creation process
2. Enter symbol (e.g., AAPL)
3. Choose BUY/SELL direction
4. Set timeframe (1H, 4H, 1D, 1W)
5. Define risk percentage (1-10%)

Bot calculates stop-loss and target automatically with 1:2 R/R ratio.

<b>💡 Pro Tips:</b>
• All signals are saved for tracking
• Morning reports sent to configured channel
• Price alerts trigger on threshold breach
• Use test mode to verify bot functionality

<i>📡 Powered by Twelve Data API</i>"""
        
        await update.message.reply_text(help_text, parse_mode='HTML')
    
    async def manual_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start manual signal creation with validation"""
        try:
            user = update.effective_user
            if not self._is_admin(user.id):
                await update.message.reply_text('❌ Unauthorized access.')
                return
            
            user_id = str(user.id)
            response = self.signal_checker.handle_manual_command(user_id)
            await update.message.reply_text(response, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Error in manual command: {e}")
            await update.message.reply_text('❌ Error starting manual signal process.')
    
    async def cancel_manual_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Cancel manual signal creation"""
        try:
            user = update.effective_user
            if not self._is_admin(user.id):
                await update.message.reply_text('❌ Unauthorized access.')
                return
            
            user_id = str(user.id)
            response = self.signal_checker.handle_cancel_manual(user_id)
            await update.message.reply_text(response)
            
        except Exception as e:
            logger.error(f"Error in cancel command: {e}")
            await update.message.reply_text('❌ Error cancelling signal process.')
    
    async def history_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show manual signals history with pagination"""
        try:
            user = update.effective_user
            if not self._is_admin(user.id):
                await update.message.reply_text('❌ Unauthorized access.')
                return
            
            # Parse limit argument
            limit = 10
            if context.args:
                try:
                    limit = min(max(int(context.args[0]), 1), 50)  # Between 1 and 50
                except ValueError:
                    await update.message.reply_text('❌ Invalid limit. Use a number between 1-50.')
                    return
            
            response = self.signal_checker.get_manual_signals_history(limit)
            await update.message.reply_text(response, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Error in history command: {e}")
            await update.message.reply_text('❌ Error retrieving signal history.')
    
    async def price_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get current stock price with enhanced formatting"""
        try:
            if not context.args:
                await update.message.reply_text('Usage: /price SYMBOL\nExample: /price AAPL')
                return
            
            symbol = context.args[0].upper().strip()
            if not symbol.isalpha() or len(symbol) > 10:
                await update.message.reply_text('❌ Invalid symbol format. Use stock symbols like AAPL, GOOGL.')
                return
            
            loading_msg = await update.message.reply_text(f"💰 Fetching current price for {symbol}...")
            
            # Get current price
            current_price = fetch_price(symbol)
            
            # Get company info for better display
            try:
                overview = fetch_company_overview(symbol)
                company_name = overview.get('name', symbol)
            except Exception:
                company_name = symbol
            
            text = (f"💰 <b>{symbol}</b>\n"
                   f"<i>{company_name}</i>\n\n"
                   f"<b>${current_price:.2f}</b>\n\n"
                   f"📡 <i>Data from Twelve Data API</i>")
            
            # Delete loading message and send result
            await loading_msg.delete()
            await update.message.reply_text(text, parse_mode="HTML")
            
            # Also send to channel
            await self.post_message(text)
            
        except Exception as e:
            error_text = f'❌ Failed to fetch price for {symbol}: {str(e)}'
            logger.error(f"Price command error: {e}")
            try:
                await loading_msg.edit_text(error_text)
            except:
                await update.message.reply_text(error_text)
    
    async def check_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get trading signal analysis with comprehensive error handling"""
        try:
            user = update.effective_user
            if not self._is_admin(user.id):
                await update.message.reply_text('❌ Unauthorized access.')
                return
            
            if not context.args:
                await update.message.reply_text('Usage: /check SYMBOL\nExample: /check AAPL')
                return
            
            symbol = context.args[0].upper().strip()
            if not symbol.isalpha() or len(symbol) > 10:
                await update.message.reply_text('❌ Invalid symbol format.')
                return
            
            loading_msg = await update.message.reply_text(f"🔍 Analyzing {symbol}...")
            
            # Generate signal
            signal_result = generate_signal(symbol)
            
            # Format message with enhanced styling
            signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(signal_result['signal'], "🟡")
            
            text = (f"{signal_emoji} <b>{signal_result['symbol']}</b> — <b>{signal_result['signal']}</b>\n"
                   f"💰 Price: ${signal_result['price']:.2f}\n"
                   f"📈 Short SMA(5): ${signal_result['short_sma']:.2f}\n"
                   f"📊 Long SMA(20): ${signal_result['long_sma']:.2f}\n\n"
                   f"📡 <i>Data from Twelve Data API</i>")
            
            # Delete loading message and reply
            await loading_msg.delete()
            await update.message.reply_text(text, parse_mode="HTML")
            
            # Send to channel
            await self.post_message(text)
            
        except Exception as e:
            error_text = f"❌ Error analyzing {symbol}: {str(e)}"
            logger.error(f"Check command error: {e}")
            try:
                await loading_msg.edit_text(error_text)
            except:
                await update.message.reply_text(error_text)
    
    async def summary_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get comprehensive stock summary"""
        try:
            if not context.args:
                await update.message.reply_text('Usage: /summary SYMBOL\nExample: /summary AAPL')
                return
            
            symbol = context.args[0].upper().strip()
            if not symbol.isalpha() or len(symbol) > 10:
                await update.message.reply_text('❌ Invalid symbol format.')
                return
            
            loading_msg = await update.message.reply_text(f"📊 Generating summary for {symbol}...")
            
            # Get summary data
            summary_data = fetch_summary(symbol)
            
            # Format comprehensive summary
            change_emoji = "🔺" if summary_data['price_change'] > 0 else "🔻" if summary_data['price_change'] < 0 else "➡️"
            
            text = f"""📊 <b>Daily Summary for {symbol}</b>
<b>{summary_data['company_name']}</b>

💰 <b>Price:</b> ${summary_data['current_price']:.2f} {change_emoji}
📈 <b>Change:</b> ${summary_data['price_change']:+.2f} ({summary_data['price_change_pct']:+.2f}%)

📊 <b>Day Range:</b> ${summary_data['day_low']:.2f} - ${summary_data['day_high']:.2f}
📦 <b>Volume:</b> {summary_data['volume']:,} (Avg: {summary_data['avg_volume']:,})

🏢 <b>Market Cap:</b> {summary_data['market_cap']}
📋 <b>P/E Ratio:</b> {summary_data['pe_ratio']}

📡 <i>Data from Twelve Data API</i>"""
            
            # Delete loading message and send result
            await loading_msg.delete()
            await update.message.reply_text(text, parse_mode="HTML")
            
            # Send to channel
            await self.post_message(text)
            
        except Exception as e:
            error_text = f'❌ Failed to fetch summary for {symbol}: {str(e)}'
            logger.error(f"Summary command error: {e}")
            try:
                await loading_msg.edit_text(error_text)
            except:
                await update.message.reply_text(error_text)
    
    async def news_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get latest news with multiple source fallback"""
        try:
            if not context.args:
                await update.message.reply_text('Usage: /news SYMBOL\nExample: /news AAPL')
                return
            
            symbol = context.args[0].upper().strip()
            if not symbol.isalpha() or len(symbol) > 10:
                await update.message.reply_text('❌ Invalid symbol format.')
                return
            
            loading_msg = await update.message.reply_text(f"🔍 Fetching latest news for {symbol}...")
            
            # Get news items
            news_items = fetch_news(symbol)
            
            if not news_items:
                text = f"📰 <b>No recent news found for {symbol}</b>\n\nTry checking financial news websites directly or ensure the symbol is correct."
            else:
                # Get company name
                try:
                    overview = fetch_company_overview(symbol)
                    company_name = overview.get('name', symbol)
                except:
                    company_name = symbol
                
                text = f"📰 <b>Latest news for {symbol}</b>\n<i>{company_name}</i>\n\n"
                
                # Add news items with source indicators
                for i, item in enumerate(news_items[:4], 1):
                    source_emoji = {
                        'twelve_data': '🔥',
                        'yahoo_rss': '📡',
                        'fallback_yahoo': '💼',
                        'fallback_quote': '📊',
                        'fallback_marketwatch': '📈'
                    }.get(item.get('source', ''), '📰')
                    
                    text += f"{source_emoji} <b>{item['title']}</b>\n{item['link']}\n\n"
                
                # Add source attribution
                text += "📡 <i>News aggregated from multiple sources</i>"
            
            # Delete loading message and send result
            await loading_msg.delete()
            await update.message.reply_text(text, parse_mode="HTML", disable_web_page_preview=True)
            
            # Send to channel
            await self.post_message(text, disable_web_page_preview=True)
            
        except Exception as e:
            error_text = f'❌ Failed to fetch news for {symbol}: {str(e)}'
            logger.error(f"News command error: {e}")
            try:
                await loading_msg.edit_text(error_text)
            except:
                await update.message.reply_text(error_text)
    
    async def addsymbol_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Add symbol to watchlist with validation"""
        try:
            user = update.effective_user
            if not self._is_admin(user.id):
                await update.message.reply_text('❌ Unauthorized access.')
                return
            
            if not context.args:
                await update.message.reply_text('Usage: /addsymbol SYMBOL\nExample: /addsymbol AAPL')
                return
            
            symbol = context.args[0].upper().strip()
            if not symbol.isalpha() or len(symbol) > 10:
                await update.message.reply_text('❌ Invalid symbol format.')
                return
            
            # Load current symbols
            symbols = self.file_manager.load_json_file('symbols.json', [])
            
            if symbol in symbols:
                await update.message.reply_text(f'📊 {symbol} is already in the watchlist.')
                return
            
            # Validate symbol by fetching price
            validation_msg = await update.message.reply_text(f'🔍 Validating {symbol}...')
            
            try:
                current_price = fetch_price(symbol)
                await validation_msg.edit_text(f'✅ {symbol} is valid (${current_price:.2f})')
            except Exception as e:
                await validation_msg.edit_text(f'❌ Invalid symbol {symbol}: {str(e)}')
                return
            
            # Add to symbols list
            symbols.append(symbol)
            
            # Save with atomic write
            if self.file_manager.save_json_file('symbols.json', symbols):
                await update.message.reply_text(f'✅ Added {symbol} to watchlist ({len(symbols)} symbols total).')
                logger.info(f"Added {symbol} to watchlist")
            else:
                await update.message.reply_text('❌ Failed to save symbol to watchlist.')
            
        except Exception as e:
            logger.error(f"Add symbol command error: {e}")
            await update.message.reply_text('❌ Error adding symbol to watchlist.')
    
    async def removesymbol_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Remove symbol from watchlist"""
        try:
            user = update.effective_user
            if not self._is_admin(user.id):
                await update.message.reply_text('❌ Unauthorized access.')
                return
            
            if not context.args:
                await update.message.reply_text('Usage: /removesymbol SYMBOL\nExample: /removesymbol AAPL')
                return
            
            symbol = context.args[0].upper().strip()
            
            # Load current symbols
            symbols = self.file_manager.load_json_file('symbols.json', [])
            
            if symbol not in symbols:
                await update.message.reply_text(f'📊 {symbol} is not in the watchlist.')
                return
            
            # Remove symbol
            symbols.remove(symbol)
            
            # Save updated list
            if self.file_manager.save_json_file('symbols.json', symbols):
                await update.message.reply_text(f'✅ Removed {symbol} from watchlist ({len(symbols)} symbols remaining).')
                logger.info(f"Removed {symbol} from watchlist")
            else:
                await update.message.reply_text('❌ Failed to update watchlist.')
            
        except Exception as e:
            logger.error(f"Remove symbol command error: {e}")
            await update.message.reply_text('❌ Error removing symbol from watchlist.')
    
    async def listsymbols_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List all symbols in watchlist"""
        try:
            symbols = self.file_manager.load_json_file('symbols.json', [])
            
            if not symbols:
                await update.message.reply_text('📊 Watchlist is empty.\nUse /addsymbol to add stocks.')
                return
            
            # Create formatted list
            text = f"📊 <b>Watchlist</b> ({len(symbols)} symbols):\n\n"
            
            # Group symbols in rows of 5 for better display
            for i in range(0, len(symbols), 5):
                row_symbols = symbols[i:i+5]
                text += "• " + " • ".join(row_symbols) + "\n"
            
            text += f"\n💡 Use /check SYMBOL to analyze any stock\n📈 Use /addsymbol to add more symbols"
            
            await update.message.reply_text(text, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"List symbols command error: {e}")
            await update.message.reply_text('❌ Error retrieving watchlist.')
    
    async def setalert_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set price change alert with validation"""
        try:
            user = update.effective_user
            if not self._is_admin(user.id):
                await update.message.reply_text('❌ Unauthorized access.')
                return
            
            if len(context.args) < 2:
                await update.message.reply_text('Usage: /setalert SYMBOL PERCENTAGE\nExample: /setalert AAPL 2.5')
                return
            
            symbol = context.args[0].upper().strip()
            
            try:
                percentage = float(context.args[1])
                if percentage <= 0 or percentage > 50:
                    await update.message.reply_text('❌ Percentage must be between 0.1 and 50.')
                    return
            except ValueError:
                await update.message.reply_text('❌ Invalid percentage value.')
                return
            
            # Load current alerts
            alerts = self.file_manager.load_json_file('alerts.json', {})
            
            # Set alert
            alerts[symbol] = percentage
            
            # Save alerts
            if self.file_manager.save_json_file('alerts.json', alerts):
                await update.message.reply_text(f'🚨 Set {percentage}% change alert for {symbol}.')
                logger.info(f"Set alert for {symbol} at {percentage}%")
            else:
                await update.message.reply_text('❌ Failed to save alert.')
            
        except Exception as e:
            logger.error(f"Set alert command error: {e}")
            await update.message.reply_text('❌ Error setting price alert.')
    
    # Timer-related commands
    
    async def morningreport_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manually trigger morning report"""
        try:
            user = update.effective_user
            if not self._is_admin(user.id):
                await update.message.reply_text('❌ Unauthorized access.')
                return
            
            await update.message.reply_text('📊 Generating morning report...')
            await self.send_morning_report()
            await update.message.reply_text('✅ Morning report sent to channel!')
            
        except Exception as e:
            logger.error(f"Morning report command error: {e}")
            await update.message.reply_text('❌ Error generating morning report.')
    
    async def setmorningtime_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set morning report time"""
        try:
            user = update.effective_user
            if not self._is_admin(user.id):
                await update.message.reply_text('❌ Unauthorized access.')
                return
            
            if not context.args:
                current_time = self.timer_manager.morning_time
                await update.message.reply_text(f'⏰ Current morning time: {current_time}\n\nUsage: /setmorningtime HH:MM\nExample: /setmorningtime 08:30')
                return
            
            new_time = context.args[0].strip()
            
            if self.timer_manager.set_morning_time(new_time):
                await update.message.reply_text(f'✅ Morning report time set to {new_time}')
            else:
                await update.message.reply_text('❌ Invalid time format. Use HH:MM (e.g., 08:30)')
            
        except Exception as e:
            logger.error(f"Set morning time command error: {e}")
            await update.message.reply_text('❌ Error setting morning time.')
    
    async def testtimer_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Toggle test mode for timer"""
        try:
            user = update.effective_user
            if not self._is_admin(user.id):
                await update.message.reply_text('❌ Unauthorized access.')
                return
            
            test_mode = self.timer_manager.toggle_test_mode()
            
            if test_mode:
                await update.message.reply_text('🧪 <b>TEST MODE ENABLED</b>\n\nMorning reports will now be sent every 2 minutes.\nUse /testtimer again to disable.', parse_mode='HTML')
            else:
                await update.message.reply_text('✅ <b>TEST MODE DISABLED</b>\n\nTimer back to normal schedule.', parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Test timer command error: {e}")
            await update.message.reply_text('❌ Error toggling test mode.')
    
    async def timerstatus_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Check comprehensive timer status"""
        try:
            user = update.effective_user
            if not self._is_admin(user.id):
                await update.message.reply_text('❌ Unauthorized access.')
                return
            
            status = self.timer_manager.get_status()
            symbols_count = len(self.file_manager.load_json_file('symbols.json', []))
            alerts_count = len(self.file_manager.load_json_file('alerts.json', {}))
            
            status_text = f"""⏰ <b>Bot Status Report</b>

🔄 <b>Timer Status:</b>
• Running: {'✅ Yes' if status['running'] else '❌ No'}
• Test Mode: {'✅ Enabled (2-min reports)' if status['test_mode'] else '❌ Disabled'}
• Thread Active: {'✅ Yes' if status['thread_alive'] else '❌ No'}

📅 <b>Schedule:</b>
• Morning Time: {status['morning_time']}
• Timezone Offset: +{status['timezone_offset']} hours

📊 <b>Watchlist:</b>
• Symbols: {symbols_count}
• Price Alerts: {alerts_count}

💬 <b>Channel:</b>
• Target: {self.config.get('CHANNEL_ID', 'Not configured')}

🔌 <b>API Status:</b>
• Data Source: Twelve Data API
• Manual Signals: {'✅ Active' if hasattr(self.signal_checker, 'manual_handler') else '❌ Inactive'}

🤖 <b>Bot Health:</b>
• Admin Users: {len(self.config.get('ADMIN_IDS', []))}
• Event Loop: {'✅ Active' if self.main_loop and not self.main_loop.is_closed() else '❌ Inactive'}"""

            await update.message.reply_text(status_text, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Timer status command error: {e}")
            await update.message.reply_text('❌ Error retrieving status information.')