import json
import os
import logging
import asyncio
import threading
import signal
from datetime import datetime, time
from pathlib import Path
from typing import Optional, Dict, Any

from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, 
    CommandHandler, 
    MessageHandler, 
    CallbackQueryHandler,
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

logger = logging.getLogger(__name__)

class TradingBotError(Exception):
    """Custom exception for bot-related errors"""
    pass

class ConfigurationError(TradingBotError):
    """Raised when there are configuration issues"""
    pass

class MessageSender:
    """Handles message sending with ultra-optimized sticker performance"""
    
    # Online sticker file_ids (public stickers from Telegram)
    # Replace these with your actual sticker file_ids after uploading once
    ONLINE_STICKERS = {
        'up': 'CAACAgUAAyEGAASye8atAAIBuGjX5C6xsQiwwpF3AoNsnzJSCxePAAMaAAJqd7lW0Asv9Lqx91Y2BA',
        'down': 'CAACAgUAAyEGAASye8atAAIBvGjX5ICGjWmsw8JsGtiU_AaenDKbAAIFGgACane5VpvbrM_e5Ny8NgQ',
        'win': 'CAACAgUAAyEGAASye8atAAIBumjX5EChnQ834Tpya6pNi51-x_iLAAICGgACane5VqlcfYk6QitHNgQ',
        'mtg_up': 'CAACAgUAAyEGAASye8atAAIBzGjjpNOgS3fMC3zhW9CbDFp5EnM3AAIBHAACrIUgV0db1hepsiAxNgQ',
        'mtg_down': 'CAACAgUAAyEGAASye8atAAIBzWjjpNVIFg5T39PlWno9lrwuHWbCAAICHAACrIUgVxtkeg98IYQtNgQ'
    }
    
    def __init__(self, bot: Bot, config: Dict[str, Any]):
        self.bot = bot
        self.config = config
        self.channel_id = config.get('CHANNEL_ID')
        
        # Sticker paths for initial upload
        self.sticker_paths = {
            'up': r"E:\malek\Downloads\telegram_trading_bot_core\up.webp",
            'down': r"E:\malek\Downloads\telegram_trading_bot_core\down.webp", 
            'win': r"E:\malek\Downloads\telegram_trading_bot_core\win.webp",
            'mtg_up': r"E:\malek\Downloads\telegram_trading_bot_core\MTGUp.webp",
            'mtg_down': r"E:\malek\Downloads\telegram_trading_bot_core\MTGDown.webp"
        }
        
        # Load and warm up stickers
        self._init_stickers()
        
    def _init_stickers(self):
        """Initialize and warm up stickers synchronously at startup"""
        cache_file = Path(__file__).parent / 'data' / 'sticker_cache.json'
        
        # Load cached file_ids
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    self.ONLINE_STICKERS.update(cached)
                    logger.info(f"Loaded {len(cached)} cached sticker file_ids")
            except Exception as e:
                logger.warning(f"Failed to load sticker cache: {e}")
    
    async def warm_up_stickers(self):
        """Upload stickers once and cache their file_ids for instant future use"""
        if not self.channel_id:
            logger.warning("No channel_id configured, skipping sticker warmup")
            return
        
        needs_upload = [name for name, file_id in self.ONLINE_STICKERS.items() if not file_id]
        
        if not needs_upload:
            logger.info("All stickers already warmed up")
            return
        
        logger.info(f"Warming up {len(needs_upload)} stickers...")
        
        for sticker_name in needs_upload:
            if sticker_name not in self.sticker_paths:
                continue
                
            sticker_path = self.sticker_paths[sticker_name]
            if not os.path.exists(sticker_path):
                logger.warning(f"Sticker file not found: {sticker_path}")
                continue
            
            try:
                with open(sticker_path, 'rb') as sticker_file:
                    message = await self.bot.send_sticker(
                        chat_id=self.channel_id,
                        sticker=sticker_file
                    )
                    
                    # Cache the file_id
                    self.ONLINE_STICKERS[sticker_name] = message.sticker.file_id
                    logger.info(f"Warmed up sticker: {sticker_name}")
                    
                    # Delete the warmup message to keep channel clean
                    try:
                        await message.delete()
                    except:
                        pass
                    
                    await asyncio.sleep(0.5)  # Rate limiting
                    
            except Exception as e:
                logger.error(f"Failed to warm up sticker {sticker_name}: {e}")
        
        # Save to cache
        self._save_sticker_cache()
        logger.info("Sticker warmup complete")
    
    def _save_sticker_cache(self):
        """Save sticker file_ids to cache"""
        try:
            cache_dir = Path(__file__).parent / 'data'
            cache_dir.mkdir(exist_ok=True)
            cache_file = cache_dir / 'sticker_cache.json'
            
            with open(cache_file, 'w') as f:
                json.dump(self.ONLINE_STICKERS, f, indent=2)
            logger.info("Sticker cache saved")
        except Exception as e:
            logger.error(f"Failed to save sticker cache: {e}")

    async def send_sticker_ultra_fast(self, sticker_name: str) -> bool:
        """Send sticker instantly using pre-cached file_id (typically <300ms)"""
        if not self.channel_id:
            return False
        
        file_id = self.ONLINE_STICKERS.get(sticker_name)
        
        if not file_id:
            logger.warning(f"Sticker {sticker_name} not warmed up, attempting upload")
            return await self._upload_and_cache_sticker(sticker_name)
        
        try:
            # Direct send with cached file_id - this is INSTANT
            await self.bot.send_sticker(
                chat_id=self.channel_id,
                sticker=file_id,
                read_timeout=5,
                write_timeout=5
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to send sticker {sticker_name}: {e}")
            # Invalidate cache and try upload
            self.ONLINE_STICKERS[sticker_name] = None
            return await self._upload_and_cache_sticker(sticker_name)
    
    async def _upload_and_cache_sticker(self, sticker_name: str) -> bool:
        """Fallback: upload sticker and cache file_id"""
        if sticker_name not in self.sticker_paths:
            return False
        
        sticker_path = self.sticker_paths[sticker_name]
        if not os.path.exists(sticker_path):
            logger.error(f"Sticker file not found: {sticker_path}")
            return False
        
        try:
            with open(sticker_path, 'rb') as sticker_file:
                message = await self.bot.send_sticker(
                    chat_id=self.channel_id,
                    sticker=sticker_file,
                    read_timeout=10,
                    write_timeout=10
                )
                
                # Cache for next time
                self.ONLINE_STICKERS[sticker_name] = message.sticker.file_id
                self._save_sticker_cache()
                logger.info(f"Uploaded and cached sticker: {sticker_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to upload sticker {sticker_name}: {e}")
            return False

    async def send_message_with_retry(self, chat_id: str, text: str, **kwargs) -> bool:
        """Send message with retry logic"""
        for attempt in range(2):
            try:
                await self.bot.send_message(
                    chat_id=chat_id, 
                    text=text,
                    read_timeout=8,
                    write_timeout=8,
                    **kwargs
                )
                return True
                
            except Exception as e:
                if attempt < 1:
                    await asyncio.sleep(0.5)
                    continue
                logger.error(f"Failed to send message: {e}")
                    
        return False

    async def send_photo_with_retry(self, chat_id: str, photo, caption: str = None, **kwargs) -> bool:
        """Send photo with retry logic"""
        for attempt in range(2):
            try:
                await self.bot.send_photo(
                    chat_id=chat_id, 
                    photo=photo, 
                    caption=caption,
                    parse_mode='HTML',
                    read_timeout=15,
                    write_timeout=15,
                    **kwargs
                )
                return True
                
            except Exception as e:
                if attempt < 1:
                    await asyncio.sleep(1.0)
                    continue
                logger.error(f"Failed to send photo: {e}")
                    
        return False

class TimerManager:
    """Manages the morning report timer"""
    
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
            return
            
        self.main_loop = event_loop
        self.timer_running = True
        self.timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self.timer_thread.start()
    
    def stop_timer(self):
        """Stop the morning timer"""
        self.timer_running = False
        if self.timer_thread and self.timer_thread.is_alive():
            self.timer_thread.join(timeout=5.0)
    
    def set_morning_time(self, time_str: str) -> bool:
        """Set new morning time with validation"""
        try:
            parsed_time = self._parse_time(time_str)
            self.morning_time = time_str
            return True
        except ValueError:
            return False
    
    def toggle_test_mode(self) -> bool:
        """Toggle test mode (reports every 2 minutes)"""
        self.test_mode = not self.test_mode
        return self.test_mode
    
    def get_status(self) -> Dict[str, Any]:
        """Get timer status"""
        return {
            'running': self.timer_running,
            'test_mode': self.test_mode,
            'morning_time': self.morning_time,
            'timezone_offset': self.timezone_offset,
            'thread_alive': self.timer_thread.is_alive() if self.timer_thread else False
        }
    
    def _timer_loop(self):
        """Main timer loop"""
        while self.timer_running:
            try:
                current_time = datetime.now()
                
                if self.test_mode:
                    sleep_seconds = 120
                else:
                    target_time = self._parse_time(self.morning_time)
                    next_trigger = current_time.replace(
                        hour=target_time.hour,
                        minute=target_time.minute,
                        second=0,
                        microsecond=0
                    )
                    
                    if current_time >= next_trigger:
                        next_trigger = next_trigger.replace(day=next_trigger.day + 1)
                    
                    sleep_seconds = (next_trigger - current_time).total_seconds()
                
                if sleep_seconds > 0:
                    sleep_interval = min(sleep_seconds, 60)
                    end_time = current_time.timestamp() + sleep_seconds
                    
                    while datetime.now().timestamp() < end_time and self.timer_running:
                        threading.Event().wait(min(sleep_interval, end_time - datetime.now().timestamp()))
                
                if self.timer_running and self._should_send_report():
                    self._schedule_morning_report()
                
                if self.timer_running:
                    threading.Event().wait(120)
                
            except Exception:
                threading.Event().wait(60)
    
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
        
        return abs((current_time - target_time_today).total_seconds()) < 120
    
    def _schedule_morning_report(self) -> bool:
        """Schedule morning report in the main event loop"""
        if not self.main_loop or self.main_loop.is_closed():
            return False
        
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.bot.send_morning_report(),
                self.main_loop
            )
            future.result(timeout=60)
            return True
        except Exception:
            return False
    
    def _parse_time(self, time_str: str) -> time:
        """Parse time string"""
        if not isinstance(time_str, str) or ':' not in time_str:
            raise ValueError("Invalid time format")
        
        try:
            parts = time_str.split(':')
            if len(parts) != 2:
                raise ValueError("Invalid time format")
            
            hour, minute = int(parts[0]), int(parts[1])
            
            if not (0 <= hour <= 23) or not (0 <= minute <= 59):
                raise ValueError("Invalid time values")
            
            return time(hour, minute)
            
        except (ValueError, TypeError):
            raise ValueError("Invalid time format")

class FileManager:
    """Handles file operations"""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def load_json_file(self, filename: str, default=None):
        """Load JSON file with error handling"""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            return default if default is not None else []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return default if default is not None else []
                
                return json.loads(content)
                
        except Exception:
            return default if default is not None else []
    
    def save_json_file(self, filename: str, data) -> bool:
        """Save JSON file with atomic write"""
        filepath = self.data_dir / filename
        temp_filepath = filepath.with_suffix('.json.tmp')
        
        try:
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            if os.name == 'nt':
                if filepath.exists():
                    filepath.unlink()
                temp_filepath.rename(filepath)
            else:
                temp_filepath.rename(filepath)
            
            return True
            
        except Exception:
            if temp_filepath.exists():
                try:
                    temp_filepath.unlink()
                except:
                    pass
            return False

class TradingBot:
    """Production-ready trading bot with webhook support"""
    
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
        
        # Store user states for photo upload
        self.user_states = {}
        
        # Register command handlers
        self._register_handlers()
        
        # Setup graceful shutdown
        self._setup_shutdown_handlers()
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration"""
        if not isinstance(config, dict):
            raise ConfigurationError("Configuration must be a dictionary")
        
        required_keys = ['TELEGRAM_TOKEN', 'CHANNEL_ID', 'TWELVE_DATA_API_KEY']
        missing_keys = [key for key in required_keys if not config.get(key)]
        
        if missing_keys:
            raise ConfigurationError(f"Missing required configuration: {', '.join(missing_keys)}")
        
        token = config['TELEGRAM_TOKEN']
        if not isinstance(token, str) or ':' not in token or len(token) < 20:
            raise ConfigurationError("Invalid TELEGRAM_TOKEN format")
        
        defaults = {
            'ADMIN_IDS': [],
            'MORNING_TIME': '09:00',
            'TIMEZONE_OFFSET': 0,
            'CHECK_INTERVAL_SECONDS': 300,
            'SHORT_WINDOW': 5,
            'LONG_WINDOW': 20,
            'MAX_RETRIES': 2,
            'MESSAGE_TIMEOUT': 8
        }
        
        for key, default_value in defaults.items():
            config.setdefault(key, default_value)
        
        return config
    
    def _init_telegram_app(self):
        """Initialize Telegram application"""
        try:
            self.app = ApplicationBuilder() \
                .token(self.config['TELEGRAM_TOKEN']) \
                .connect_timeout(20) \
                .read_timeout(20) \
                .write_timeout(20) \
                .build()
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Telegram application: {e}")
    async def handle_update(self, update_data: dict):
        """
        Handle incoming webhook updates - CRITICAL for webhook mode
        """
        try:
            # Convert dict to Update object
            update = Update.de_json(update_data, self.app.bot)
            
            if not update:
                logger.warning("Received invalid update")
                return
            
            # Process the update through the application
            await self.app.process_update(update)
            
        except Exception as e:
            logger.error(f"Error handling update: {e}")
            logger.exception("Detailed error:")
    
    async def initialize_webhook_mode(self):
        """Initialize bot for webhook mode"""
        try:
            # Initialize the application
            await self.app.initialize()
            
            # Warm up stickers
            await self.message_sender.warm_up_stickers()
            
            logger.info("✅ Bot initialized for webhook mode")
            
        except Exception as e:
            logger.error(f"Failed to initialize webhook mode: {e}")
            raise
    
    def set_webhook(self, webhook_url: str):
        """Set webhook URL"""
        try:
            import requests
            api_url = f"https://api.telegram.org/bot{self.config['TELEGRAM_TOKEN']}/setWebhook"
            payload = {"url": webhook_url}
            
            response = requests.post(api_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("ok"):
                    logger.info(f"✅ Webhook set successfully: {webhook_url}")
                    return True
                else:
                    logger.error(f"❌ Webhook set failed: {result.get('description')}")
                    return False
            else:
                logger.error(f"❌ HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error setting webhook: {e}")
            return False
            
    def _register_handlers(self):
        """Register all command and message handlers"""
        handlers = [
            ('start', self.start_cmd),
            ('help', self.help_cmd),
            ('addsymbol', self.addsymbol_cmd),
            ('removesymbol', self.removesymbol_cmd),
            ('symbols', self.listsymbols_cmd),
            ('listsymbols', self.listsymbols_cmd),
            ('setalert', self.setalert_cmd),
            ('news', self.news_cmd),
            ('summary', self.summary_cmd),
            ('check', self.check_cmd),
            ('price', self.price_cmd),
            ('custom_signal', self.custom_signal_cmd),
            ('cancel', self.cancel_custom_cmd),
            ('history', self.history_cmd),
            ('morningreport', self.morningreport_cmd),
            ('setmorningtime', self.setmorningtime_cmd),
            ('testtimer', self.testtimer_cmd),
            ('timerstatus', self.timerstatus_cmd),
            ('warmup_stickers', self.warmup_stickers_cmd),
        ]
        
        for command, handler in handlers:
            self.app.add_handler(CommandHandler(command, handler))
        
        self.app.add_handler(CallbackQueryHandler(self.handle_callback_query))
        
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_input)
        )
        
        self.app.add_handler(
            MessageHandler(filters.PHOTO, self.handle_photo_upload)
        )
    
    def _setup_shutdown_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _post_init(self):
        """Post-initialization tasks (run after event loop starts)"""
        try:
            # Warm up stickers for instant sending
            await self.message_sender.warm_up_stickers()
            logger.info("Bot initialization complete")
        except Exception as e:
            logger.error(f"Error during post-init: {e}")
    
    def start(self):
        """Start the bot"""
        try:
            self.main_loop = asyncio.get_event_loop()
            
            # Schedule post-init tasks
            self.main_loop.create_task(self._post_init())
            
            self.timer_manager.start_timer(self.main_loop)
            
            self.app.run_polling(
                poll_interval=1.0,
                timeout=30,
                bootstrap_retries=3,
                allowed_updates=None,
                drop_pending_updates=True
            )
            
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            raise
        finally:
            self.stop()
    
    def stop(self):
        """Stop all bot components gracefully"""
        try:
            if self.timer_manager:
                self.timer_manager.stop_timer()
            
            if self.signal_checker and hasattr(self.signal_checker, 'stop'):
                self.signal_checker.stop()
            
            if self.app:
                self.app.stop_running()
            
        except Exception:
            pass
    
    def _is_admin(self, user_id: int) -> bool:
        """Check if user is an admin"""
        try:
            admin_ids = self.config.get('ADMIN_IDS', [])
            return int(user_id) in [int(admin_id) for admin_id in admin_ids]
        except (ValueError, TypeError):
            return False
    
    async def post_message(self, text: str, **kwargs) -> bool:
        """Post message to configured channel"""
        channel_id = self.config.get('CHANNEL_ID')
        if not channel_id:
            return False
        
        return await self.message_sender.send_message_with_retry(
            chat_id=channel_id,
            text=text,
            parse_mode='HTML',
            **kwargs
        )
    
    def post_message_from_thread(self, text: str) -> bool:
        """Post message from another thread safely - WEBHOOK COMPATIBLE"""
        if not self.main_loop or self.main_loop.is_closed():
            # Fallback for webhook mode: create new task in current loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Loop is running (webhook mode) - schedule task
                    asyncio.ensure_future(self.post_message(text), loop=loop)
                    return True
                else:
                    # No running loop - try to create one
                    return asyncio.run(self.post_message(text))
            except Exception as e:
                logger.error(f"Failed to post message (no loop): {e}")
                return False
        
        try:
            # Original method for polling mode
            future = asyncio.run_coroutine_threadsafe(
                self.post_message(text),
                self.main_loop
            )
            return future.result(timeout=self.config.get('MESSAGE_TIMEOUT', 8))
        except Exception as e:
            logger.error(f"Failed to post message from thread: {e}")
            return False

    def send_sticker_to_channel_instant(self, sticker_name: str) -> bool:
        """Send sticker to channel INSTANTLY - WEBHOOK COMPATIBLE"""
        if not self.main_loop or self.main_loop.is_closed():
            # Fallback for webhook mode
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(
                        self.message_sender.send_sticker_ultra_fast(sticker_name),
                        loop=loop
                    )
                    return True
                else:
                    return asyncio.run(self.message_sender.send_sticker_ultra_fast(sticker_name))
            except Exception as e:
                logger.error(f"Failed to send sticker (no loop): {e}")
                return False
        
        try:
            # Original method for polling mode
            future = asyncio.run_coroutine_threadsafe(
                self.message_sender.send_sticker_ultra_fast(sticker_name),
                self.main_loop
            )
            return future.result(timeout=1.5)
        except Exception as e:
            logger.error(f"Failed to send sticker instantly: {e}")
            return False
    async def _send_photo_async(self, photo, caption: str = None) -> bool:
        """Async photo sending method"""
        channel_id = self.config.get('CHANNEL_ID')
        if not channel_id:
            return False
        
        return await self.message_sender.send_photo_with_retry(
            chat_id=channel_id,
            photo=photo,
            caption=caption
        )
    
    async def send_morning_report(self):
        """Generate and send morning report"""
        try:
            symbols = self.file_manager.load_json_file('symbols.json', [])
            if not symbols:
                return
            
            current_date = datetime.now().strftime('%B %d, %Y')
            report_text = f"🌅 <b>Good Morning! Daily Market Report</b>\n📅 {current_date}\n\n"
            
            processed_count = 0
            for symbol in symbols[:5]:
                try:
                    if processed_count > 0:
                        await asyncio.sleep(1.2)
                    
                    signal_data = generate_signal(symbol)
                    summary_data = fetch_summary(symbol)
                    
                    change_emoji = "🔺" if summary_data['price_change'] > 0 else "🔻" if summary_data['price_change'] < 0 else "➡️"
                    signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(signal_data['signal'], "🟡")
                    
                    report_text += (
                        f"{signal_emoji} <b>{symbol}</b> — {signal_data['signal']}\n"
                        f"💰 ${summary_data['current_price']:.2f} {change_emoji} {summary_data['price_change_pct']:+.2f}%\n"
                        f"📊 Vol: {summary_data['volume']:,}\n\n"
                    )
                    
                    processed_count += 1
                    
                except Exception:
                    report_text += f"❌ <b>{symbol}</b> — Data unavailable\n\n"
            
            report_text += "📈 <i>Have a profitable trading day!</i>\n"
            report_text += "📡 <i>Data from Twelve Data API</i>"
            
            await self.post_message(report_text)
            
        except Exception:
            pass
    
    def _get_custom_signal_keyboard(self, user_id: str) -> Optional[InlineKeyboardMarkup]:
        """Generate keyboard for custom signal creation"""
        if not hasattr(self.signal_checker, 'custom_handler'):
            return None
        
        step = self.signal_checker.custom_handler.get_user_current_step(user_id)
        
        if step == 3:  # Direction selection
            keyboard = [
                [InlineKeyboardButton("🔼 UP", callback_data="direction_UP")],
                [InlineKeyboardButton("🔽 DOWN", callback_data="direction_DOWN")]
            ]
            return InlineKeyboardMarkup(keyboard)
        
        elif step == 4:  # First Result selection
            keyboard = [
                [InlineKeyboardButton("🎉 WIN", callback_data="first_result_WIN")],
                [InlineKeyboardButton("💔 LOSS", callback_data="first_result_LOSS")]
            ]
            return InlineKeyboardMarkup(keyboard)
        
        elif step == 5:  # MTG Direction selection
            keyboard = [
                [InlineKeyboardButton("🔼 MTG UP", callback_data="mtg_direction_MTGUP")],
                [InlineKeyboardButton("🔽 MTG DOWN", callback_data="mtg_direction_MTGDOWN")]
            ]
            return InlineKeyboardMarkup(keyboard)
        
        elif step == 6:  # Final Result selection
            keyboard = [
                [InlineKeyboardButton("🎉 WIN", callback_data="final_result_WIN")],
                [InlineKeyboardButton("💔 LOSS", callback_data="final_result_LOSS")]
            ]
            return InlineKeyboardMarkup(keyboard)
        
        elif step == 7:  # Final action
            keyboard = [
                [InlineKeyboardButton("📸 Upload Screenshot", callback_data="final_action_SCREENSHOT")],
                [InlineKeyboardButton("🆕 New Signal", callback_data="final_action_NEW_SIGNAL")]
            ]
            return InlineKeyboardMarkup(keyboard)
        
        return None
    
    async def handle_photo_upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle photo uploads for screenshot functionality - OPTIMIZED for instant posting"""
        try:
            user = update.effective_user
            if not self._is_admin(user.id):
                await update.message.reply_text("❌ Access denied. Admin privileges required.")
                return
            
            user_id = str(user.id)
            
            if user_id not in self.user_states or self.user_states[user_id] != 'waiting_for_screenshot':
                await update.message.reply_text(
                    "📸 I received your photo, but you're not currently in screenshot upload mode. "
                    "Use /custom_signal and select 'Upload Screenshot' when prompted."
                )
                return
            
            photo = update.message.photo[-1]
            photo_file = await photo.get_file()
            
            self.user_states[user_id] = None
            
            caption = "📸 <b>Trading Signal Screenshot</b>\n\n"
            caption += f"📊 Uploaded by admin\n"
            caption += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # INSTANT confirmation to user - don't wait for channel upload
            await update.message.reply_text(
                "📤 Uploading screenshot to channel... ⚡",
            )
            
            channel_id = self.config.get('CHANNEL_ID')
            if channel_id:
                # Send photo asynchronously 
                success = await self.message_sender.send_photo_with_retry(
                    chat_id=channel_id,
                    photo=photo_file.file_id,
                    caption=caption
                )
                
                if success:
                    await update.message.reply_text(
                        "✅ Screenshot uploaded successfully to the channel! ⚡\n\n"
                        "What would you like to do next?",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🆕 New Signal", callback_data="final_action_NEW_SIGNAL")]
                        ])
                    )
                else:
                    await update.message.reply_text("❌ Failed to upload screenshot to channel.")
            else:
                await update.message.reply_text("❌ Channel not configured.")
                
        except Exception:
            user_id = str(update.effective_user.id)
            if user_id in self.user_states:
                self.user_states[user_id] = None
            await update.message.reply_text("❌ Error processing your screenshot. Please try again.")

    async def handle_text_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text input during custom signal process"""
        try:
            user = update.effective_user
            if not self._is_admin(user.id):
                return
            
            user_id = str(user.id)
            message_text = update.message.text.strip()
            
            if (hasattr(self.signal_checker, 'custom_handler') and 
                self.signal_checker.custom_handler.is_user_in_process(user_id)):
                
                response = self.signal_checker.custom_handler.process_input(user_id, message_text)
                keyboard = self._get_custom_signal_keyboard(user_id)
                
                await update.message.reply_text(
                    response, 
                    parse_mode='HTML',
                    reply_markup=keyboard
                )
                
        except Exception:
            await update.message.reply_text("❌ An error occurred processing your message.")

    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries from inline keyboards"""
        try:
            query = update.callback_query
            user_id = str(query.from_user.id)
            callback_data = query.data
            
            if not self._is_admin(query.from_user.id):
                await query.answer("❌ Unauthorized access.")
                return
            
            await query.answer()
            
            if not (hasattr(self.signal_checker, 'custom_handler') and 
                    self.signal_checker.custom_handler.is_user_in_process(user_id)):
                await query.edit_message_text("❌ No active custom signal process. Use /custom_signal to start.")
                return
            
            # Process callback data
            if callback_data.startswith("direction_"):
                direction = callback_data.replace("direction_", "")
                response = self.signal_checker.custom_handler.process_input(user_id, callback_data=direction)
                
            elif callback_data.startswith("first_result_"):
                result = callback_data.replace("first_result_", "")
                response = self.signal_checker.custom_handler.process_input(user_id, callback_data=result)
                
            elif callback_data.startswith("mtg_direction_"):
                mtg_direction = callback_data.replace("mtg_direction_", "")
                response = self.signal_checker.custom_handler.process_input(user_id, callback_data=mtg_direction)
                
            elif callback_data.startswith("final_result_"):
                final_result = callback_data.replace("final_result_", "")
                response = self.signal_checker.custom_handler.process_input(user_id, callback_data=final_result)
                
            elif callback_data.startswith("final_action_"):
                action = callback_data.replace("final_action_", "")
                
                if action == "SCREENSHOT":
                    self.user_states[user_id] = 'waiting_for_screenshot'
                    
                    await query.edit_message_text(
                        "📸 <b>Upload Screenshot</b>\n\n"
                        "Please send your screenshot now. You can:\n"
                        "• Take a photo with your camera\n"
                        "• Choose from your gallery\n"
                        "• Send any image file\n\n"
                        "The screenshot will be posted to the trading channel with your signal information.",
                        parse_mode='HTML'
                    )
                    return
                    
                else:
                    response = self.signal_checker.custom_handler.process_input(user_id, callback_data=action)
                
            else:
                await query.edit_message_text("❌ Unknown action. Please try again.")
                return
            
            # Get keyboard for next step
            keyboard = self._get_custom_signal_keyboard(user_id)
            
            # Update the message
            await query.edit_message_text(
                response,
                parse_mode='HTML',
                reply_markup=keyboard
            )
            
        except Exception:
            try:
                await query.edit_message_text("❌ An error occurred processing your action. Please try /custom_signal to restart.")
            except Exception:
                try:
                    await query.message.reply_text("❌ An error occurred. Please use /custom_signal to start over.")
                except Exception:
                    pass

    # Command handlers
    
    async def start_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced start command"""
        welcome_text = """👋 <b>Welcome to TradePulse Bot!</b>
Your AI-powered trading assistant for signals and market analysis.

🚀 <b>Quick Start:</b>
• <code>/help</code> - See all commands
• <code>/custom_signal</code> - Create trading signal
• <code>/check AAPL</code> - Analyze a stock
• <code>/addsymbol TSLA</code> - Add to watchlist

💡 <b>Key Features:</b>
✅ Custom signal creation with step-by-step process
✅ Signal history tracking with results
✅ Enhanced market analysis
✅ Automated morning reports
✅ Price alerts and notifications

📊 <b>Data Sources:</b>
• Real-time prices from Twelve Data API
• Technical analysis with SMA crossovers
• Multi-source news aggregation
• Company fundamentals and metrics

<i>Ready to start trading smarter! 🎯</i>"""
        
        await update.message.reply_text(welcome_text, parse_mode='HTML')

    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show help with all available commands"""
        help_text = """🤖 <b>TradePulse Bot Commands</b>

📊 <b>Market Data (Posted to Channel):</b>
• /check SYMBOL - Get trading signal
• /price SYMBOL - Get current price
• /summary SYMBOL - Get detailed summary  
• /news SYMBOL - Get latest news

🎯 <b>Custom Signals:</b>
• /custom_signal - Create custom signal
• /history [limit] - View signal history
• /cancel - Cancel signal creation

📋 <b>Watchlist Management:</b>
• /addsymbol SYMBOL - Add to watchlist
• /removesymbol SYMBOL - Remove from watchlist
• /symbols - View watchlist
• /setalert SYMBOL % - Set price alert

⏰ <b>Reports & Automation:</b>
• /morningreport - Send report now
• /setmorningtime HH:MM - Set report time
• /testtimer - Toggle test mode
• /timerstatus - View timer status

🔧 <b>Admin Commands:</b>
• /warmup_stickers - Warm up stickers for instant sending

❓ Use /help to see this message again."""

        await update.message.reply_text(help_text, parse_mode='HTML')

    async def custom_signal_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /custom_signal command"""
        try:
            user = update.effective_user
            if not self._is_admin(user.id):
                await update.message.reply_text("❌ Access denied. Admin privileges required.")
                return
            
            user_id = str(user.id)
            
            if (hasattr(self.signal_checker, 'custom_handler') and 
                self.signal_checker.custom_handler.is_user_in_process(user_id)):
                await update.message.reply_text(
                    "⚠️ You already have an active custom signal process. "
                    "Use /cancel to cancel it first, or complete the current one."
                )
                return
            
            response = self.signal_checker.custom_handler.start_custom_signal(user_id)
            await update.message.reply_text(response, parse_mode='HTML')
            
        except Exception:
            await update.message.reply_text("❌ Error starting custom signal creation. Please try again.")

    async def cancel_custom_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /cancel command"""
        try:
            user = update.effective_user
            if not self._is_admin(user.id):
                await update.message.reply_text("❌ Access denied. Admin privileges required.")
                return
            
            user_id = str(user.id)
            response = self.signal_checker.custom_handler.cancel_signal(user_id)
            await update.message.reply_text(response, parse_mode='HTML')
            
        except Exception:
            await update.message.reply_text("❌ Error cancelling custom signal.")

    async def history_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /history command"""
        try:
            user = update.effective_user
            if not self._is_admin(user.id):
                await update.message.reply_text("❌ Access denied. Admin privileges required.")
                return
            
            limit = 10
            if context.args:
                try:
                    limit = int(context.args[0])
                    limit = max(1, min(limit, 50))
                except ValueError:
                    await update.message.reply_text("❌ Invalid limit. Please use a number between 1 and 50.")
                    return
            
            response = self.signal_checker.custom_handler.get_signal_history(limit)
            await update.message.reply_text(response, parse_mode='HTML')
            
        except Exception:
            await update.message.reply_text("❌ Error retrieving signal history.")

    async def warmup_stickers_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manually trigger sticker warmup"""
        try:
            user = update.effective_user
            if not self._is_admin(user.id):
                await update.message.reply_text("❌ Access denied. Admin privileges required.")
                return
            
            await update.message.reply_text("🔄 Warming up stickers for instant sending...")
            await self.message_sender.warm_up_stickers()
            await update.message.reply_text("✅ Sticker warmup complete! All stickers ready for instant delivery.")
            
        except Exception:
            await update.message.reply_text("❌ Error warming up stickers.")

    async def addsymbol_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Add symbol to watchlist"""
        try:
            if not context.args:
                await update.message.reply_text("Usage: /addsymbol SYMBOL\nExample: /addsymbol AAPL")
                return
            
            symbol = context.args[0].upper().strip()
            symbols = self.file_manager.load_json_file('symbols.json', [])
            
            if symbol in symbols:
                await update.message.reply_text(f"📊 {symbol} is already in the watchlist.")
                return
            
            symbols.append(symbol)
            
            if self.file_manager.save_json_file('symbols.json', symbols):
                await update.message.reply_text(f"✅ Added {symbol} to watchlist.")
            else:
                await update.message.reply_text(f"❌ Failed to add {symbol} to watchlist.")
        
        except Exception:
            await update.message.reply_text("❌ Error adding symbol to watchlist.")

    async def removesymbol_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Remove symbol from watchlist"""
        try:
            if not context.args:
                await update.message.reply_text("Usage: /removesymbol SYMBOL\nExample: /removesymbol AAPL")
                return
            
            symbol = context.args[0].upper().strip()
            symbols = self.file_manager.load_json_file('symbols.json', [])
            
            if symbol not in symbols:
                await update.message.reply_text(f"📊 {symbol} is not in the watchlist.")
                return
            
            symbols.remove(symbol)
            
            if self.file_manager.save_json_file('symbols.json', symbols):
                await update.message.reply_text(f"✅ Removed {symbol} from watchlist.")
            else:
                await update.message.reply_text(f"❌ Failed to remove {symbol} from watchlist.")
        
        except Exception:
            await update.message.reply_text("❌ Error removing symbol from watchlist.")

    async def listsymbols_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List all symbols in watchlist"""
        try:
            symbols = self.file_manager.load_json_file('symbols.json', [])
            
            if not symbols:
                await update.message.reply_text("📊 No symbols in watchlist. Use /addsymbol to add some.")
                return
            
            symbol_list = ", ".join(symbols)
            response = f"📊 <b>Current Watchlist ({len(symbols)} symbols):</b>\n\n{symbol_list}"
            await update.message.reply_text(response, parse_mode='HTML')
        
        except Exception:
            await update.message.reply_text("❌ Error retrieving symbol list.")

    async def setalert_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set price alert for symbol"""
        try:
            if len(context.args) != 2:
                await update.message.reply_text(
                    "Usage: /setalert SYMBOL PERCENTAGE\n"
                    "Example: /setalert AAPL 5.0\n"
                    "(Alert when AAPL moves 5% or more)"
                )
                return
            
            symbol = context.args[0].upper().strip()
            try:
                threshold = float(context.args[1])
            except ValueError:
                await update.message.reply_text("❌ Invalid percentage. Please use a number.")
                return
            
            if not (0.1 <= threshold <= 50):
                await update.message.reply_text("❌ Percentage must be between 0.1 and 50.")
                return
            
            alerts = self.file_manager.load_json_file('alerts.json', {})
            alerts[symbol] = threshold
            
            if self.file_manager.save_json_file('alerts.json', alerts):
                await update.message.reply_text(f"🚨 Alert set for {symbol} at {threshold}% movement.")
            else:
                await update.message.reply_text(f"❌ Failed to set alert for {symbol}.")
        
        except Exception:
            await update.message.reply_text("❌ Error setting price alert.")

    async def news_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get news for symbol and post to channel"""
        try:
            if not context.args:
                await update.message.reply_text("Usage: /news SYMBOL\nExample: /news AAPL")
                return
            
            symbol = context.args[0].upper().strip()
            
            # Notify user that news is being fetched
            await update.message.reply_text(f"🔄 Fetching news for {symbol}...")
            
            news_items = fetch_news(symbol)
            
            if not news_items:
                await update.message.reply_text(f"📰 No recent news found for {symbol}")
                return
            
            response = f"📰 <b>Latest News for {symbol}</b>\n"
            response += f"👤 Requested by: {update.effective_user.first_name}\n\n"
            
            for i, item in enumerate(news_items[:5], 1):
                title = item.get('title', 'No title')[:100]
                link = item.get('link', '')
                source = item.get('source', 'unknown')
                
                response += f"{i}. <a href='{link}'>{title}</a>\n"
                response += f"   <i>Source: {source}</i>\n\n"
            
            # Post to channel
            success = await self.post_message(response)
            
            if success:
                await update.message.reply_text(f"✅ News for {symbol} posted to channel!")
            else:
                # Fallback: send to user if channel post fails
                await update.message.reply_text(response, parse_mode='HTML', disable_web_page_preview=True)
                await update.message.reply_text("⚠️ Could not post to channel, sent here instead.")
        
        except Exception as e:
            error_msg = f"❌ Error fetching news for {symbol if 'symbol' in locals() else 'symbol'}: {str(e)}"
            await update.message.reply_text(error_msg)

    async def summary_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get summary for symbol and post to channel"""
        try:
            if not context.args:
                await update.message.reply_text("Usage: /summary SYMBOL\nExample: /summary AAPL")
                return
            
            symbol = context.args[0].upper().strip()
            
            # Notify user that summary is being fetched
            await update.message.reply_text(f"🔄 Fetching summary for {symbol}...")
            
            summary = fetch_summary(symbol)
            
            change_emoji = "🔺" if summary['price_change'] > 0 else "🔻" if summary['price_change'] < 0 else "➡️"
            
            response = f"📊 <b>{summary['company_name']} ({symbol})</b>\n"
            response += f"👤 Requested by: {update.effective_user.first_name}\n\n"
            response += f"💰 Current Price: ${summary['current_price']:.2f}\n"
            response += f"{change_emoji} Change: ${summary['price_change']:+.2f} ({summary['price_change_pct']:+.2f}%)\n"
            response += f"📈 Day High: ${summary['day_high']:.2f}\n"
            response += f"📉 Day Low: ${summary['day_low']:.2f}\n"
            response += f"📊 Volume: {summary['volume']:,}\n"
            response += f"📊 Avg Volume: {summary['avg_volume']:,}\n"
            response += f"🏢 Market Cap: {summary['market_cap']}\n"
            response += f"📊 P/E Ratio: {summary['pe_ratio']}"
            
            # Post to channel
            success = await self.post_message(response)
            
            if success:
                await update.message.reply_text(f"✅ Summary for {symbol} posted to channel!")
            else:
                # Fallback: send to user if channel post fails
                await update.message.reply_text(response, parse_mode='HTML')
                await update.message.reply_text("⚠️ Could not post to channel, sent here instead.")
        
        except Exception as e:
            error_msg = f"❌ Error fetching summary for {symbol if 'symbol' in locals() else 'symbol'}: {str(e)}"
            await update.message.reply_text(error_msg)

    async def check_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get trading signal for symbol and post to channel"""
        try:
            if not context.args:
                await update.message.reply_text("Usage: /check SYMBOL\nExample: /check AAPL")
                return
            
            symbol = context.args[0].upper().strip()
            
            # Notify user that signal is being generated
            await update.message.reply_text(f"🔄 Generating signal for {symbol}...")
            
            signal_data = generate_signal(symbol)
            
            signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(signal_data['signal'], "🟡")
            
            response = f"{signal_emoji} <b>{symbol} - {signal_data['signal']}</b>\n\n"
            response += f"💰 Current Price: ${signal_data['price']:.2f}\n"
            response += f"📊 Short SMA(5): ${signal_data['short_sma']:.2f}\n"
            response += f"📊 Long SMA(20): ${signal_data['long_sma']:.2f}\n"
            response += f"⏰ Generated: {datetime.now().strftime('%H:%M:%S')}\n"
            response += f"👤 Requested by: {update.effective_user.first_name}"
            
            # Post to channel
            success = await self.post_message(response)
            
            if success:
                await update.message.reply_text(f"✅ Signal for {symbol} posted to channel!")
            else:
                # Fallback: send to user if channel post fails
                await update.message.reply_text(response, parse_mode='HTML')
                await update.message.reply_text("⚠️ Could not post to channel, sent here instead.")
        
        except Exception as e:
            error_msg = f"❌ Error generating signal for {symbol if 'symbol' in locals() else 'symbol'}: {str(e)}"
            await update.message.reply_text(error_msg)

    async def price_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get current price for symbol and post to channel"""
        try:
            if not context.args:
                await update.message.reply_text("Usage: /price SYMBOL\nExample: /price AAPL")
                return
            
            symbol = context.args[0].upper().strip()
            
            # Notify user that price is being fetched
            await update.message.reply_text(f"🔄 Fetching price for {symbol}...")
            
            price = fetch_price(symbol)
            
            response = f"💰 <b>{symbol}</b>\n"
            response += f"Current Price: ${price:.2f}\n"
            response += f"👤 Requested by: {update.effective_user.first_name}"
            
            # Post to channel
            success = await self.post_message(response)
            
            if success:
                await update.message.reply_text(f"✅ Price for {symbol} posted to channel!")
            else:
                # Fallback: send to user if channel post fails
                await update.message.reply_text(response, parse_mode='HTML')
                await update.message.reply_text("⚠️ Could not post to channel, sent here instead.")
        
        except Exception as e:
            error_msg = f"❌ Error fetching price for {symbol if 'symbol' in locals() else 'symbol'}: {str(e)}"
            await update.message.reply_text(error_msg)

    async def morningreport_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send morning report immediately"""
        try:
            user = update.effective_user
            if not self._is_admin(user.id):
                await update.message.reply_text("❌ Access denied. Admin privileges required.")
                return
            
            await update.message.reply_text("📊 Generating morning report...")
            await self.send_morning_report()
            await update.message.reply_text("✅ Morning report sent!")
            
        except Exception:
            await update.message.reply_text("❌ Error generating morning report.")

    async def setmorningtime_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set morning report time"""
        try:
            user = update.effective_user
            if not self._is_admin(user.id):
                await update.message.reply_text("❌ Access denied. Admin privileges required.")
                return
            
            if not context.args:
                await update.message.reply_text("Usage: /setmorningtime HH:MM\nExample: /setmorningtime 09:30")
                return
            
            time_str = context.args[0].strip()
            
            if self.timer_manager.set_morning_time(time_str):
                await update.message.reply_text(f"✅ Morning report time set to {time_str}")
            else:
                await update.message.reply_text("❌ Invalid time format. Use HH:MM (24-hour format)")
            
        except Exception:
            await update.message.reply_text("❌ Error setting morning time.")

    async def testtimer_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Toggle test mode for timer"""
        try:
            user = update.effective_user
            if not self._is_admin(user.id):
                await update.message.reply_text("❌ Access denied. Admin privileges required.")
                return
            
            test_mode = self.timer_manager.toggle_test_mode()
            mode_text = "enabled (reports every 2 minutes)" if test_mode else "disabled"
            await update.message.reply_text(f"🔧 Test mode {mode_text}")
            
        except Exception:
            await update.message.reply_text("❌ Error toggling test mode.")

    async def timerstatus_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get timer status"""
        try:
            user = update.effective_user
            if not self._is_admin(user.id):
                await update.message.reply_text("❌ Access denied. Admin privileges required.")
                return
            
            status = self.timer_manager.get_status()
            
            response = f"⏰ <b>Timer Status</b>\n\n"
            response += f"Running: {'✅' if status['running'] else '❌'}\n"
            response += f"Test Mode: {'✅' if status['test_mode'] else '❌'}\n"
            response += f"Morning Time: {status['morning_time']}\n"
            response += f"Timezone Offset: UTC{status['timezone_offset']:+d}\n"
            response += f"Thread Alive: {'✅' if status['thread_alive'] else '❌'}"
            
            await update.message.reply_text(response, parse_mode='HTML')
            
        except Exception:
            await update.message.reply_text("❌ Error getting timer status.")