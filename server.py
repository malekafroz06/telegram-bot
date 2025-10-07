import os
import logging
from flask import Flask, request
from bot import TradingBot
from main import ConfigurationManager, setup_logging

# Set up Flask app
app = Flask(__name__)

# Logging setup
logger = setup_logging()

# Load configuration
config_manager = ConfigurationManager()
config = config_manager.load_config()

# Initialize your bot instance
bot = TradingBot(config)

TOKEN = config["TELEGRAM_TOKEN"]

@app.route("/", methods=["GET"])
def index():
    return "✅ TradePulse Bot is live on Render!", 200

@app.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    update = request.get_json()
    try:
        # Telegram update forwarding logic
        bot.process_update(update)
        return "ok", 200
    except Exception as e:
        logger.error(f"Error processing update: {e}")
        return "error", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
