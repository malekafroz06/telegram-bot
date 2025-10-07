from flask import Flask, request
from bot import TradingBot
import os

TOKEN = os.environ.get('TELEGRAM_TOKEN')
bot = TradingBot({'TELEGRAM_TOKEN': TOKEN})
app = Flask(__name__)

@app.route(f"/{TOKEN}", methods=['POST'])
def webhook():
    update = request.get_json(force=True)
    bot.handle_update(update)  # Implement handle_update in TradingBot
    return "OK"

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8443))
    bot.set_webhook(f"https://telegram-bot-45sk.onrender.com/{TOKEN}")
    app.run(host="0.0.0.0", port=PORT)
