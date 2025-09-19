# Telegram Trading Bot (Core functions)

## Features
- Fetches stock data using `yfinance`
- Generates simple trading signals (5-period vs 20-period SMA crossover)
- Posts signals automatically to a Telegram channel/group
- Admin commands: `/addsymbol`, `/removesymbol`, `/listsymbols`, `/setalert`, `/news`, `/start`

## Requirements
- Python 3.9+
- Create a bot with BotFather and get `TELEGRAM_TOKEN`
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Files
- `main.py` - entrypoint that starts the bot and scheduler
- `bot.py` - Telegram handlers and admin commands
- `signal.py` - signal generation and periodic checks
- `config.example.json` - example configuration (rename to `config.json` and fill values)
- `data/symbols.json` - persisted symbol list (initially empty)
- `data/alerts.json` - persisted alerts configuration
- `requirements.txt` - pip dependencies

## Setup
1. Copy `config.example.json` to `config.json` and fill your `TELEGRAM_TOKEN`, `CHANNEL_ID` (chat id or @channelusername), and `ADMIN_IDS` (list of Telegram user IDs allowed to run admin commands).
2. Run the bot:
   ```bash
   python main.py
   ```
3. Use admin commands in chat with the bot or in the channel (if bot is admin):
   - `/addsymbol SYMBOL` e.g. `/addsymbol AAPL`
   - `/removesymbol SYMBOL`
   - `/listsymbols`
   - `/setalert SYMBOL percent_change` e.g. `/setalert AAPL 2.5` (alert when intraday change exceeds +/- percent)
   - `/news SYMBOL` (attempts to show recent news via yfinance)

## Notes
- This project uses a simple SMA crossover strategy as an example signal generator. It is educational only — NOT financial advice.
- For 24/7 hosting consider deploying on a VPS or server and using a process manager (pm2, systemd) or using a container.
