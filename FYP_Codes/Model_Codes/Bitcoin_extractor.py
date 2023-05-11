import pandas as pd
from binance.client import Client

# Replace with your API keys
api_key = 'API_KEY'
api_secret = 'API_SECRET'

client = Client(api_key, api_secret)

# Define the start and end dates for the data
start_date = '2023-04-29'
end_date = '2023-05-02'

# Get the data from the Binance API
klines = []
interval = Client.KLINE_INTERVAL_1HOUR
symbols = client.get_all_tickers()
symbols = [symbol['symbol'] for symbol in symbols]
if 'BTCUSDT' not in symbols:
    print('BTCUSDT symbol not found')
else:
    symbol = 'BTCUSDT'
    while start_date < end_date:
        temp_date = pd.to_datetime(start_date) + pd.Timedelta(days=1)
        temp_date = temp_date.strftime('%Y-%m-%d')
        temp_klines = client.get_historical_klines(symbol, interval, start_date, temp_date)
        klines += temp_klines
        start_date = temp_date

# Convert the data to a pandas dataframe
columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'num_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
df = pd.DataFrame(klines, columns=columns)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Save the data to a CSV file
df.to_csv('bitcoin_data_2023.csv')

print("All data downloaded and saved to file")