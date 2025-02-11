import ccxt
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime

# Initialize the exchange
exchange = ccxt.binance({
    'rateLimit': 1200,
    'enableRateLimit': True,
})

def fetch_data(symbol, timeframe, limit=500):
    """Fetch historical data from Binance."""
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df

def apply_indicators(df):
    """Apply a comprehensive set of technical indicators to the dataframe."""
    # RSI
    df['RSI'] = ta.rsi(df['close'], length=14)

    # MACD
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if isinstance(macd, pd.DataFrame):
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
        df['MACD_hist'] = macd['MACDh_12_26_9']

    # Bollinger Bands
    bb = ta.bbands(df['close'], length=20, std=2)
    if isinstance(bb, pd.DataFrame):
        df['BB_upper'] = bb['BBU_20_2.0']
        df['BB_middle'] = bb['BBM_20_2.0']
        df['BB_lower'] = bb['BBL_20_2.0']

    # ATR
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    # ADX
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    if isinstance(adx, pd.DataFrame):
        df['ADX'] = adx['ADX_14']
        df['DI_plus'] = adx['DMP_14']
        df['DI_minus'] = adx['DMN_14']

    # Stochastic Oscillator
    stoch = ta.stoch(df['high'], df['low'], df['close'])
    if isinstance(stoch, pd.DataFrame):
        df['Stoch_K'] = stoch['STOCHk_14_3_3']
        df['Stoch_D'] = stoch['STOCHd_14_3_3']

    # VWAP
    df['VWAP'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])

    # CCI
    df['CCI'] = ta.cci(df['high'], df['low'], df['close'], length=20)

    # ROC
    df['ROC'] = ta.roc(df['close'], length=12)

    # MFI
    df['MFI'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)

    # EMA
    df['EMA_50'] = ta.ema(df['close'], length=50)
    df['EMA_200'] = ta.ema(df['close'], length=200)

    # Williams %R
    df['Williams_%R'] = ta.willr(df['high'], df['low'], df['close'], length=14)

    # Fill missing values
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

def detect_trend(df):
    """Detect market trend using EMA crossover."""
    if df['EMA_50'].iloc[-1] > df['EMA_200'].iloc[-1]:
        return "UPTREND"
    elif df['EMA_50'].iloc[-1] < df['EMA_200'].iloc[-1]:
        return "DOWNTREND"
    else:
        return "RANGE"

def generate_composite_signal(df):
    """Generate a composite signal based on a scoring system."""
    try:
        trend = detect_trend(df)
        close = df['close'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        macd_signal = df['MACD_signal'].iloc[-1]
        bb_upper = df['BB_upper'].iloc[-1]
        bb_lower = df['BB_lower'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        adx = df['ADX'].iloc[-1]
        stoch_k = df['Stoch_K'].iloc[-1]
        stoch_d = df['Stoch_D'].iloc[-1]
        cci = df['CCI'].iloc[-1]
        roc = df['ROC'].iloc[-1]
        mfi = df['MFI'].iloc[-1]
        williams_r = df['Williams_%R'].iloc[-1]

        score = 0

        # RSI
        if rsi < 30:
            score += 1
        elif rsi > 70:
            score -= 1

        # MACD
        if macd > macd_signal:
            score += 1
        elif macd < macd_signal:
            score -= 1

        # Bollinger Bands
        if close < bb_lower:
            score += 1
        elif close > bb_upper:
            score -= 1

        # Stochastic Oscillator
        if stoch_k < 20 and stoch_d < 20:
            score += 1
        elif stoch_k > 80 and stoch_d > 80:
            score -= 1

        # CCI
        if cci < -100:
            score += 1
        elif cci > 100:
            score -= 1

        # MFI
        if mfi < 20:
            score += 1
        elif mfi > 80:
            score -= 1

        # Williams %R
        if williams_r < -80:
            score += 1
        elif williams_r > -20:
            score -= 1

        # Trend adjustment
        if trend == "UPTREND" and score > 0:
            return "BUY"
        elif trend == "DOWNTREND" and score < 0:
            return "SELL"
        else:
            return "HOLD"
    except Exception as e:
        print(f"Error in signal generation: {e}")
        return "HOLD"

def main():
    symbol = 'BTC/USDT'
    timeframe = '4h'

    print("Starting the signal bot...")

    while True:
        try:
            # Fetch and process data
            df = fetch_data(symbol, timeframe)
            df = apply_indicators(df)

            # Generate signal
            signal = generate_composite_signal(df)

            # Print the latest signal with the current timestamp
            print(f"{datetime.now()} - Latest Signal: {signal}")

        except Exception as e:
            print(f"Error: {e}")

        # Sleep for 60 seconds
        time.sleep(60)

if __name__ == "__main__":
    main()
