# data_collection.py
#
# November 20, 2017
import pandas as pd
import pandas_datareader as web
from pandas_datareader._utils import RemoteDataError


# File to read tickers/sector from
SP500_FILE = 'constituents.csv'

# API to call for time series data ('yahoo' or 'google')
TS_API = 'yahoo'

# Number of times to try to get price data before giving up
MAX_ERRS = 5

# Get times series data frame for collection of symbols for given
# time period. The [priceType] parameter will be the price listed for
# each time-tick: 'Open', 'Close', 'Adj Close', 'High', 'Low', or 'Volume'
def get_ts(symbols, startDate, endDate, priceType):
    stock_dict = {}
    total_errs = 0
    for i,symbol in enumerate(symbols):
        error_count = 0
        while True:
            if error_count >= MAX_ERRS:
                print("Errored {} times trying to retrieve {}".format(MAX_ERRS, symbol))
                total_errs += 1
                break
            try:
                symbol_ts = web.DataReader(symbol, TS_API, startDate, endDate)
                stock_dict[symbol] = symbol_ts[priceType]
                print("{}. Retrieved {}".format(i,symbol))
                break
            except RemoteDataError:
                error_count += 1
    print("Skipped {} stocks".format(total_errs))
    stocks = pd.DataFrame(stock_dict)
    return stocks

# startDate and endDate should be datetime objects
def get_SP500_ts(startDate, endDate):
    SP500 = pd.read_csv(SP500_FILE)
    SP500_ts = get_ts(SP500['Symbol'], startDate, endDate, 'Open')
    return SP500_ts
