import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

start_date = "2017-01-01"
end_date = "2017-12-31"

# Retrieve SLV (Silver Shares) data from Yahoo Finance using yfinance
slv2 = yf.download("SLV", start=start_date, end=end_date)
slv=slv2.copy()
day = np.arange(1, len(slv) + 1)
slv['day'] = day
slv.drop(columns=['Adj Close', 'Volume'], inplace = True)
slv = slv[['day', 'Open', 'High', 'Low', 'Close']]
slv.loc[:, '9-day'] = slv['Close'].ewm(span=9, adjust=False).mean()
slv.loc[:, '21-day'] = slv['Close'].ewm(span=21, adjust=False).mean()
slv.loc[:,'MACD'] = slv['9-day'] - slv['21-day']
# Calculate the Signal Line (9-period EMA of the MACD-line)
slv.loc[:,'Signal_Line'] = slv['MACD'].ewm(span=9, adjust=False).mean()
slv['signal'] = np.where(slv['MACD'] > slv['Signal_Line'], 1, 0)
slv['signal'] = np.where(slv['MACD'] < slv['Signal_Line'], -1, slv['signal'])
slv.dropna(inplace=True)
slv['return'] = np.log(slv['Close']).diff()
slv['system_return'] = slv['signal'] * slv['return']
slv['entry'] = slv.signal.diff()
# slv.head()
print(slv.head())

plt.rcParams['figure.figsize'] = 12, 6
plt.grid(True, alpha = .3)
plt.plot(slv.iloc[-252:]['Close'], label = 'SLV')
plt.plot(slv.iloc[-252:]['MACD'], label = 'MACD')
plt.plot(slv.iloc[-252:]['Signal_Line'], label = 'Signal_Line')
plt.plot(slv[-252:].loc[slv.entry == 2].index, slv[-252:]['MACD'][slv.entry == 2], '^',
         color = 'g', markersize = 12)
plt.plot(slv[-252:].loc[slv.entry == -2].index, slv[-252:]['Signal_Line'][slv.entry == -2], 'v',
         color = 'r', markersize = 12)
plt.legend(loc=2)
plt.savefig("macd1.jpg")
plt.show()
plt.plot(np.exp(slv['return']).cumprod(), label='Buy/Hold')
plt.plot(np.exp(slv['system_return']).cumprod(), label='System')
plt.legend(loc=2)
plt.grid(True, alpha=.3)
plt.savefig("macd2.jpg")
plt.show()