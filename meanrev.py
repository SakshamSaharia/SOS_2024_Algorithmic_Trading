import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import rcParams


start_date = "2017-01-01"
end_date = "2017-12-31"

# Retrieve SLV (SPDR Gold Shares) data from Yahoo Finance using yfinance
slv2 = yf.download("SLV", start=start_date, end=end_date)
slv=slv2.copy()
day = np.arange(1, len(slv) + 1)
slv['day'] = day
slv.drop(columns=['Adj Close', 'Volume'], inplace = True)
slv = slv[['day', 'Open', 'High', 'Low', 'Close']]

ma = 21
slv['returns'] = np.log(slv["Close"]).diff()
slv['ma'] = slv['Close'].rolling(ma).mean()
slv['ratio'] = slv['Close'] / slv['ma']

percentiles = [5, 10, 50, 90, 95]
p = np.percentile(slv['ratio'].dropna(), percentiles)
plt.rcParams['figure.figsize'] = 12, 6
slv['ratio'].dropna().plot(legend = True)

plt.axhline(p[0], c= (.5,.5,.5), ls='--')
plt.axhline(p[2], c= (.5,.5,.5), ls='--')
plt.axhline(p[-1], c= (.5,.5,.5), ls='--')
plt.savefig("mean1.jpg")
plt.show()

short = p[-1]
long = p[0]
slv['position'] = np.where(slv.ratio > short, -1, np.nan)
slv['position'] = np.where(slv.ratio < long, 1, slv['position'])
slv['position'] = slv['position'].ffill()

slv.position.dropna().plot()
plt.savefig("mean2.jpg")
plt.show()
slv['system_return'] = slv['returns'] * slv['position'].shift()

plt.plot(np.exp(slv['returns']).cumprod(), label='Buy/Hold')
plt.plot(np.exp(slv['system_return']).cumprod(), label='System')
plt.legend()
plt.savefig("mean3.jpg")
plt.show()

