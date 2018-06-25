'''
Created on Jul 25, 2018
Author: @G_Sansigolo
'''

import pandas as pd
import quandl

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['Highlow_Pct'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['Pct_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0


df = df[['Adj. Close', 'Highlow_Pct', 'Pct_Change', 'Adj. Volume']]

print(df.head())
