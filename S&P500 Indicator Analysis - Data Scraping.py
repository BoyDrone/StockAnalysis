# -*- coding: utf-8 -*-
#https://pythonprogramming.net/stock-data-manipulation-python-programming-for-finance/?completed=/handling-stock-data-graphing-python-programming-for-finance/

from datetime import date
import datetime as dt
from datetime import timedelta
import os.path
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
from pandas_datareader import data as pdr
import pickle
import numpy as np
import urllib
import os #this is to create folders
import bs4 as bs #this is for beautiful soup
import requests #grabs source code from websites
import fix_yahoo_finance as yf
from matplotlib.pyplot import figure

yf.pdr_override

start_time = dt.datetime.now()

# with open("sp500tickers.pickle", "rb") as f:
#     tickers = pickle.load(f)

if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data'): #creates folder if the folder doesn't exist
        os.makedirs('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data')
if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs'): #creates folder if the folder doesn't exist
        os.makedirs('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs')        
#https://datatofish.com/if-condition-in-pandas-dataframe/
#ran a mix of this website
#https://stackoverflow.com/questions/57006437/calculate-rsi-indicator-from-pandas-dataframe
#as well as a bit of a change to the mean to make life easier

def save_sp500_tickers():
    print("Gathering S&P ticker list. (1 of 11)")
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.replace('.', '-')
        ticker = ticker[:-1]
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers

save_sp500_tickers()

def get_data_from_yahoo(reload_sp500=False):
    print("Updating price database. (2 of 11)")
    print(dt.datetime.now())
    startyear = 1970
    start = dt.datetime(startyear, 1, 1)
    end = dt.datetime.now()
    MAX_ATTEMPTS = 1

    df_tickerpull = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/yahootickersymbolsUSA.csv')
    df_tickerpull = df_tickerpull.set_index('ticker') #, inplace=True)
    print(df_tickerpull)
    tickers = df_tickerpull.index #this outputs the correct weight
    #print(tickers)
    for ticker in tickers:
        #print(df_tickerpull)
        #print(ticker)
        try:
            print("Looking up data for " + ticker)
            if os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):
                df_current = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker), parse_dates=True, index_col=0)
                print("File exists for " + ticker + ", checking for new data")
                for attempt in range(MAX_ATTEMPTS):
                    print('Trying to fetch')
                    if df_current.index[-1] < date.today():
                        print("Adding new data to " + ticker)
                        start = df_current.index[-1] + timedelta(days=1)
                        df_new = web.DataReader(ticker, 'yahoo', start, end)
                        df_new.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker), mode='a', header=False)
                    else:
                        print("No new data for " +  ticker)
            else:
                try:
                    print('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
                    start_20year = dt.datetime(startyear, 1, 1)
                    df = web.DataReader(ticker, 'yahoo', start_20year, end)
                except Exception as e:
                    open('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/NO DATA FOR THIS TICKER ON YAHOO ' + ticker + '.txt', 'a').close()
                    df = []
                df.to_csv('Reference_data/stock_dfs/{}.csv'.format(ticker))
        except Exception as e:
            print("Data does not exist for " + ticker + " from yahoo finance")
            open('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/NO DATA FOR THIS TICKER ON YAHOO ' + ticker + '.txt', 'a').close()
            print(e)

get_data_from_yahoo()
         
def save_sp500_tickersslickchart(): #this runs but the file is not used at the moment, it will be used in the future for weighting of the data
    print("Gathering S&P500 weighting. (3 of 11)")
    resp = requests.get('https://www.slickcharts.com/sp500')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'table'})
    df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/sp500tickersslickchart.csv', parse_dates=True, index_col=0)
    #replace . with - in ticker list

    records = []
    for row in table.findAll('tr')[1:]:
        company = row.findAll('td')[1].text
        ticker = row.findAll('td')[2].text
        weight = row.findAll('td')[3].text
        price = row.findAll(('td'), {'class': 'text-nowrap'})[0].text
        records.append((company, ticker, weight, price))
    df = pd.DataFrame(records, columns=['company','ticker','weight','price'])
    df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/sp500tickersslickchart.csv', index=False, encoding='utf-8')

def indicator_analysis():
    print("Adding indicators & markers to stocks. (4 of 11)")
    i = 0
    df_weightpull = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/sp500tickersslickchart.csv', index_col=1)
    icount = df_weightpull.count()
    for ticker in df_weightpull.index:
        weight = (df_weightpull.loc[ticker, 'weight'])/100 #this outputs the correct weight
        i = i + 1
        print(((i/icount)*100) + " %")
        print(ticker)
        # for count, ticker in enumerate(tickers):
        print("File for " + ticker + " exists, expanding data & adding indicators")

        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        else:
            df = pd.read_csv('Reference_data/stock_dfs/{}.csv'.format(ticker))
            print("opened file for " + ticker)
            df.set_index('Date', inplace=True)
            df['30ma'] = df['Adj Close'].rolling(window=30,min_periods=0).mean()
            df['50ma'] = df['Adj Close'].rolling(window=50,min_periods=0).mean()
            df['100ma'] = df['Adj Close'].rolling(window=100,min_periods=0).mean()
            df['150ma'] = df['Adj Close'].rolling(window=150,min_periods=0).mean()
            df['200ma'] = df['Adj Close'].rolling(window=200,min_periods=0).mean()
            df['change'] = df['Adj Close'].diff()
            df['gain'] = df.change.mask(df.change < 0, 0.0)
            df['loss'] = -df.change.mask(df.change > 0, -0.0)
            n = 14
            df['avg_gain'] = df['gain'].rolling(window=n,min_periods=0).mean()
            df['avg_loss'] = df['loss'].rolling(window=n,min_periods=0).mean()
            df['rs'] = df.avg_gain / df.avg_loss
            df['rsi_14'] = 100 - (100 / (1 + df.rs))
            
            weight = (df_weightpull.loc[ticker, 'weight'])/100 #this outputs the correct weight
            
            #the following counts when the price is below given moving average
            df.loc[df['30ma'] >= df['Adj Close'], '30ma_O1/U0_weighted'] = weight #if price is below 30 dma, ticker market as a count
            df.loc[df['30ma'] >= df['Adj Close'], '30ma_O1/U0_count'] = 1 #if price is below 30 dma, ticker market as a count
            df.loc[df['50ma'] >= df['Adj Close'], '50ma_O1/U0_weighted'] = weight #if price is below 50 dma, ticker market as a count
            df.loc[df['50ma'] >= df['Adj Close'], '50ma_O1/U0_count'] = 1 #if price is below 50 dma, ticker market as a count
            df.loc[df['100ma'] >= df['Adj Close'], '100ma_O1/U0_weighted'] = weight #if price is below 100 dma, ticker market as a count
            df.loc[df['100ma'] >= df['Adj Close'], '100ma_O1/U0_count'] = 1 #if price is below 100 dma, ticker market as a count
            df.loc[df['150ma'] >= df['Adj Close'], '150ma_O1/U0_weighted'] = weight #if price is below 150 dma, ticker market as a count
            df.loc[df['150ma'] >= df['Adj Close'], '150ma_O1/U0_count'] = 1 #if price is below 150 dma, ticker market as a count
            df.loc[df['200ma'] >= df['Adj Close'], '200ma_O1/U0_weighted'] = weight #if price is below 200 dma, ticker market as a count
            df.loc[df['200ma'] >= df['Adj Close'], '200ma_O1/U0_count'] = 1 #if price is below 200 dma, ticker market as a count
            
            #the following counts how many stocks in the S&P 500 have RSI above given thresholds and adds respective weight
            df.loc[df['rsi_14'] >= 30, 'rsi_14_O1/U0_30_weighted'] = weight
            df.loc[df['rsi_14'] >= 35, 'rsi_14_O1/U0_35_weighted'] = weight
            df.loc[df['rsi_14'] >= 40, 'rsi_14_O1/U0_40_weighted'] = weight
            df.loc[df['rsi_14'] >= 45, 'rsi_14_O1/U0_45_weighted'] = weight
            df.loc[df['rsi_14'] >= 50, 'rsi_14_O1/U0_50_weighted'] = weight
            df.loc[df['rsi_14'] >= 55, 'rsi_14_O1/U0_55_weighted'] = weight
            df.loc[df['rsi_14'] >= 60, 'rsi_14_O1/U0_60_weighted'] = weight
            df.loc[df['rsi_14'] >= 65, 'rsi_14_O1/U0_65_weighted'] = weight
            df.loc[df['rsi_14'] >= 70, 'rsi_14_O1/U0_70_weighted'] = weight
            
            #the following counts how many stocks in the S&P 500 have RSI above given thresholds
            df.loc[df['rsi_14'] >= 30, 'rsi_14_O1/U0_30_count'] = 1
            df.loc[df['rsi_14'] >= 35, 'rsi_14_O1/U0_35_count'] = 1
            df.loc[df['rsi_14'] >= 40, 'rsi_14_O1/U0_40_count'] = 1
            df.loc[df['rsi_14'] >= 45, 'rsi_14_O1/U0_45_count'] = 1
            df.loc[df['rsi_14'] >= 50, 'rsi_14_O1/U0_50_count'] = 1
            df.loc[df['rsi_14'] >= 55, 'rsi_14_O1/U0_55_count'] = 1
            df.loc[df['rsi_14'] >= 60, 'rsi_14_O1/U0_60_count'] = 1
            df.loc[df['rsi_14'] >= 65, 'rsi_14_O1/U0_65_count'] = 1
            df.loc[df['rsi_14'] >= 70, 'rsi_14_O1/U0_70_count'] = 1
            
            df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))

def compile_data_30ma_weighted():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
    
            df.rename(columns={'30ma_O1/U0_weighted': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_count',
                     '50ma_O1/U0_weighted','50ma_O1/U0_count','100ma_O1/U0_weighted',
                     '100ma_O1/U0_count','150ma_O1/U0_weighted','150ma_O1/U0_count',
                     '200ma_O1/U0_weighted','200ma_O1/U0_count','rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted','rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_45_weighted','rsi_14_O1/U0_50_weighted',
                     'rsi_14_O1/U0_55_weighted','rsi_14_O1/U0_60_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count','rsi_14_O1/U0_35_count',
                     'rsi_14_O1/U0_40_count','rsi_14_O1/U0_45_count','rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count','rsi_14_O1/U0_60_count','rsi_14_O1/U0_65_count',
                     'rsi_14_O1/U0_70_count'], 1, inplace=True)
            
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
    
    main_df['weight30'] = main_df.sum(axis=1)
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP50030MA_weighted.csv')

def compile_data_30ma_count():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
    
            df.rename(columns={'30ma_O1/U0_count': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','50ma_O1/U0_count','100ma_O1/U0_weighted',
                     '100ma_O1/U0_count','150ma_O1/U0_weighted','150ma_O1/U0_count',
                     '200ma_O1/U0_weighted','200ma_O1/U0_count','rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted','rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_45_weighted','rsi_14_O1/U0_50_weighted',
                     'rsi_14_O1/U0_55_weighted','rsi_14_O1/U0_60_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count','rsi_14_O1/U0_35_count',
                     'rsi_14_O1/U0_40_count','rsi_14_O1/U0_45_count','rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count','rsi_14_O1/U0_60_count','rsi_14_O1/U0_65_count',
                     'rsi_14_O1/U0_70_count'], 1, inplace=True)
            
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
    
    main_df['count30'] = main_df.sum(axis=1)
    main_df['countperc30'] = main_df['count30']/(count+1)
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP50030MA_count.csv')

def compile_data_50ma_weighted():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
    
            df.rename(columns={'50ma_O1/U0_weighted': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_count',
                     '30ma_O1/U0_weighted','50ma_O1/U0_count','100ma_O1/U0_weighted',
                     '100ma_O1/U0_count','150ma_O1/U0_weighted','150ma_O1/U0_count',
                     '200ma_O1/U0_weighted','200ma_O1/U0_count','rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted','rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_45_weighted','rsi_14_O1/U0_50_weighted',
                     'rsi_14_O1/U0_55_weighted','rsi_14_O1/U0_60_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count','rsi_14_O1/U0_35_count',
                     'rsi_14_O1/U0_40_count','rsi_14_O1/U0_45_count','rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count','rsi_14_O1/U0_60_count','rsi_14_O1/U0_65_count',
                     'rsi_14_O1/U0_70_count'], 1, inplace=True)
            
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
    
    main_df['weight50'] = main_df.sum(axis=1)
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP50050MA_weighted.csv')

def compile_data_50ma_count():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
    
            df.rename(columns={'50ma_O1/U0_count': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','30ma_O1/U0_count','100ma_O1/U0_weighted',
                     '100ma_O1/U0_count','150ma_O1/U0_weighted','150ma_O1/U0_count',
                     '200ma_O1/U0_weighted','200ma_O1/U0_count','rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted','rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_45_weighted','rsi_14_O1/U0_50_weighted',
                     'rsi_14_O1/U0_55_weighted','rsi_14_O1/U0_60_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count','rsi_14_O1/U0_35_count',
                     'rsi_14_O1/U0_40_count','rsi_14_O1/U0_45_count','rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count','rsi_14_O1/U0_60_count','rsi_14_O1/U0_65_count',
                     'rsi_14_O1/U0_70_count'], 1, inplace=True)
            
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
    
    main_df['count50'] = main_df.sum(axis=1)
    main_df['countperc50'] = main_df['count50']/(count+1)
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP50050MA_count.csv')
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP50050MA_count.csv')

def compile_data_100ma_weighted():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
    
            df.rename(columns={'100ma_O1/U0_weighted': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_count',
                     '30ma_O1/U0_weighted','50ma_O1/U0_count','50ma_O1/U0_weighted',
                     '100ma_O1/U0_count','150ma_O1/U0_weighted','150ma_O1/U0_count',
                     '200ma_O1/U0_weighted','200ma_O1/U0_count','rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted','rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_45_weighted','rsi_14_O1/U0_50_weighted',
                     'rsi_14_O1/U0_55_weighted','rsi_14_O1/U0_60_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count','rsi_14_O1/U0_35_count',
                     'rsi_14_O1/U0_40_count','rsi_14_O1/U0_45_count','rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count','rsi_14_O1/U0_60_count','rsi_14_O1/U0_65_count',
                     'rsi_14_O1/U0_70_count'], 1, inplace=True)
            
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
    
    main_df['weight100'] = main_df.sum(axis=1)
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500100MA_weighted.csv')

def compile_data_100ma_count():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
    
            df.rename(columns={'100ma_O1/U0_count': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','30ma_O1/U0_count','100ma_O1/U0_weighted',
                     '50ma_O1/U0_count','150ma_O1/U0_weighted','150ma_O1/U0_count',
                     '200ma_O1/U0_weighted','200ma_O1/U0_count','rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted','rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_45_weighted','rsi_14_O1/U0_50_weighted',
                     'rsi_14_O1/U0_55_weighted','rsi_14_O1/U0_60_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count','rsi_14_O1/U0_35_count',
                     'rsi_14_O1/U0_40_count','rsi_14_O1/U0_45_count','rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count','rsi_14_O1/U0_60_count','rsi_14_O1/U0_65_count',
                     'rsi_14_O1/U0_70_count'], 1, inplace=True)
            
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
    
    main_df['count100'] = main_df.sum(axis=1)
    main_df['countperc100'] = main_df['count100']/(count+1)
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500100MA_count.csv')

def compile_data_150ma_weighted():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
    
            df.rename(columns={'150ma_O1/U0_weighted': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_count',
                     '30ma_O1/U0_weighted','50ma_O1/U0_count','50ma_O1/U0_weighted',
                     '100ma_O1/U0_count','100ma_O1/U0_weighted','150ma_O1/U0_count',
                     '200ma_O1/U0_weighted','200ma_O1/U0_count','rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted','rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_45_weighted','rsi_14_O1/U0_50_weighted',
                     'rsi_14_O1/U0_55_weighted','rsi_14_O1/U0_60_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count','rsi_14_O1/U0_35_count',
                     'rsi_14_O1/U0_40_count','rsi_14_O1/U0_45_count','rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count','rsi_14_O1/U0_60_count','rsi_14_O1/U0_65_count',
                     'rsi_14_O1/U0_70_count'], 1, inplace=True)
            
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
    
    main_df['weight150'] = main_df.sum(axis=1)
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500150MA_weighted.csv')

def compile_data_150ma_count():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
    
            df.rename(columns={'150ma_O1/U0_count': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','30ma_O1/U0_count','100ma_O1/U0_weighted',
                     '50ma_O1/U0_count','150ma_O1/U0_weighted','100ma_O1/U0_count',
                     '200ma_O1/U0_weighted','200ma_O1/U0_count','rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted','rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_45_weighted','rsi_14_O1/U0_50_weighted',
                     'rsi_14_O1/U0_55_weighted','rsi_14_O1/U0_60_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count','rsi_14_O1/U0_35_count',
                     'rsi_14_O1/U0_40_count','rsi_14_O1/U0_45_count','rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count','rsi_14_O1/U0_60_count','rsi_14_O1/U0_65_count',
                     'rsi_14_O1/U0_70_count'], 1, inplace=True)
            
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
    
    main_df['count150'] = main_df.sum(axis=1)
    main_df['countperc150'] = main_df['count150']/(count+1)
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500150MA_count.csv')

def compile_data_200ma_weighted():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
    
            df.rename(columns={'200ma_O1/U0_weighted': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_count',
                     '30ma_O1/U0_weighted','50ma_O1/U0_count','50ma_O1/U0_weighted',
                     '100ma_O1/U0_count','100ma_O1/U0_weighted','150ma_O1/U0_count',
                     '150ma_O1/U0_weighted','200ma_O1/U0_count','rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted','rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_45_weighted','rsi_14_O1/U0_50_weighted',
                     'rsi_14_O1/U0_55_weighted','rsi_14_O1/U0_60_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count','rsi_14_O1/U0_35_count',
                     'rsi_14_O1/U0_40_count','rsi_14_O1/U0_45_count','rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count','rsi_14_O1/U0_60_count','rsi_14_O1/U0_65_count',
                     'rsi_14_O1/U0_70_count'], 1, inplace=True)

            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
    
    main_df['weight200'] = main_df.sum(axis=1)
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500200MA_weighted.csv')

def compile_data_200ma_count():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
    
            df.rename(columns={'200ma_O1/U0_count': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','30ma_O1/U0_count','100ma_O1/U0_weighted',
                     '50ma_O1/U0_count','150ma_O1/U0_weighted','100ma_O1/U0_count',
                     '200ma_O1/U0_weighted','150ma_O1/U0_count','rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted','rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_45_weighted','rsi_14_O1/U0_50_weighted',
                     'rsi_14_O1/U0_55_weighted','rsi_14_O1/U0_60_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count','rsi_14_O1/U0_35_count',
                     'rsi_14_O1/U0_40_count','rsi_14_O1/U0_45_count','rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count','rsi_14_O1/U0_60_count','rsi_14_O1/U0_65_count',
                     'rsi_14_O1/U0_70_count'], 1, inplace=True)
            
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
    
    main_df['count200'] = main_df.sum(axis=1)
    main_df['countperc200'] = main_df['count200']/(count+1)
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500200MA_count.csv')

def compile_data_rsi_above30_weighted():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
        
            df.rename(columns={'rsi_14_O1/U0_30_weighted': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','30ma_O1/U0_count','100ma_O1/U0_weighted',
                     '50ma_O1/U0_count','150ma_O1/U0_weighted','100ma_O1/U0_count',
                     '200ma_O1/U0_weighted', '200ma_O1/U0_count', '150ma_O1/U0_count',
                     'rsi_14_O1/U0_35_weighted',
                     'rsi_14_O1/U0_40_weighted', 'rsi_14_O1/U0_45_weighted',
                     'rsi_14_O1/U0_50_weighted', 'rsi_14_O1/U0_55_weighted',
                     'rsi_14_O1/U0_60_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count',
                     'rsi_14_O1/U0_35_count', 'rsi_14_O1/U0_40_count',
                     'rsi_14_O1/U0_45_count', 'rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count', 'rsi_14_O1/U0_60_count',
                     'rsi_14_O1/U0_65_count', 'rsi_14_O1/U0_70_count'], 1, inplace=True)
        
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        
            #if count % 10 == 0:
            #    print(count)
    
    main_df['RSIweight30'] = main_df.sum(axis=1)
    main_df['LowerRSI'] = 30
    main_df['UpperRSI'] = 70
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI30_weighted.csv')

def compile_data_rsi_above35_weighted():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
        
            df.rename(columns={'rsi_14_O1/U0_35_weighted': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','30ma_O1/U0_count','100ma_O1/U0_weighted',
                     '50ma_O1/U0_count','150ma_O1/U0_weighted','100ma_O1/U0_count',
                     '200ma_O1/U0_weighted', '200ma_O1/U0_count', '150ma_O1/U0_count',
                     'rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_40_weighted', 'rsi_14_O1/U0_45_weighted',
                     'rsi_14_O1/U0_50_weighted', 'rsi_14_O1/U0_55_weighted',
                     'rsi_14_O1/U0_60_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count',
                     'rsi_14_O1/U0_35_count', 'rsi_14_O1/U0_40_count',
                     'rsi_14_O1/U0_45_count', 'rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count', 'rsi_14_O1/U0_60_count',
                     'rsi_14_O1/U0_65_count', 'rsi_14_O1/U0_70_count'], 1, inplace=True)
        
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        
            #if count % 10 == 0:
            #    print(count)
    
    main_df['RSIweight35'] = main_df.sum(axis=1)
    main_df['LowerRSI'] = 30
    main_df['UpperRSI'] = 70
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI35_weighted.csv')

def compile_data_rsi_above40_weighted():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
        
            df.rename(columns={'rsi_14_O1/U0_40_weighted': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','30ma_O1/U0_count','100ma_O1/U0_weighted',
                     '50ma_O1/U0_count','150ma_O1/U0_weighted','100ma_O1/U0_count',
                     '200ma_O1/U0_weighted', '200ma_O1/U0_count', '150ma_O1/U0_count',
                     'rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted', 'rsi_14_O1/U0_45_weighted',
                     'rsi_14_O1/U0_50_weighted', 'rsi_14_O1/U0_55_weighted',
                     'rsi_14_O1/U0_60_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count',
                     'rsi_14_O1/U0_35_count', 'rsi_14_O1/U0_40_count',
                     'rsi_14_O1/U0_45_count', 'rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count', 'rsi_14_O1/U0_60_count',
                     'rsi_14_O1/U0_65_count', 'rsi_14_O1/U0_70_count'], 1, inplace=True)
        
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        
            #if count % 10 == 0:
            #    print(count)
    
    main_df['RSIweight40'] = main_df.sum(axis=1)
    main_df['LowerRSI'] = 30
    main_df['UpperRSI'] = 70
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI40_weighted.csv')

def compile_data_rsi_above45_weighted():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
        
            df.rename(columns={'rsi_14_O1/U0_45_weighted': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','30ma_O1/U0_count','100ma_O1/U0_weighted',
                     '50ma_O1/U0_count','150ma_O1/U0_weighted','100ma_O1/U0_count',
                     '200ma_O1/U0_weighted', '200ma_O1/U0_count', '150ma_O1/U0_count',
                     'rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted', 'rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_50_weighted', 'rsi_14_O1/U0_55_weighted',
                     'rsi_14_O1/U0_60_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count',
                     'rsi_14_O1/U0_35_count', 'rsi_14_O1/U0_40_count',
                     'rsi_14_O1/U0_45_count', 'rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count', 'rsi_14_O1/U0_60_count',
                     'rsi_14_O1/U0_65_count', 'rsi_14_O1/U0_70_count'], 1, inplace=True)
        
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        
            #if count % 10 == 0:
            #    print(count)
    
    main_df['RSIweight45'] = main_df.sum(axis=1)
    main_df['LowerRSI'] = 30
    main_df['UpperRSI'] = 70
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI45_weighted.csv')

def compile_data_rsi_above50_weighted():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
        
            df.rename(columns={'rsi_14_O1/U0_50_weighted': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','30ma_O1/U0_count','100ma_O1/U0_weighted',
                     '50ma_O1/U0_count','150ma_O1/U0_weighted','100ma_O1/U0_count',
                     '200ma_O1/U0_weighted', '200ma_O1/U0_count', '150ma_O1/U0_count',
                     'rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted', 'rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_45_weighted', 'rsi_14_O1/U0_55_weighted',
                     'rsi_14_O1/U0_60_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count',
                     'rsi_14_O1/U0_35_count', 'rsi_14_O1/U0_40_count',
                     'rsi_14_O1/U0_45_count', 'rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count', 'rsi_14_O1/U0_60_count',
                     'rsi_14_O1/U0_65_count', 'rsi_14_O1/U0_70_count'], 1, inplace=True)
        
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        
            #if count % 10 == 0:
            #    print(count)
    
    main_df['RSIweight50'] = main_df.sum(axis=1)
    main_df['LowerRSI'] = 30
    main_df['UpperRSI'] = 70
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI50_weighted.csv')

def compile_data_rsi_above55_weighted():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
        
            df.rename(columns={'rsi_14_O1/U0_55_weighted': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','30ma_O1/U0_count','100ma_O1/U0_weighted',
                     '50ma_O1/U0_count','150ma_O1/U0_weighted','100ma_O1/U0_count',
                     '200ma_O1/U0_weighted', '200ma_O1/U0_count', '150ma_O1/U0_count',
                     'rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted', 'rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_45_weighted', 'rsi_14_O1/U0_50_weighted',
                     'rsi_14_O1/U0_60_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count',
                     'rsi_14_O1/U0_35_count', 'rsi_14_O1/U0_40_count',
                     'rsi_14_O1/U0_45_count', 'rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count', 'rsi_14_O1/U0_60_count',
                     'rsi_14_O1/U0_65_count', 'rsi_14_O1/U0_70_count'], 1, inplace=True)
        
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        
            #if count % 10 == 0:
            #    print(count)
    
    main_df['RSIweight55'] = main_df.sum(axis=1)
    main_df['LowerRSI'] = 30
    main_df['UpperRSI'] = 70
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI55_weighted.csv')

def compile_data_rsi_above60_weighted():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
        
            df.rename(columns={'rsi_14_O1/U0_60_weighted': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','30ma_O1/U0_count','100ma_O1/U0_weighted',
                     '50ma_O1/U0_count','150ma_O1/U0_weighted','100ma_O1/U0_count',
                     '200ma_O1/U0_weighted', '200ma_O1/U0_count', '150ma_O1/U0_count',
                     'rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted', 'rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_45_weighted', 'rsi_14_O1/U0_50_weighted',
                     'rsi_14_O1/U0_55_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count',
                     'rsi_14_O1/U0_35_count', 'rsi_14_O1/U0_40_count',
                     'rsi_14_O1/U0_45_count', 'rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count', 'rsi_14_O1/U0_60_count',
                     'rsi_14_O1/U0_65_count', 'rsi_14_O1/U0_70_count'], 1, inplace=True)
        
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        
            #if count % 10 == 0:
            #    print(count)
    
    main_df['RSIweight60'] = main_df.sum(axis=1)
    main_df['LowerRSI'] = 30
    main_df['UpperRSI'] = 70
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI60_weighted.csv')

def compile_data_rsi_above65_weighted():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
        
            df.rename(columns={'rsi_14_O1/U0_65_weighted': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','30ma_O1/U0_count','100ma_O1/U0_weighted',
                     '50ma_O1/U0_count','150ma_O1/U0_weighted','100ma_O1/U0_count',
                     '200ma_O1/U0_weighted', '200ma_O1/U0_count', '150ma_O1/U0_count',
                     'rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted', 'rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_45_weighted', 'rsi_14_O1/U0_50_weighted',
                     'rsi_14_O1/U0_55_weighted','rsi_14_O1/U0_60_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count',
                     'rsi_14_O1/U0_35_count', 'rsi_14_O1/U0_40_count',
                     'rsi_14_O1/U0_45_count', 'rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count', 'rsi_14_O1/U0_60_count',
                     'rsi_14_O1/U0_65_count', 'rsi_14_O1/U0_70_count'], 1, inplace=True)
        
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        
            #if count % 10 == 0:
            #    print(count)
    
    main_df['RSIweight65'] = main_df.sum(axis=1)
    main_df['LowerRSI'] = 30
    main_df['UpperRSI'] = 70
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI65_weighted.csv')

def compile_data_rsi_above70_weighted():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
        
            df.rename(columns={'rsi_14_O1/U0_70_weighted': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','30ma_O1/U0_count','100ma_O1/U0_weighted',
                     '50ma_O1/U0_count','150ma_O1/U0_weighted','100ma_O1/U0_count',
                     '200ma_O1/U0_weighted', '200ma_O1/U0_count', '150ma_O1/U0_count',
                     'rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted', 'rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_45_weighted', 'rsi_14_O1/U0_50_weighted',
                     'rsi_14_O1/U0_55_weighted','rsi_14_O1/U0_60_weighted',
                     'rsi_14_O1/U0_65_weighted','rsi_14_O1/U0_30_count',
                     'rsi_14_O1/U0_35_count', 'rsi_14_O1/U0_40_count',
                     'rsi_14_O1/U0_45_count', 'rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count', 'rsi_14_O1/U0_60_count',
                     'rsi_14_O1/U0_65_count', 'rsi_14_O1/U0_70_count'], 1, inplace=True)
        
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        
            #if count % 10 == 0:
            #    print(count)
    
    main_df['RSIweight70'] = main_df.sum(axis=1)
    main_df['LowerRSI'] = 30
    main_df['UpperRSI'] = 70
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI70_weighted.csv')

def compile_data_rsi_above30_count():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
        
            df.rename(columns={'rsi_14_O1/U0_30_count': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','30ma_O1/U0_count','100ma_O1/U0_weighted',
                     '50ma_O1/U0_count','150ma_O1/U0_weighted','100ma_O1/U0_count',
                     '200ma_O1/U0_weighted', '200ma_O1/U0_count', '150ma_O1/U0_count',
                     'rsi_14_O1/U0_35_weighted',
                     'rsi_14_O1/U0_40_weighted', 'rsi_14_O1/U0_45_weighted',
                     'rsi_14_O1/U0_50_weighted', 'rsi_14_O1/U0_55_weighted',
                     'rsi_14_O1/U0_60_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_count', 'rsi_14_O1/U0_40_count',
                     'rsi_14_O1/U0_45_count', 'rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count', 'rsi_14_O1/U0_60_count',
                     'rsi_14_O1/U0_65_count', 'rsi_14_O1/U0_70_count'], 1, inplace=True)
        
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        
            #if count % 10 == 0:
            #    print(count)
    
    main_df['countRSI30'] = main_df.sum(axis=1)
    main_df['countpercRSI30'] = main_df['countRSI30']/(count+1)
    main_df['LowerRSI'] = 30
    main_df['UpperRSI'] = 70
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI30_count.csv')

def compile_data_rsi_above35_count():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
        
            df.rename(columns={'rsi_14_O1/U0_35_count': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','30ma_O1/U0_count','100ma_O1/U0_weighted',
                     '50ma_O1/U0_count','150ma_O1/U0_weighted','100ma_O1/U0_count',
                     '200ma_O1/U0_weighted', '200ma_O1/U0_count', '150ma_O1/U0_count',
                     'rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_40_weighted', 'rsi_14_O1/U0_45_weighted',
                     'rsi_14_O1/U0_50_weighted', 'rsi_14_O1/U0_55_weighted',
                     'rsi_14_O1/U0_60_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count',
                     'rsi_14_O1/U0_35_weighted', 'rsi_14_O1/U0_40_count',
                     'rsi_14_O1/U0_45_count', 'rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count', 'rsi_14_O1/U0_60_count',
                     'rsi_14_O1/U0_65_count', 'rsi_14_O1/U0_70_count'], 1, inplace=True)
        
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        
            #if count % 10 == 0:
            #    print(count)
    main_df['countRSI35'] = main_df.sum(axis=1)
    main_df['countpercRSI35'] = main_df['countRSI35']/(count+1)
    main_df['LowerRSI'] = 30
    main_df['UpperRSI'] = 70
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI35_count.csv')

def compile_data_rsi_above40_count():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
        
            df.rename(columns={'rsi_14_O1/U0_40_count': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','30ma_O1/U0_count','100ma_O1/U0_weighted',
                     '50ma_O1/U0_count','150ma_O1/U0_weighted','100ma_O1/U0_count',
                     '200ma_O1/U0_weighted', '200ma_O1/U0_count', '150ma_O1/U0_count',
                     'rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted', 'rsi_14_O1/U0_45_weighted',
                     'rsi_14_O1/U0_50_weighted', 'rsi_14_O1/U0_55_weighted',
                     'rsi_14_O1/U0_60_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count',
                     'rsi_14_O1/U0_35_count', 'rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_45_count', 'rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count', 'rsi_14_O1/U0_60_count',
                     'rsi_14_O1/U0_65_count', 'rsi_14_O1/U0_70_count'], 1, inplace=True)
        
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        
            #if count % 10 == 0:
            #    print(count)
    main_df['countRSI40'] = main_df.sum(axis=1)
    main_df['countpercRSI40'] = main_df['countRSI40']/(count+1)
    main_df['LowerRSI'] = 30
    main_df['UpperRSI'] = 70
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI40_count.csv')

def compile_data_rsi_above45_count():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
        
            df.rename(columns={'rsi_14_O1/U0_45_count': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','30ma_O1/U0_count','100ma_O1/U0_weighted',
                     '50ma_O1/U0_count','150ma_O1/U0_weighted','100ma_O1/U0_count',
                     '200ma_O1/U0_weighted', '200ma_O1/U0_count', '150ma_O1/U0_count',
                     'rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted', 'rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_50_weighted', 'rsi_14_O1/U0_55_weighted',
                     'rsi_14_O1/U0_60_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count',
                     'rsi_14_O1/U0_35_count', 'rsi_14_O1/U0_40_count',
                     'rsi_14_O1/U0_45_weighted', 'rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count', 'rsi_14_O1/U0_60_count',
                     'rsi_14_O1/U0_65_count', 'rsi_14_O1/U0_70_count'], 1, inplace=True)
        
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        
            #if count % 10 == 0:
            #    print(count)
    main_df['countRSI45'] = main_df.sum(axis=1)
    main_df['countpercRSI45'] = main_df['countRSI45']/(count+1)
    main_df['LowerRSI'] = 30
    main_df['UpperRSI'] = 70
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI45_count.csv')

def compile_data_rsi_above50_count():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
        
            df.rename(columns={'rsi_14_O1/U0_50_count': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','30ma_O1/U0_count','100ma_O1/U0_weighted',
                     '50ma_O1/U0_count','150ma_O1/U0_weighted','100ma_O1/U0_count',
                     '200ma_O1/U0_weighted', '200ma_O1/U0_count', '150ma_O1/U0_count',
                     'rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted', 'rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_45_weighted', 'rsi_14_O1/U0_55_weighted',
                     'rsi_14_O1/U0_60_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count',
                     'rsi_14_O1/U0_35_count', 'rsi_14_O1/U0_40_count',
                     'rsi_14_O1/U0_45_count', 'rsi_14_O1/U0_50_weighted',
                     'rsi_14_O1/U0_55_count', 'rsi_14_O1/U0_60_count',
                     'rsi_14_O1/U0_65_count', 'rsi_14_O1/U0_70_count'], 1, inplace=True)
        
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        
            #if count % 10 == 0:
            #    print(count)
    main_df['countRSI50'] = main_df.sum(axis=1)
    main_df['countpercRSI50'] = main_df['countRSI50']/(count+1)
    main_df['LowerRSI'] = 30
    main_df['UpperRSI'] = 70
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI50_count.csv')

def compile_data_rsi_above55_count():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
        
            df.rename(columns={'rsi_14_O1/U0_55_count': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','30ma_O1/U0_count','100ma_O1/U0_weighted',
                     '50ma_O1/U0_count','150ma_O1/U0_weighted','100ma_O1/U0_count',
                     '200ma_O1/U0_weighted', '200ma_O1/U0_count', '150ma_O1/U0_count',
                     'rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted', 'rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_45_weighted', 'rsi_14_O1/U0_50_weighted',
                     'rsi_14_O1/U0_60_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count',
                     'rsi_14_O1/U0_35_count', 'rsi_14_O1/U0_40_count',
                     'rsi_14_O1/U0_45_count', 'rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_weighted', 'rsi_14_O1/U0_60_count',
                     'rsi_14_O1/U0_65_count', 'rsi_14_O1/U0_70_count'], 1, inplace=True)
        
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        
            #if count % 10 == 0:
            #    print(count)
    main_df['countRSI55'] = main_df.sum(axis=1)
    main_df['countpercRSI55'] = main_df['countRSI55']/(count+1)
    main_df['LowerRSI'] = 30
    main_df['UpperRSI'] = 70
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI55_count.csv')

def compile_data_rsi_above60_count():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
        
            df.rename(columns={'rsi_14_O1/U0_60_count': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','30ma_O1/U0_count','100ma_O1/U0_weighted',
                     '50ma_O1/U0_count','150ma_O1/U0_weighted','100ma_O1/U0_count',
                     '200ma_O1/U0_weighted', '200ma_O1/U0_count', '150ma_O1/U0_count',
                     'rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted', 'rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_45_weighted', 'rsi_14_O1/U0_50_weighted',
                     'rsi_14_O1/U0_55_weighted','rsi_14_O1/U0_65_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count',
                     'rsi_14_O1/U0_35_count', 'rsi_14_O1/U0_40_count',
                     'rsi_14_O1/U0_45_count', 'rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count', 'rsi_14_O1/U0_60_weighted',
                     'rsi_14_O1/U0_65_count', 'rsi_14_O1/U0_70_count'], 1, inplace=True)
        
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        
            #if count % 10 == 0:
            #    print(count)
    main_df['countRSI60'] = main_df.sum(axis=1)
    main_df['countpercRSI60'] = main_df['countRSI60']/(count+1)
    main_df['LowerRSI'] = 30
    main_df['UpperRSI'] = 70
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI60_count.csv')

def compile_data_rsi_above65_count():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
        
            df.rename(columns={'rsi_14_O1/U0_65_count': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','30ma_O1/U0_count','100ma_O1/U0_weighted',
                     '50ma_O1/U0_count','150ma_O1/U0_weighted','100ma_O1/U0_count',
                     '200ma_O1/U0_weighted', '200ma_O1/U0_count', '150ma_O1/U0_count',
                     'rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted', 'rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_45_weighted', 'rsi_14_O1/U0_50_weighted',
                     'rsi_14_O1/U0_55_weighted','rsi_14_O1/U0_60_weighted',
                     'rsi_14_O1/U0_70_weighted','rsi_14_O1/U0_30_count',
                     'rsi_14_O1/U0_35_count', 'rsi_14_O1/U0_40_count',
                     'rsi_14_O1/U0_45_count', 'rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count', 'rsi_14_O1/U0_60_count',
                     'rsi_14_O1/U0_65_weighted', 'rsi_14_O1/U0_70_count'], 1, inplace=True)
        
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        
            #if count % 10 == 0:
            #    print(count)
    main_df['countRSI65'] = main_df.sum(axis=1)
    main_df['countpercRSI65'] = main_df['countRSI65']/(count+1)
    main_df['LowerRSI'] = 30
    main_df['UpperRSI'] = 70
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI65_count.csv')

def compile_data_rsi_above70_count():
    
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        if not os.path.exists('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker)):   
            print("File for " + ticker + " does not exist")
        else:
            df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
        
            df.rename(columns={'rsi_14_O1/U0_70_count': ticker}, inplace=True)
            df.drop(['High','Low','Open','Close','Volume','Adj Close','30ma',
                     '50ma','100ma','150ma','200ma','change','gain','loss','avg_gain',
                     'avg_loss','rs','rsi_14','30ma_O1/U0_weighted',
                     '50ma_O1/U0_weighted','30ma_O1/U0_count','100ma_O1/U0_weighted',
                     '50ma_O1/U0_count','150ma_O1/U0_weighted','100ma_O1/U0_count',
                     '200ma_O1/U0_weighted', '200ma_O1/U0_count', '150ma_O1/U0_count',
                     'rsi_14_O1/U0_30_weighted',
                     'rsi_14_O1/U0_35_weighted', 'rsi_14_O1/U0_40_weighted',
                     'rsi_14_O1/U0_45_weighted', 'rsi_14_O1/U0_50_weighted',
                     'rsi_14_O1/U0_55_weighted','rsi_14_O1/U0_60_weighted',
                     'rsi_14_O1/U0_65_weighted','rsi_14_O1/U0_30_count',
                     'rsi_14_O1/U0_35_count', 'rsi_14_O1/U0_40_count',
                     'rsi_14_O1/U0_45_count', 'rsi_14_O1/U0_50_count',
                     'rsi_14_O1/U0_55_count', 'rsi_14_O1/U0_60_count',
                     'rsi_14_O1/U0_65_count', 'rsi_14_O1/U0_70_weighted'], 1, inplace=True)
        
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        
            #if count % 10 == 0:
            #    print(count)
    main_df['countRSI70'] = main_df.sum(axis=1)
    main_df['countpercRSI70'] = main_df['countRSI70']/(count+1)
    main_df['LowerRSI'] = 30
    main_df['UpperRSI'] = 70
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI70_count.csv')

def consolidate_ou_data():
    
    start = dt.datetime(2000, 1, 1)
    end = dt.datetime.now()
    
    if Path('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500_Consolidated.csv').is_file():
        df_current = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500_Consolidated.csv', parse_dates=True, index_col=0)
        if df_current.index[-1] < date.today():
            start = df_current.index[-1] + timedelta(days=1)
            df_new = web.DataReader('^GSPC', 'yahoo', start, end)
            df_new.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500_Consolidated.csv', mode='a', header=False)
    else:
        df_new = web.DataReader('^GSPC', 'yahoo', start, end)
        df_new.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500_Consolidated.csv')
    
    #adding typical indicators to the S&P 500 index
    main_df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500_Consolidated.csv')
    main_df.set_index('Date', inplace=True)
    main_df['30ma'] = main_df['Adj Close'].rolling(window=30,min_periods=0).mean()
    main_df['50ma'] = main_df['Adj Close'].rolling(window=50,min_periods=0).mean()
    main_df['100ma'] = main_df['Adj Close'].rolling(window=100,min_periods=0).mean()
    main_df['150ma'] = main_df['Adj Close'].rolling(window=150,min_periods=0).mean()
    main_df['200ma'] = main_df['Adj Close'].rolling(window=200,min_periods=0).mean()
    main_df['change'] = main_df['Adj Close'].diff()
    main_df['gain'] = main_df.change.mask(main_df.change < 0, 0.0)
    main_df['loss'] = -main_df.change.mask(main_df.change > 0, -0.0)
    n = 14
    main_df['avg_gain'] = main_df['gain'].rolling(window=n,min_periods=0).mean()
    main_df['avg_loss'] = main_df['loss'].rolling(window=n,min_periods=0).mean()
    main_df['rs'] = main_df.avg_gain / main_df.avg_loss
    main_df['rsi_14'] = 100 - (100 / (1 + main_df.rs))
    
    #compiling moving average weight data into master file for charting
    df_30_w = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP50030MA_weighted.csv') #opens identified file
    df_30_w.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['weight30'] = df_30_w['weight30'] #pulls the count## value to the main dataframe
    
    df_50_w = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP50050MA_weighted.csv') #opens identified file
    df_50_w.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['weight50'] = df_50_w['weight50'] #pulls the count## value to the main dataframe
    
    df_100_w = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500100MA_weighted.csv') #opens identified file
    df_100_w.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['weight100'] = df_100_w['weight100'] #pulls the count## value to the main dataframe
    
    df_150_w = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500150MA_weighted.csv') #opens identified file
    df_150_w.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['weight150'] = df_150_w['weight150'] #pulls the count## value to the main dataframe
    
    df_200_w = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500200MA_weighted.csv') #opens identified file
    df_200_w.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['weight200'] = df_200_w['weight200'] #pulls the count## value to the main dataframe
    
    #compiling moving average count data into master file for charting
    df_30_c = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP50030MA_count.csv') #opens identified file
    df_30_c.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['count30'] = df_30_c['count30'] #pulls the count## value to the main dataframe
    main_df['countperc30'] = df_30_c['countperc30'] #pulls the count## value to the main dataframe
    
    df_50_c = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP50050MA_count.csv') #opens identified file
    df_50_c.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['count50'] = df_50_c['count50'] #pulls the count## value to the main dataframe
    main_df['countperc50'] = df_50_c['countperc50'] #pulls the count## value to the main dataframe
    
    df_100_c = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500100MA_count.csv') #opens identified file
    df_100_c.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['count100'] = df_100_c['count100'] #pulls the count## value to the main dataframe
    main_df['countperc100'] = df_100_c['countperc100'] #pulls the count## value to the main dataframe
    
    df_150_c = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500150MA_count.csv') #opens identified file
    df_150_c.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['count150'] = df_150_c['count150'] #pulls the count## value to the main dataframe
    main_df['countperc150'] = df_150_c['countperc150'] #pulls the count## value to the main dataframe
    
    df_200_c = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500200MA_count.csv') #opens identified file
    df_200_c.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['count200'] = df_200_c['count200'] #pulls the count## value to the main dataframe
    main_df['countperc200'] = df_200_c['countperc200'] #pulls the count## value to the main dataframe
    
    #compiling RSI count data into master file for charting
    df_30_RSIc = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI30_count.csv') #opens identified file
    df_30_RSIc.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['countRSI30'] = df_30_RSIc['countRSI30'] #pulls the count## value to the main dataframe
    main_df['countpercRSI30'] = df_30_RSIc['countpercRSI30'] #pulls the count## value to the main dataframe
    df_35_RSIc = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI35_count.csv') #opens identified file
    df_35_RSIc.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['countRSI35'] = df_35_RSIc['countRSI35'] #pulls the count## value to the main dataframe
    main_df['countpercRSI35'] = df_35_RSIc['countpercRSI35'] #pulls the count## value to the main dataframe
    df_40_RSIc = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI40_count.csv') #opens identified file
    df_40_RSIc.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['countRSI40'] = df_40_RSIc['countRSI40'] #pulls the count## value to the main dataframe
    main_df['countpercRSI40'] = df_40_RSIc['countpercRSI40'] #pulls the count## value to the main dataframe
    df_45_RSIc = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI45_count.csv') #opens identified file
    df_45_RSIc.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['countRSI45'] = df_45_RSIc['countRSI45'] #pulls the count## value to the main dataframe
    main_df['countpercRSI45'] = df_45_RSIc['countpercRSI45'] #pulls the count## value to the main dataframe
    df_50_RSIc = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI50_count.csv') #opens identified file
    df_50_RSIc.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['countRSI50'] = df_50_RSIc['countRSI50'] #pulls the count## value to the main dataframe
    main_df['countpercRSI50'] = df_50_RSIc['countpercRSI50'] #pulls the count## value to the main dataframe
    df_55_RSIc = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI55_count.csv') #opens identified file
    df_55_RSIc.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['countRSI55'] = df_55_RSIc['countRSI55'] #pulls the count## value to the main dataframe
    main_df['countpercRSI55'] = df_55_RSIc['countpercRSI55'] #pulls the count## value to the main dataframe
    df_60_RSIc = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI60_count.csv') #opens identified file
    df_60_RSIc.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['countRSI60'] = df_60_RSIc['countRSI60'] #pulls the count## value to the main dataframe
    main_df['countpercRSI60'] = df_60_RSIc['countpercRSI60'] #pulls the count## value to the main dataframe
    df_65_RSIc = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI65_count.csv') #opens identified file
    df_65_RSIc.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['countRSI65'] = df_65_RSIc['countRSI65'] #pulls the count## value to the main dataframe
    main_df['countpercRSI65'] = df_65_RSIc['countpercRSI65'] #pulls the count## value to the main dataframe
    df_70_RSIc = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI70_count.csv') #opens identified file
    df_70_RSIc.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['countRSI70'] = df_70_RSIc['countRSI70'] #pulls the count## value to the main dataframe
    main_df['countpercRSI70'] = df_70_RSIc['countpercRSI70'] #pulls the count## value to the main dataframe
    
    #compiling RSI weight data into master file for charting
    df_30_RSIw = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI30_weighted.csv') #opens identified file
    df_30_RSIw.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['RSIweight30'] = df_30_RSIw['RSIweight30'] #pulls the count## value to the main dataframe
    df_35_RSIw = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI35_weighted.csv') #opens identified file
    df_35_RSIw.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['RSIweight35'] = df_35_RSIw['RSIweight35'] #pulls the count## value to the main dataframe
    df_40_RSIw = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI40_weighted.csv') #opens identified file
    df_40_RSIw.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['RSIweight40'] = df_40_RSIw['RSIweight40'] #pulls the count## value to the main dataframe
    df_45_RSIw = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI45_weighted.csv') #opens identified file
    df_45_RSIw.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['RSIweight45'] = df_45_RSIw['RSIweight45'] #pulls the count## value to the main dataframe
    df_50_RSIw = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI50_weighted.csv') #opens identified file
    df_50_RSIw.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['RSIweight50'] = df_50_RSIw['RSIweight50'] #pulls the count## value to the main dataframe
    df_55_RSIw = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI55_weighted.csv') #opens identified file
    df_55_RSIw.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['RSIweight55'] = df_55_RSIw['RSIweight55'] #pulls the count## value to the main dataframe
    df_60_RSIw = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI60_weighted.csv') #opens identified file
    df_60_RSIw.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['RSIweight60'] = df_60_RSIw['RSIweight60'] #pulls the count## value to the main dataframe
    df_65_RSIw = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI65_weighted.csv') #opens identified file
    df_65_RSIw.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['RSIweight65'] = df_65_RSIw['RSIweight65'] #pulls the count## value to the main dataframe
    df_70_RSIw = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500RSI70_weighted.csv') #opens identified file
    df_70_RSIw.set_index('Date', inplace=True) #defined the date as the index so ID is inserted
    main_df['RSIweight70'] = df_70_RSIw['RSIweight70'] #pulls the count## value to the main dataframe
    
    main_df['LowerRSI'] = 30
    main_df['UpperRSI'] = 70
    
    main_df.to_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500_Consolidated.csv')

def chart_data():
    print("Generating Chart.")
    style.use('ggplot')
    figure(num=None, figsize=(38, 24), dpi=150, facecolor='w', edgecolor='k')
    plt.title('S&P 500')
    #reads data for dataframe from the csv file generated above
    df = pd.read_csv('//192.168.1.13/Server/OTHER/Investing/Python/Projects/Reference_data/SP500_Consolidated.csv', parse_dates=True, index_col=0)
    #available data Date,High,Low,Open,Close,Volume,Adj Close,30ma,50ma,100ma,150ma,200ma,change,
    # gain,loss,avg_gain,avg_loss,rs,rsi_14,weight30,weight50,weight100,weight150,weight200,count30,
    # countperc30,count50,countperc50,count100,countperc100,count150,countperc150,count200,countperc200,
    # countRSI30,countpercRSI30,countRSI35,countpercRSI35,countRSI40,countpercRSI40,countRSI45,
    # countpercRSI45,countRSI50,countpercRSI50,countRSI55,countpercRSI55,countRSI60,countpercRSI60,
    # countRSI65,countpercRSI65,countRSI70,countpercRSI70,RSIweight30,RSIweight35,RSIweight40,RSIweight45,
    # RSIweight50,RSIweight55,RSIweight60,RSIweight65,RSIweight70,LowerRSI,UpperRSI
    
    plt.rcParams['axes.xmargin'] = 0.01
    plt.rcParams['axes.ymargin'] = 0.05
    
    start = '2005-01-01'
    end  = '2020-06-01'    
    # start = '2008-01-01'
    # end  = '2010-06-01'
    #start = '2017-01-01'
    #end  = '2021-01-01' 
    df=df.loc[start:end]
    
    price_ax_height = 8
    price_ax_position = 0
    RSI_ax_height = 8
    RSI_ax_position = price_ax_height + 2
    ax9_height = 8
    ax9_position = RSI_ax_height + RSI_ax_position + 2
    ax5_height = 8
    ax5_position = ax9_height + ax9_position + 2
    ax6_height = 8
    ax6_position = ax5_height + ax5_position + 2
    ax8_height = 8
    ax8_position = ax6_height + ax6_position + 0
    plot_height = ax8_position + ax8_height
    plot_width = 100
    
    #the following only generates the chart, with no data
    price_ax = plt.subplot2grid((plot_height,plot_width), (price_ax_position,0), rowspan=price_ax_height, colspan=plot_width)
    plt.ylabel('Price')
    plt.title(start + " to " + end)
    
    RSI_ax = plt.subplot2grid((plot_height,plot_width), (RSI_ax_position,0), rowspan=RSI_ax_height, colspan=plot_width,sharex=price_ax)
    plt.ylabel('RSI')
    plt.title('S&P 500 RSI (14)')
    ax9 = plt.subplot2grid((plot_height,plot_width), (ax9_position,0), rowspan=ax9_height, colspan=plot_width,sharex=price_ax)
    plt.ylabel('Percent')
    plt.title('Weighted count of quantity above or below specified RSI')
    ax5 = plt.subplot2grid((plot_height,plot_width), (ax5_position,0), rowspan=ax5_height, colspan=plot_width, sharex=price_ax)
    plt.ylabel('Count')
    plt.title('Count of stocks below DMA')
    ax6 = plt.subplot2grid((plot_height,plot_width), (ax6_position,0), rowspan=ax6_height, colspan=plot_width, sharex=price_ax)
    plt.ylabel('Percent')
    plt.title('Weighted count of stocks below DMA')
    
    price_ax.plot(df['Adj Close'], linewidth=1,label="Price")
    price_ax.plot(df['50ma'], linewidth=1,label="50 DMA")
    price_ax.plot(df['200ma'], linewidth=1,label="200 DMA")
    
    RSI_ax.plot(df['rsi_14'], linewidth=1)
    RSI_ax.plot(df['LowerRSI'], linewidth=2,color="pink")
    RSI_ax.plot(df['UpperRSI'], linewidth=2,color="pink")
    
    #program counts when the RSI was above given threshold, this is why we start with '1-' when plotting to show how many are below a threshold
    RSIweight1 = '65'
    ax9.plot(df['RSIweight' + RSIweight1], linewidth=1, color="r", label="Above " + RSIweight1 + " RSI",alpha=0.8)
    RSIweight2 = '35'
    ax9.plot(1-df['RSIweight' + RSIweight2], linewidth=1, color="g", label="Below " + RSIweight2 + " RSI",alpha=0.8)
    ax9.legend(loc=1)

    ax5.plot(df['count50'], linewidth=1,color="m", label="50 DMA")
    ax5.plot(df['count100'], linewidth=1,color="y", label="100 DMA")
    ax5.plot(df['count200'], linewidth=1,color="b", label="200 DMA")
    ax5.legend(loc=1)
    
    ax6.plot(df['weight50'], linewidth=1,color="m", label="50 DMA")
    ax6.plot(df['weight100'], linewidth=1,color="y", label="100 DMA")
    ax6.plot(df['weight200'], linewidth=1,color="b", label="200 DMA")
    ax6.legend(loc=1)

    price_ax.legend()
    plt.show()
    
#save_sp500_tickers() #(1 of 11) #gets list of S&P tickers only and inputs into txt file
#get_data_from_yahoo() #(2 of 11) #get price data for given timeframe from yahoo and creates csv file for each ticker in stock_dfs folder
#save_sp500_tickersslickchart() #(3 of 11) #pulls S&P_500 list from slickcharts with ticker, company name, weight in S&P list, and current price
#indicator_analysis() #(4 of 11)

def compile_mass_data():
    print("Generating S&P 500 matrix for tickers with 30ma data with weighting. (5 of 11)")
    compile_data_30ma_weighted()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 30ma data for count. (6 of 11)")
    compile_data_30ma_count()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 50ma data with weighting. (7 of 11)")
    compile_data_50ma_weighted()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 50ma data for count. (8 of 11)")
    compile_data_50ma_count()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 100ma data with weighting. (9 of 11)")
    compile_data_100ma_weighted()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 100ma data for count. (10 of 11)")
    compile_data_100ma_count()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 150ma data with weighting. (11 of 11)")
    compile_data_150ma_weighted()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 150ma data for count. (12 of 11)")
    compile_data_150ma_count()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 200ma data with weighting. (13 of 11)")
    compile_data_200ma_weighted()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 200ma data for count. (14 of 11)")
    compile_data_200ma_count()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 30 RSI and above data with weighting. (15 of 11)")
    compile_data_rsi_above30_weighted()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 35 RSI and above data with weighting. (16 of 11)")
    compile_data_rsi_above35_weighted()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 40 RSI and above data with weighting. (17 of 11)")
    compile_data_rsi_above40_weighted()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 45 RSI and above data with weighting. (18 of 11)")
    compile_data_rsi_above45_weighted()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 50 RSI and above data with weighting. (19 of 11)")
    compile_data_rsi_above50_weighted()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 55 RSI and above data with weighting. (20 of 11)")
    compile_data_rsi_above55_weighted()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 60 RSI and above data with weighting. (21 of 11)")
    compile_data_rsi_above60_weighted()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 65 RSI and above data with weighting. (22 of 11)")
    compile_data_rsi_above65_weighted()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 70 RSI and above data with weighting. (23 of 11)")
    compile_data_rsi_above70_weighted()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 30 RSI and above data as count. (24 of 11)")
    compile_data_rsi_above30_count()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 35 RSI and above data as count. (25 of 11)")
    compile_data_rsi_above35_count()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 40 RSI and above data as count. (26 of 11)")
    compile_data_rsi_above40_count()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 45 RSI and above data as count. (27 of 11)")
    compile_data_rsi_above45_count()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 50 RSI and above data as count. (28 of 11)")
    compile_data_rsi_above50_count()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 55 RSI and above data as count. (29 of 11)")
    compile_data_rsi_above55_count()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 60 RSI and above data as count. (30 of 11)")
    compile_data_rsi_above60_count()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 65 RSI and above data as count. (31 of 11)")
    compile_data_rsi_above65_count()
    print(dt.datetime.now() - start_time)
    print("Generating S&P 500 matrix for tickers with 70 RSI and above data as count. (32 of 11)")
    compile_data_rsi_above70_count()
    print(dt.datetime.now() - start_time)
    print("Compiling & Consolidating mass data. (33 of 11)")
    consolidate_ou_data()
    print(dt.datetime.now() - start_time)

#compile_mass_data()
print(dt.datetime.now() - start_time)
#chart_data()
#consolidate_ou_data()
print("Done")