# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

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
#import fix_yahoo_finance as yf
import yfinance as yf
from matplotlib.pyplot import figure
import time

yf.pdr_override

start_time = dt.datetime.now()
startyear = 1970 #1970

with open("sp500tickers.pickle", "rb") as f:
    tickers = pickle.load(f)

if not os.path.exists('Reference_data'): #creates folder if the folder doesn't exist
        os.makedirs('Reference_data')
if not os.path.exists('Reference_data/stock_dfs'): #creates folder if the folder doesn't exist
        os.makedirs('Reference_data/stock_dfs')
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

def get_data_from_yahoo(reload_sp500=False):
    print("Updating price database. (2 of 11)")
    print(dt.datetime.now())

    start = dt.datetime(startyear, 1, 1)
    end = dt.datetime.now()
    MAX_ATTEMPTS = 5
    i = 0
    df_tickerpull = pd.read_csv('Reference_data/stock_dfs/sp500wikilist.csv', index_col=0)
    tickers = df_tickerpull.index #this outputs the correct weight
    #print(tickers)
    
    icount = df_tickerpull.count()
    #print(icount)
    
    for ticker in tickers:
        i = i + 1 #increase counter for % progress indicator
        iperc = (i/icount)*100 #% progress indicator for the next line
        print("\r" + str(round(iperc, 1)) + ' % ' + ticker + "      ", "\r", end='\r',flush=True)
        #following is attempted with MAX_ATTEMPTS above
        try:
            #print("Looking up data for " + ticker)
            #first line checks if the file already exists
            if os.path.exists('Reference_data/stock_dfs/{}.csv'.format(ticker)):
                #if file exists, the following line extracts the dataframe to df_current
                df_current = pd.read_csv('Reference_data/stock_dfs/{}.csv'.format(ticker), parse_dates=True, index_col=0)
                #prints statement that file exists for specific ticker
                print("File exists for " + ticker + ", checking for new data")
                #for first attempt to try
                for attempt in range(MAX_ATTEMPTS):
                    #date column check is pulled to the 'Date' column in the dataframe
                    df_current['Date'] = pd.to_datetime(df_current.index)
                    #recent_date finds the latest date in the current data set
                    recent_date = df_current['Date'].max()
                    #set todays date to todaydate
                    todaydate = date.today()
                    #check if the most recent data in file is older than today
                    if recent_date < todaydate:
                        #if more data exists, statement that more data is being added
                        print("Adding new data to " + ticker)
                        #set the start date as the following business day from the previous data set
                        start = recent_date + timedelta(days=1)
                        #extracts the new data required to build full data set and sets to df_new
                        df_new = web.DataReader(ticker, 'yahoo', start, end)
                        #appands the new df_new dataframe to the df_current dataframe and replaces the new dataframe df_new
                        df_new = df_current.append(df_new)
                        #writes the df_new dataframe back to the file
                        df_new.to_csv('Reference_data/stock_dfs/{}.csv'.format(ticker), mode='a', header=False)
                    else:
                        print("No new data for " +  ticker)
            else:
                try:
                    #print('Reference_data/stock_dfs/{}.csv'.format(ticker))
                    start_20year = dt.datetime(startyear, 1, 1)
                    df = web.DataReader(ticker, 'yahoo', start_20year, end)
                except Exception as e:
                    open('Reference_data/stock_dfs/NO DATA FOR THIS TICKER ON YAHOO ' + ticker + '.txt', 'a').close()
                    df = []
                df.to_csv('Reference_data/stock_dfs/{}.csv'.format(ticker))
        except Exception as e:
            print("Data does not exist for " + ticker + " from yahoo finance")
            open('Reference_data/stock_dfs/NO DATA FOR THIS TICKER ON YAHOO ' + ticker + '.txt', 'a').close()
            print(e)

def backupget_data_from_yahoo(reload_sp500=False):
    print("Updating price database. (2 of 11)")
    print(dt.datetime.now())

    start = dt.datetime(startyear, 1, 1)
    end = dt.datetime.now()
    MAX_ATTEMPTS = 1
    i = 0
    #df_tickerpull = pd.read_csv('Reference_data/stock_dfs/sp500tickersslickchart.csv', index_col=1)
    #sp500wikilist
    df_tickerpull = pd.read_csv('Reference_data/stock_dfs/sp500wikilist.csv', index_col=0)
    tickers = df_tickerpull.index #this outputs the correct weight
    icount = df_tickerpull.count()
    for ticker in tickers:
        i = i + 1
        iperc = (i/icount)*100
        print("\r" + str(round(iperc, 1)) + ' % ' + ticker + "      ", "\r", end='\r',flush=True)
        try:
            #print("Looking up data for " + ticker)
            if os.path.exists('Reference_data/stock_dfs/{}.csv'.format(ticker)):
                df_current = pd.read_csv('Reference_data/stock_dfs/{}.csv'.format(ticker), parse_dates=True, index_col=0)
                print("File exists for " + ticker + ", checking for new data")
                for attempt in range(MAX_ATTEMPTS):
                    print('Trying to fetch')
                    print(df_current.index[-1])
                    if df_current.index[-1] < date.today():
                        print("Adding new data to " + ticker)
                        start = df_current.index[-1] + timedelta(days=1)
                        df_new = web.DataReader(ticker, 'yahoo', start, end)
                        if start == df_new.iloc[1:].index:
                            df_new = df_new.iloc[1:]
                        df_new.to_csv('Reference_data/stock_dfs/{}.csv'.format(ticker), mode='a', header=False)
                    else:
                        print("No new data for " +  ticker)
            else:
                try:
                    #print('Reference_data/stock_dfs/{}.csv'.format(ticker))
                    start_20year = dt.datetime(startyear, 1, 1)
                    df = web.DataReader(ticker, 'yahoo', start_20year, end)
                except Exception as e:
                    open('Reference_data/stock_dfs/NO DATA FOR THIS TICKER ON YAHOO ' + ticker + '.txt', 'a').close()
                    df = []
                df.to_csv('Reference_data/stock_dfs/{}.csv'.format(ticker))
        except Exception as e:
            print("Data does not exist for " + ticker + " from yahoo finance")
            open('Reference_data/stock_dfs/NO DATA FOR THIS TICKER ON YAHOO ' + ticker + '.txt', 'a').close()
            print(e)

def save_sp500_tickersslickchart(): #this runs but the file is not used at the moment, it will be used in the future for weighting of the data
    print("Gathering S&P500 weighting. (3 of 11)")
    #request.get downloads the website sourcecode into html
    resp = requests.get('https://www.slickcharts.com/sp500')
    #bs.beautifulsoup makes it into text
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    print(soup)
    data = soup.find('table', {'class': 'table'})
    print("hi")
    records = []
    print(data)
    company = []
    for row in data.findAll('tr')[1:]:
        company = row.findAll('td')[1].text
        print(company)
        ticker = row.findAll('td')[2].text
        weight = row.findAll('td')[3].text
        price = row.findAll(('td'), {'class': 'text-nowrap'})[0].text
        records.append((company, ticker, weight, price))
    df = pd.DataFrame(records, columns=['company','ticker','weight','price'])
    print(df)
    df['ticker'] = df['ticker'].str.replace('.','-')
    df.to_csv('Reference_data/stock_dfs/sp500tickersslickchart.csv', index=False, encoding='utf-8')

def save_sp500_tickers_wiki():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    wikisoup = bs.BeautifulSoup(resp.text, 'lxml')
    wikitable = wikisoup.find('table', {'class': 'wikitable sortable'})
    wiki_tickers = []
    for row in wikitable.findAll('tr')[1:]:
        wiki_ticker = row.findAll('td')[0].text
        wiki_tickers.append(wiki_ticker)
    wiki_df = pd.DataFrame(wiki_tickers, columns=['ticker'])
    wiki_df = wiki_df.replace('\n', '', regex=True)
    wiki_df['ticker'] = wiki_df['ticker'].str.replace('.','-')
    print(wiki_df)
    wiki_df.to_csv('Reference_data/stock_dfs/sp500wikilist.csv', index=False, encoding='utf-8')    

def get_SP500data_from_yahoo():
    start = dt.datetime(startyear, 1, 1)
    #end = dt.datetime.now()
    end = date.today()
    ticker = '^GSPC'
    MAX_ATTEMPTS = 1
    
    try:
        #print("Looking up data for " + ticker)
        if os.path.exists('Reference_data/stock_dfs/SP500.csv'):
            df_current = pd.read_csv('Reference_data/stock_dfs/SP500.csv', parse_dates=True, index_col=0)
            print("File exists for " + ticker + ", checking for new data")
            for attempt in range(MAX_ATTEMPTS):
                print('Trying to fetch')
                if df_current.index[-1] < date.today():
                    print("Adding new data to " + ticker)
                    start = df_current.index[-1] + timedelta(days=1)
                    df_new = web.DataReader(ticker, 'yahoo', start, end)
                    if start == df_new.iloc[1:].index:
                        df_new = df_new.iloc[1:]
                    df_new.to_csv('Reference_data/stock_dfs/SP500.csv', mode='a', header=False)
                else:
                    print("No new data for " +  ticker)
        else:
            try:
                #print('Reference_data/stock_dfs/SP500.csv')
                start_20year = dt.datetime(startyear, 1, 1)
                df = web.DataReader(ticker, 'yahoo', start_20year, end)
                #time.sleep(0.5)
            except Exception as e:
                open('Reference_data/stock_dfs/NO DATA FOR THIS TICKER ON YAHOO ' + ticker + '.txt', 'a').close()
                df = []
            df.to_csv('Reference_data/stock_dfs/SP500.csv')
    except Exception as e:
        print("Data does not exist for " + ticker + " from yahoo finance")
        open('Reference_data/stock_dfs/NO DATA FOR THIS TICKER ON YAHOO ' + ticker + '.txt', 'a').close()
        print(e)
        
def get_RUS2000data_from_yahoo():
    start = dt.datetime(startyear, 1, 1)
    #end = dt.datetime.now()
    end = date.today()
    ticker = '^RUT'
    MAX_ATTEMPTS = 1
    
    try:
        #print("Looking up data for " + ticker)
        if os.path.exists('Reference_data/stock_dfs/RUS2000.csv'):
            df_current = pd.read_csv('Reference_data/stock_dfs/RUS2000.csv', parse_dates=True, index_col=0)
            print("File exists for " + ticker + ", checking for new data")
            for attempt in range(MAX_ATTEMPTS):
                print('Trying to fetch')
                if df_current.index[-1] < date.today():
                    print("Adding new data to " + ticker)
                    start = df_current.index[-1] + timedelta(days=1)
                    df_new = web.DataReader(ticker, 'yahoo', start, end)
                    if start == df_new.iloc[1:].index:
                        df_new = df_new.iloc[1:]
                    df_new.to_csv('Reference_data/stock_dfs/RUS2000.csv', mode='a', header=False)
                else:
                    print("No new data for " +  ticker)
        else:
            try:
                #print('Reference_data/stock_dfs/SP500.csv')
                start_20year = dt.datetime(startyear, 1, 1)
                df = web.DataReader(ticker, 'yahoo', start_20year, end)
                #time.sleep(0.5)
            except Exception as e:
                open('Reference_data/stock_dfs/NO DATA FOR THIS TICKER ON YAHOO ' + ticker + '.txt', 'a').close()
                df = []
            df.to_csv('Reference_data/stock_dfs/RUS2000.csv')
    except Exception as e:
        print("Data does not exist for " + ticker + " from yahoo finance")
        open('Reference_data/stock_dfs/NO DATA FOR THIS TICKER ON YAHOO ' + ticker + '.txt', 'a').close()
        print(e)

def get_specific_from_yahoo():
    startyear = 1970
    start = dt.datetime(startyear, 1, 1)
    #end = dt.datetime.now()
    end = date.today()
    ticker = '^RUT'
    MAX_ATTEMPTS = 1
    
    try:
        if os.path.exists('Reference_data/stock_dfs/{}.csv'.format(ticker)):
            df_current = pd.read_csv('Reference_data/stock_dfs/{}.csv'.format(ticker), index_col='Date')
            print("File exists for " + ticker + ", checking for new data")
            for attempt in range(MAX_ATTEMPTS):
                #the following pulls the most recent date from the file
                df_current['Date'] = pd.to_datetime(df_current.index)
                recent_date = df_current['Date'].max()
                print(recent_date)
                print(date.today())
                todaydate = date.today()
                if recent_date < todaydate:
                    print('Trying to fetch')
                    #print("Adding new data to " + ticker)
                    print("Adding new data to " + "SP500")
                    start = recent_date + timedelta(days=1)
                    print(start)
                    df_new = web.DataReader(ticker, 'yahoo', start, end)
                    df_new = df_current.append(df_new)
                    df_new.to_csv('Reference_data/stock_dfs/{}.csv'.format(ticker))
                else:
                    print("No new data for " +  ticker)
        else:
            try:
                #print('Reference_data/stock_dfs/SP500.csv')
                start_20year = dt.datetime(startyear, 1, 1)
                df = web.DataReader(ticker, 'yahoo', start_20year, end)
                time.sleep(5)
            except Exception as e:
                open('Reference_data/stock_dfs/NO DATA FOR THIS TICKER ON YAHOO ' + ticker + '.txt', 'a').close()
                df = []
            df.to_csv('Reference_data/stock_dfs/{}.csv'.format(ticker))
    except Exception as e:
        print("Data does not exist for " + ticker + " from yahoo finance")
        open('Reference_data/stock_dfs/NO DATA FOR THIS TICKER ON YAHOO ' + ticker + '.txt', 'a').close()
        print(e)

#save_sp500_tickers()
#save_sp500_tickersslickchart() #keeping this as a backup, blacklisted
#save_sp500_tickers_wiki()
get_data_from_yahoo()
#backupget_data_from_yahoo()
get_SP500data_from_yahoo()
get_RUS2000data_from_yahoo()
#get_specific_from_yahoo()

print(dt.datetime.now())
print(dt.datetime.now() - start_time)
print("Done")