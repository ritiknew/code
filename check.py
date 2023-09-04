import torch
STD_PATH='C:\\Users\\lenovo\\PycharmProjects\intraday\\'
import pickle
import  tna
import fyers
import other as ot
import pandas as pd
import os
import fyers as fy
import datetime
def update_stock(name):
    # it update stocks data  for 100 days to today
    today=datetime.datetime.now()
    start_day=today-datetime.timedelta(100*1.28)
    a=fyers.historical_data(name,start_day)
    return a
def ready_to_machine_input(a,name_stock,date=1):
    a = a.copy()
    a.set_index('time', inplace=True)
    a['time'] = a.index
    # day name
    # a['day'] = a['time'].apply(lambda x: x.strftime('%A'))


    # vi means volume input,vo=volume output means net day data


    x = [2, 4, 8,16, 32,64,128]
    x = [1]
    # slope of close value of 2,4,16,32, days
    for i in x:
        a['cs' + str(i)] = tna.slope(a, 'close', i)
    sma = {'sma2': 2,'sma4': 4,'sma8':8, 'sma16': 16, 'sma32': 32, 'sma64': 64,'sma128':128}
    # ema = {'ema4': 4, 'ema16': 16, 'ema32': 32, 'ema64': 64}
    # sma = {'sma16': 16, 'sma64': 64}
    # ema = {'ema4': 4, 'ema16': 8}
    # sma and ema value  of 4,16,32,64
    for i in sma:
        a[i] = tna.sma(a, 'close', sma[i])
        a[i + 'i'] = a[i] / a['close']
    for i in sma:
        for j in x:
            a[i + 's' + str(j)] = tna.slope(a, i, j)
    a['date'] = a['time']
    a.set_index('time', inplace=True)
    a['ordinal'] = a['date'].apply(lambda x: x.toordinal())
    a.set_index('ordinal', inplace=True)
    stock=pd.read_pickle(STD_PATH+'swing\\'+name_stock)
    buy_date = stock['buy_date'].unique()
    buy_date = list(map(lambda x: int(x), buy_date))
    sell_date = stock['sell_date'].unique()
    sell_date = list(map(lambda x: int(x), sell_date))
    sell = []
    buy = []
    nothing = []
    a=a[a.index>=min(buy_date)]
    for i in a.index:
        if i in sell_date or i in buy_date:
            nothing.append(0)
            if i in sell_date:
                sell.append(1)
            else:
                sell.append(0)
            if i in buy_date:
                buy.append(1)
            else:
                buy.append(0)
        else:
            nothing.append(1)
            sell.append(0)
            buy.append(0)
    a['sell']=sell
    a['buy']=buy
    a['nothing']=nothing
    a.drop(['open','high', 'low', 'close','volume','sma2','sma4','sma8','sma16','sma32','sma64','sma128','date'], axis=1, inplace=True)
    # a = torch.from_numpy(a.to_numpy())
    # a = a.to(torch.float32)
    return a

def output_tomorrow(date=1):
    stock_name=['RELIANCE','TCS','HINDUNILVR','KOTAKBANK','ICICIBANK','SBIN','WIPRO','AXISBANK','TATASTEEL','TATAMOTORS','ADANIENT','FACT']
    # stock_name=['RELIANCE','TCS',]
    for i in stock_name:
        pass
        # a=update_stock(i)
        # a.to_pickle('.\\stock_training\\'+i)

    df=pd.DataFrame({'name':[],'o':[],'h':[],'l':[],'c':[],'red':[],'green':[]})
    for i in stock_name:
        stock=pd.read_pickle('.\\stock_training\\'+i)
        stock_model_r = torch.jit.load('.\\model_color\\'+i)
        stock_model_r.eval()
        stock,open=ready_to_machine_input(stock,date)
        output_color=stock_model_r(stock).detach()
        print(output_color,torch.round(output_color),i)
        output_color=output_color[0].tolist()
        # -------------------------------
        stock = pd.read_pickle('.\\stock_training\\' + i)
        stock_model_r = torch.jit.load('.\\model_regression\\' + i)
        stock_model_r.eval()
        stock,open = ready_to_machine_input(stock, date)
        output_ohlc = stock_model_r(stock).detach()[0].tolist()
        y=[i,output_ohlc[0]*open,output_ohlc[1]*open,output_ohlc[2]*open,output_ohlc[3]*open,output_color[0],output_color[1]]
        df.loc[len(df.index)]=y

        # df['output_ohlc[0]']=df['output_ohlc[0]'].apply(lambda x:)
    df['red_round'] = df['red'].apply(lambda x: round(x))
    df['green_round'] = df['green'].apply(lambda x: round(x))
    green={1:'green',0:'red'}
    df['color']=df['green_round'].apply(lambda x:green[x])
    return df
# y=output_tomorrow()
# y.to_csv('output.csv')
# ready_to_machine_input(pd.read_pickle(STD_PATH+'stock\\CIPLA'))




