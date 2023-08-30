import torch
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
def ready_to_machine_input(a,date=1):
    a = a.copy()
    a.set_index('time', inplace=True)
    a['time'] = a.index
    # day name
    a['day'] = a['time'].apply(lambda x: x.strftime('%A'))

    a['next_open'] = a['open'].shift(periods=-1)
    a['next_high'] = a['high'].shift(periods=-1)
    a['next_low'] = a['low'].shift(periods=-1)
    a['next_close'] = a['close'].shift(periods=-1)
    a['next_volume'] = a['volume'].shift(periods=-1)
    a['mv'] = a['volume'].rolling(10).mean()
    # vi means volume input,vo=volume output means net day data
    a['vi'] = a['volume'] / a['mv']
    a['vo'] = a['next_volume'] / a['mv']
    a['oi'] = a['open'] / a['open']
    a['hi'] = a['high'] / a['open']
    a['li'] = a['low'] / a['open']
    a['ci'] = a['close'] / a['open']
    a['oo'] = a['next_open'] / a['open']
    a['co'] = a['next_close'] / a['open']
    a['ho'] = a['next_high'] / a['open']
    a['lo'] = a['next_low'] / a['open']
    a['rsi'] = tna.rsi(a, 'close', 14) / 100
    x = [2, 4, 16, 32]
    x = [2, 4, 16]
    # slope of close value of 2,4,16,32, days
    for i in x:
        a['cs' + str(i)] = tna.slope(a, 'close', i)
    sma = {'sma4': 4, 'sma16': 16, 'sma32': 32, 'sma64': 64}
    ema = {'ema4': 4, 'ema16': 16, 'ema32': 32, 'ema64': 64}
    sma = {'sma16': 16, 'sma64': 64}
    ema = {'ema4': 4, 'ema16': 8}
    # sma and ema value  of 4,16,32,64
    for i in sma:
        a[i] = tna.sma(a, 'close', sma[i])
        a[i + 'i'] = a[i] / a['open']
    for i in ema:
        a[i] = tna.ema(a, 'close', ema[i])
        a[i + 'i'] = a[i] / a['open']
    for i in sma:
        for j in x:
            a[i + 's' + str(j)] = tna.slope(a, i, j)
    for i in ema:
        for j in x:
            a[i + 's' + str(j)] = tna.slope(a, i, j)

    def r_g(open, close):
        if open > close:
            return 'red'
        else:
            return 'green'

    a['r_g_i'] = a.apply(lambda x: r_g(x['open'], x['close']), axis=1)
    a['r_g_o'] = a['r_g_i'].shift(periods=-1)
    open=a['open']
    a.drop(['high', 'low', 'open', 'close', 'mv', 'vo', 'oi', 'volume', 'time', 'next_open', 'next_high', 'next_low',
            'next_close', 'next_volume'], axis=1, inplace=True)
    for i in sma:
        a.pop(i)
    for i in ema:
        a.pop(i)
    day={'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Monday':0,'Sunday':6,'Saturday':5}
    r_g = {'red': 0, 'green': 1}
    red = {'red': 1, 'green': 0}
    green = {'red': 0, 'green': 1}
    a['day'] = a['day'].apply(lambda x: day[x])
    a['next_day'] = a['day'].shift(periods=-1)
    # a.astype({'next_day': 'int'}).dtypes
    a['r_g_o'] = a['r_g_i'].shift(periods=-1)
    a['r_g_i'] = a['r_g_i'].apply(lambda x: r_g[x])
    # a = a.dropna()
    # a['red_o'] = a['r_g_o'].apply(lambda x: red[x])
    # a['green_o'] = a['r_g_o'].apply(lambda x: green[x])
    # a.drop(['r_g_o'], axis=1, inplace=True)
    out_feature = ['oo', 'ho', 'lo', 'co', 'red_o', 'green_o','red_o', 'green_o','r_g_o']
    out_feature = ['oo', 'ho', 'lo', 'co','r_g_o']
    # out_feature = []
    # output = a[out_feature]
    # output.drop(['red_o','green_o'],axis=1,inplace=True)#this is for only regression code
    a.drop(out_feature, axis=1, inplace=True)
    # a.astype({'r_g_o': 'int'}).dtypes
    if date==1:
        a= a[-1:]
        open=open[-1:][0]
    else:
        a=a[date]
        open=open['date'][0]
    date=a.index[0]
    print('input date:',date)

    date=date+datetime.timedelta(1)
    while date.strftime('%A')=='Saturday' or date.strftime('%A')=='Sunday':
        date = date + datetime.timedelta(1)
    print('output date:',date)
    a['next_day']=day[date.strftime('%A')]
    a = torch.from_numpy(a.to_numpy())
    a = a.to(torch.float32)
    return (a,open)
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
y=output_tomorrow()
y.to_csv('output.csv')
# print(y)



