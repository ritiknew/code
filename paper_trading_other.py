import requests
import pandas as pd
# import arrow
import datetime
import tna
def ready_for_learning(a):
    # a is dataframe for the object
    a=a.copy()
    a.set_index('time',inplace=True)

    # day name
    a['mid']=(a['high']+a['low'])/2

    a['rsi']=tna.rsi(a.copy())/100
    macd=tna.macd(a.copy())
    a['macd']=macd['macd']
    a['signal_macd']=macd['signal']
    a['histogram']=macd['histogram']
    buy_sell={'buy':1,'sell':0}
    a['super_signal']=tna.supertrend(a)['super_signal']
    a['super_signal']=a['super_signal'].apply(lambda x:buy_sell[x])
    a['h7'] = a['high'].rolling(7).max()/a['mid']
    a['l7'] = a['low'].rolling(7).min()/a['mid']
    a['h30'] = a['high'].rolling(30).max() / a['mid']
    a['l30'] = a['low'].rolling(30).min() / a['mid']
    a['h52'] = a['high'].rolling(200).max() / a['mid']
    a['l52'] = a['low'].rolling(200).min() / a['mid']



    # x=[2,4,16,32]
    x = [2, 4,8]
    # slope of close value of 2,4,16,32, days
    for i in x:
        a['cs'+str(i)]=tna.slope(a,'close',i)
    sma={'sma2':2,'sma4':4,'sma8':8,'sma16':16}
    for i in sma:
        a[i]=tna.sma(a,'close',sma[i])
        a[i]=a[i]/a['mid']
    # for i in ema:
    #     a[i]=tna.ema(a,'close',ema[i])
    #     a[i+'i']=a[i]/a['open']
    for i in sma:
        for j in x:
            a[i+'s' + str(j)] = tna.slope(a, i, j)
    for i in sma:
        # a[i]=tna.sma(a,'close',sma[i])
        a[i]=a[i]/a['mid']

    a['open']=a['open']/a['mid']
    a['close']=a['close']/a['mid']
    a['low']=a['low']/a['mid']
    a['high']=a['high']/a['mid']
    a['ordinal'] = a.index
    a['ordinal'] = a['ordinal'].apply(lambda x: x.toordinal())
    # a['date_back'] = a['date'].shift(periods=1)
    # a['ordinal_'] = a['ordinal']
    a.set_index('ordinal', inplace=True)
    # a['time'] = a.index
    a.drop(['volume','mid'], axis=1, inplace=True)
    a = a.dropna()
    # a.astype({'r_g_o': 'int'}).dtypes
    return a
def in_out_split(a,drop_na=True):
    a=a.copy()
    if drop_na:
        a.dropna(inplace=True)
    out_feature = ['oo', 'ho', 'lo', 'co']#for regression
    # out_feature = ['red_o', 'green_o']#for candle
    output = a[out_feature]
    # output.drop(,axis=1,inplace=True)#this is for only regression code
    a.drop(['oo', 'ho', 'lo', 'co','red_o', 'green_o'], axis=1, inplace=True)
    return (output,a)
def train_test_split(out,input,train=80):
    out=out.copy()
    input=input.copy()
    if train>100:
        return
    train_size=int(out.shape[0]*(train/100))
    # test_size=out.shape[0]-train_size
    out_train=out[:train_size]
    out_test=out[train_size:]
    input_train = input[:train_size]
    input_test = input[train_size:]
    return (out_train,out_test,input_train,input_test)
# a=pd.read_pickle('.\\stock\\PNB')
# ready_for_learning(a)