import requests
import pandas as pd
# import arrow
import datetime
import tna
def ready_for_learning(a):
    # a is dataframe for the object
    a=a.copy()
    a.set_index('time',inplace=True)
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
    a['vo']=a['next_volume']/a['mv']
    a['oi']=a['open']/a['open']
    a['hi'] = a['high'] / a['open']
    a['li'] = a['low'] / a['open']
    a['ci'] = a['close'] / a['open']
    a['oo'] = a['next_open'] / a['open']
    a['co'] = a['next_close'] / a['open']
    a['ho'] = a['next_high'] / a['open']
    a['lo'] = a['next_low'] / a['open']
    a['rsi']=tna.rsi(a,'close',14)/100
    x=[2,4,16,32]
    x = [2, 4,16]
    # slope of close value of 2,4,16,32, days
    for i in x:
        a['cs'+str(i)]=tna.slope(a,'close',i)
    sma={'sma4':4,'sma16':16,'sma32':32,'sma64':64}
    ema = {'ema4':4, 'ema16':16, 'ema32':32, 'ema64':64}
    sma = {'sma16': 16, 'sma64': 64}
    ema = {'ema4': 4, 'ema16': 8}
    # sma and ema value  of 4,16,32,64
    for i in sma:
        a[i]=tna.sma(a,'close',sma[i])
        a[i+'i']=a[i]/a['open']
    for i in ema:
        a[i]=tna.ema(a,'close',ema[i])
        a[i+'i']=a[i]/a['open']
    for i in sma:
        for j in x:
            a[i+'s' + str(j)] = tna.slope(a, i, j)
    for i in ema:
        for j in x:
            a[i+'s' + str(j)] = tna.slope(a, i, j)

    def r_g(open,close):
        if open > close:
            return 'red'
        else:
            return 'green'

    a['r_g_i'] = a.apply(lambda x: r_g(x['open'], x['close']), axis=1)
    a['r_g_o']=a['r_g_i'].shift(periods=-1)
    a.drop(['high', 'low', 'open','close','mv','vo','oi','volume', 'time', 'next_open', 'next_high', 'next_low','next_close', 'next_volume'], axis=1,inplace=True)
    for i in sma:
        a.pop(i)
    for i in ema:
        a.pop(i)
    day={'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Monday':0,'Sunday':6,'Saturday':5}
    r_g={'red':0, 'green':1}
    red={'red':1, 'green':0}
    green={'red':0,'green':1}
    a['day'] = a['day'].apply(lambda x:day[x])
    a['next_day'] = a['day'].shift(periods=-1)
    # a.astype({'next_day': 'int'}).dtypes
    a['r_g_o'] = a['r_g_i'].shift(periods=-1)
    a['r_g_i'] = a['r_g_i'].apply(lambda x:r_g[x])
    a = a.dropna()
    a['red_o']=a['r_g_o'].apply(lambda x:red[x])
    a['green_o']=a['r_g_o'].apply(lambda x:green[x])
    a.drop(['r_g_o'],axis=1,inplace=True)
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
