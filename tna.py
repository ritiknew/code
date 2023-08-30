#technical analysis
import time
import pandas as pd
import math
import datetime
import gc
# import h_data
# import fyers
def sma(df,col='close',length=14):
    sm=df[col].rolling(length).mean()
    return sm
def ema(df,col='close',length=14):
    sm=sma(df,col,length)
    k=(2/(length+1))
    em=[]
    for i in sm.index:
        if math.isnan(sm[i]):
            em.append(float('NaN'))
        else:
            if math.isnan(em[-1]):
                em.append(sm[i])
            else:
                pema=em[-1]
                em.append(k*df[col][i]+pema*(1-k))
    df['ema']=em
    x=df['ema']
    df.drop('ema',axis=1,inplace=True)
    return x
def rma(df,col='close',length=14):
    sm=sma(df,col,length)
    k=(1/(length))
    em=[]
    for i in sm.index:
        if math.isnan(sm[i]):
            em.append(float('NaN'))
        else:
            if math.isnan(em[-1]):
                em.append(sm[i])
            else:
                pema=em[-1]
                em.append(k*df[col][i]+pema*(1-k))
    df['ema']=em
    x=df['ema']
    df.drop('ema',axis=1,inplace=True)
    return x
def macd(df,f=12,s=26,sg=9,col='close'):
    fast=ema(df,col,f)
    slow=ema(df,col,s)

    m_line=fast-slow
    df['macd']=m_line
    sig_line=ema(df,'macd',sg)
    df['signal']=sig_line
    df['histogram']=m_line-sig_line
    x=df[['macd','signal','histogram']]
    df.drop('macd', axis=1, inplace=True)
    df.drop('signal', axis=1, inplace=True)
    df.drop('histogram', axis=1, inplace=True)
    return x
def rsi(df,col='close',length=14):
    pcol=df[col].shift(periods=1)
    change=df[col]-pcol
    gain=[]
    loss=[]
    for i in change:
        if i>0:
            gain.append(i)
            loss.append(0)
        elif i<0:
            gain.append(0)
            loss.append(i)
        else:
            gain.append(0)
            loss.append(0)
    df['gain']=gain
    df['loss']=loss
    df['loss']=abs(df['loss'])
    avg_gain=rma(df,'gain',length)
    avg_loss=rma(df,'loss',length)
    rs=avg_gain/avg_loss
    df['a_gain']=avg_gain
    df['a_loss']=avg_loss
    df['rs']=rs
    rsi_=[]
    for i in df.index:
        if math.isnan(df['a_gain'][i]):
            rsi_.append(float('NaN'))
        else:
            if df['a_loss'][i]==0:
                rsi_.append(100)
            else:
                rsi_.append(100-(100/(1+df['rs'][i])))
    df['rsi']=rsi_
    # x=pd.DataFrame
    x=df['rsi'].copy()
    df.drop('rsi', axis=1, inplace=True)
    df.drop('gain', axis=1, inplace=True)
    df.drop('loss', axis=1, inplace=True)
    df.drop('a_gain', axis=1, inplace=True)
    df.drop('a_loss', axis=1, inplace=True)
    df.drop('rs', axis=1, inplace=True)
    return x
def atr(df,lenght=14):
    h_l=df['high']-df['low']
    cp=df['close'].shift(periods=1)
    h_cp=df['high']-cp
    l_cp=df['low']-cp
    tr=pd.DataFrame([h_l,h_cp,l_cp]).T.max(axis=1)
    atr=tr.rolling(14).mean()
    return atr
def supertrend(df,length=14,multiplier=2):
    at=atr(df,length)
    ub=(df['high']+df['low'])/2+multiplier*at
    lb=(df['high']+df['low'])/2-multiplier*at
    pclose=df['close'].shift(periods=1)
    uf=[]
    lf=[]
    for i in df.index:
        if math.isnan(at[i]):
            uf.append(float('NaN'))
            lf.append(float('NaN'))
        else:
            if math.isnan(uf[-1]):
                uf.append(ub[i])
                lf.append(lb[i])
            else:
                if ub[i]<uf[-1] or pclose[i]>uf[-1]:
                    uf.append(ub[i])
                else:
                    uf.append(uf[len(uf)-1])
                if lb[i]>lf[-1] or pclose[i]<lf[-1]:
                    lf.append(lb[i])
                else:
                    lf.append(lf[-1])
    df['uf']=uf
    uf=df['uf']
    df.drop('uf', axis=1, inplace=True)
    df['lf']=lf
    lf=df['lf']
    df.drop('lf', axis=1, inplace=True)
    p_uf=uf.shift(periods=1)
    p_lf=lf.shift(periods=1)
    super=[]
    for i in df.index:
        if math.isnan(uf[i]):
            super.append(float('NaN'))
            continue
        else:
            if math.isnan(super[-1]):
                if df['close'][i]<=uf[i]:
                    super.append(uf[i])
                else:
                    super.append(lf[i])
            else:
                if super[-1]==p_uf[i] and df['close'][i]<=uf[i]:
                    super.append(uf[i])
                else:
                    if super[-1]==p_uf[i] and df['close'][i]>=uf[i]:
                        super.append(lf[i])
                    else:
                        if super[-1]==p_lf[i] and df['close'][i]>=lf[i]:
                            super.append(lf[i])
                        else:
                            if super[-1]==p_lf[i] and df['close'][i]<=lf[i]:
                                super.append(uf[i])

    df['supertrend'] = super
    super = df['supertrend']
    sig=[]
    for i in df.index:
        if df['close'][i]>super[i]:
            sig.append('buy')
        else:
            sig.append('sell')
    df['super_signal']=sig
    x=df[['supertrend','super_signal']]
    df.drop('supertrend', axis=1, inplace=True)
    df.drop('super_signal', axis=1, inplace=True)
    return x
def slope(df,series,n_day=1):#df,ema(df,ema(df,'close',24),3),

    if type(series)==str:
        new=df[series].shift(periods=n_day)
        df['new']=new
        slp=(((df[series]-df['new'])/df[series])/n_day)*100
        # print(slp)
        df.drop('new',axis=1,inplace=True)
        return slp
    else:
        df['series']=series
        x= df['series'].shift(periods=n_day)
        df['new']=x
        slp = (((df['series'] - df['new']) / df['series']) / n_day) * 100
        df.drop('new',axis=1,inplace=True)
        df.drop('series', axis=1, inplace=True)
        return slp
def pivot_classic(df,length=15):
    h=df['high'].rolling(length).max()
    l = df['low'].rolling(length).min()
    c=df['close'].rolling(length).mean()
    p=(h+l+c)/3
    r1=p*2-l
    r2=p+h-l
    s1=p*2-h
    s2=p-(h-l)
    x=pd.DataFrame()
    x['r2'] = r2
    x['r1']=r1
    x['p']=p
    x['s1']=s1
    x['s2']=s2
    return x
def pivot_cam(df,length=15):
    h = df['high'].rolling(length).max()
    l = df['low'].rolling(length).min()
    c = df['close'].rolling(length).mean()
    r1 = c+(h-l)*1.1/12
    r2 = c+(h-l)*1.1/6
    s1 = c-(h-l)*1.1/12
    s2 = c-(h-l)*1.1/6
    x = pd.DataFrame()
    x['r2'] = r2
    x['r1'] = r1
    x['s1'] = s1
    x['s2'] = s2
    return x

def one_signal(temp):# temp=list type
    sig=[]
    for i in range(len(temp)):
        if i==0:
            sig.append(temp[i])
        else:
            if temp[i]=='buy':
                if temp[i-1]=='buy':
                    sig.append('not')
                else:
                    sig.append(temp[i])
            elif temp[i]=='sell':
                if temp[i-1]=='sell':
                    sig.append('not')
                else:
                    sig.append(temp[i])
            else :
                sig.append('not')
    return sig


def convert_timeframe(a,days):
    # it convert dataframe as different time frame like weekly, monthly or yearly basis
    # give input dataframe and days
    b=pd.DataFrame({'open':[],'close':[],'low':[],'high':[],'volume':[]})
    series=[]
    i=0
    while i<a.shape[0]:
        if a[i:i+days].shape[0]==days:
            open= a[i:i + days][0:1]['open'][0]
            close= a[i:i + days][-1:]['close'][0]
            low= a[i:i + days]['low'].min()
            high= a[i:i + days]['high'].max()
            series=[open, close, low, high, a[i:i + days]['volume'].sum()]
            b.loc[a[i:i+days][0:1].index[0]]=series
            i= i + days
        else:
            i=i+days
    # print(b)
    return b
def date_just_before(df,date):#it return that week or month data according to date input=series ,data
    d=df.index.to_list()
    for i in d:
        if date>i:
            n=i
        else:
            return df[n]
    return df[n]
def plot(df,list=[]):
    # to be printed in order for ex if want to print ['close','open',('open','close')]
    import matplotlib.pyplot as plt
    df.dropna(inplace=True)
    if len(list)==0:
        list=df.columns.to_list()
    figure, axis = plt.subplots(len(list), 1)
    k=0
    for i in list:
        axis[k].plot(df[i])
        axis[k].grid()
        k=k+1
    plt.show()
    return
def ratio(stock_name,day=1):
    # it tells percentage of stocks rose and fallen prvious day from given list retrun ratio of rise to fall:
    rise=0
    fall=0
    total=len(stock_name)
    for i in stock_name:
        a=h_data.share(i)
        if day==1:
            a=a[-(day):]['close']-a[-(day):]['open']
        else:
            a = a[-(day):-(day - 1)]['close'] - a[-(day):-(day - 1)]['open']
        a=a[0]
        if a>=0:
            rise=rise+1
        else:
            fall=fall+1
    ratio=rise/fall
    fall=(fall/total)*100
    rise=(rise/total)*100
    # print(rise,'percent stock rose and',fall,'percent stock fall')
    return ratio
# print(ratio(fyers.rv('nifty')))
# for i in range(1,10):
#     print(i,ratio(fyers.rv('nifty'),i))



