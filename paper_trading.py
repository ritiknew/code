import pandas as pd
STD_PATH='.\\'
import tna
def tax(sell_price,buy_price,qty):
    # this function return total tax on swing trading
    brokerage_charge=20*2
    stt=sell_price*qty*0.00025
    transaction_charge=(sell_price+buy_price)*qty*0.0000345
    gst=(brokerage_charge+transaction_charge)*0.18
    sebi=1e-06*(sell_price+buy_price)*qty
    stamp_duty=buy_price*qty*0.00003
    charge=brokerage_charge+stt+transaction_charge+gst+sebi+stamp_duty
    return round(charge,2)

print(tax(100,100,400))
def trading_test(stock,param):
    a=pd.read_pickle(STD_PATH+'stock\\'+stock)
    a['mid']=(a['high']+a['low'])/2
    a.set_index('time',inplace=True)
    buy_limit=param[0]
    sell_limit=param[1]

    a['rsi']=tna.rsi(a)
    a['buy_p']=a['mid'].shift(periods=-1)
    a['sell_p']=a['mid'].shift(periods=-1)
    # date,rsi,b_p,s_p,tax,qty,profit
    a=a.dropna()
    trade_df=pd.DataFrame({'date':[],'rsi':[],'b_p':[],'s_p':[],'tax':[],'qty':[],'profit':[]})
    buy_flag=True
    for i in a.index:

        if a['rsi'][i]<buy_limit and buy_flag:
            b_p=a['buy_p'][i]
            s_p=0
            date=i
            qty=100
            profit=0
            new_row={'date':date,'rsi':a['rsi'][i],'b_p':b_p,'s_p':s_p,'tax':0,'qty':qty,'profit':profit}
            trade_df=trade_df.append(new_row,ignore_index=True)
            buy_flag=False
        elif a['rsi'][i]>sell_limit and not(buy_flag):
            s_p = a['sell_p'][i]
            date = i
            qty = 100
            profit = s_p-b_p
            profit=profit/b_p*100
            tax_=tax(s_p,b_p,100)
            new_row = {'date': date, 'rsi': a['rsi'][i], 'b_p': b_p, 's_p': s_p, 'tax': tax_, 'qty': qty, 'profit': profit}
            trade_df = trade_df.append(new_row, ignore_index=True)
            buy_flag=True
        else:
            pass


    trade_df.to_csv(STD_PATH+'stock\\PNB.csv')

# trading_test('PNB',[55,60])

#         sell a stock



def trading_():

    stock=['HDFCBANK','ICICIBANK','SBIN','RELIANCE','TITAN','INFY','TATAMOTORS','TCS',
           'NTPC','ITC','TATASTEEL','CIPLA','ADANIGREEN','TATAPOWER', 'POWERGRID','GAIL']
    df=pd.DataFrame({'stock':[],'profit':[],'buy_date':[],'sell_date':[],'no_of_days':[],'buy_price':[],'sell_price':[]})
    for k in stock:
        print(k)
        a=pd.read_pickle(STD_PATH+'stock\\'+k)
        a['date']=a['time']
        a.set_index('time',inplace=True)
        a['ordinal']=a['date'].apply(lambda x: x.toordinal())
        a['date_back']=a['date'].shift(periods=1)
        # a['ordinal'] = a['date_back'].apply(lambda x: x.toordinal())
        a['ordinal_']=a['ordinal'].shift(periods=1)
        a.set_index('ordinal_',inplace=True)
        a['mid']=(a['high']+a['low'])/2
        for i in range(3,100):
            # i is no of days
            for j in range(int((a.shape[0]-300)-i*1.3)):
                date_b=a.index[300+j]
                date_s=a.index[300+j+i]
                profit=a['mid'][date_s]-a['mid'][date_b]
                profit=profit/a['mid'][date_b]
                profit=profit*100
                if profit>10:
                    new_row=pd.DataFrame({'stock':[k],'profit':[profit.round(1)],'buy_date':[date_b],
                             'sell_date':[date_s],'no_of_days':[date_s-date_b],'buy_price':[a['mid'][date_s]],'sell_price':[a['mid'][date_b]]})
                    # new_row=pd.DataFrame(new_row)
                    df = pd.concat([df,new_row], ignore_index=True)
        # df.to_pickle(STD_PATH+'swing\\'+k)
    df.to_pickle(STD_PATH+'swing\\df')

#rsi, macd, macd difference, supertrend, low of time, high of time, no of days, sma (4,8,16,32), slope (close,smae 4,8,16,32)(2,4,6),o,h,l,c,buy_sell

# array=[rsi, macd_1, macd_2, difference_macd, supertrend,low_of_time, high_of_time, no_of_days, sma_4,sma_8,sma_16,s_sma4_2,s_sma4_
# s_sma4_4,s_sma4_6,s_sma8_2,s_sma8_4,s_sma8_6,s_sma16_2,s_sma16_,s_sma16_2,s_sma16_2,s_sma16_2slope (close,smae 4,8,16,32)(2,4,6),o,h,l,c,buy_sell
                # print(k,profit.round(1),date_b,date_s,i,j,date_s-date_b,a['mid'][date_s],a['mid'][date_b])

def one():
    a=pd.read_pickle('.\\swing\\df')
    c=a['stock'].unique()
    df=pd.DataFrame({'stock:':[],'buy_date':[],'sell_date':[]})
    for i in c:
        d=a[a['stock']==i]
        buy_date=d['buy_date'].unique().tolist()
        sell_date=d['sell_date'].unique().tolist()
        k=max(len(sell_date),len(buy_date))
        while len(sell_date)<k:
            sell_date.append(sell_date[-1])
        while len(buy_date) < k:
            buy_date.append(buy_date[-1])
        for j in range(k):
            row=pd.DataFrame({'stock:':[i],'buy_date':[buy_date[j]],'sell_date':[sell_date[j]]})
            df = pd.concat([df,row], ignore_index=True)
    df.to_pickle('.\\swing\\uniquedf')
    print('good')
# trading_()
def create_data():
    import paper_trading_other as pto
    # data for ml for swing learning
    a=pd.read_pickle('.\\swing\\uniquedf')
    stock_data = pd.read_pickle(STD_PATH + 'stock\\PNB')
    stock_data = pto.ready_for_learning(stock_data)
    output=stock_data[stock_data['close']==55]
    output['output_buy'] = []
    output['output_sell'] = []
    # output=pd.DataFrame({schema of previous df +buy_sell_otput})
    unique_stock=a['stock:'].unique().tolist()
    for i in unique_stock:
        stock_data=pd.read_pickle(STD_PATH+'stock\\'+i)
        stock_data=pto.ready_for_learning(stock_data)
        temp=a[a['stock:']==i]
        for j in temp['buy_date']:
            new_row=stock_data.loc[[int(j)]]
            new_row['output_buy']=[1]
            new_row['output_sell']=[0]
            output=pd.concat([output,new_row],ignore_index=True)
        for j in temp['sell_date']:
            new_row=stock_data.loc[[int(j)]]
            # new_row['output'] = ['sell']
            new_row['output_buy'] = [0]
            new_row['output_sell'] = [1]
            output = pd.concat([output, new_row], ignore_index=True)
    output.to_pickle('.\\swing\\learning')


# one()
create_data()