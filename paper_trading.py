import pandas as pd
STD_PATH='C:\\Users\\lenovo\\PycharmProjects\intraday\\'
import tna
import check
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

# print(tax(100,100,400))
def trading_test(name):
    import torch
    stock = ['HDFCBANK', 'ICICIBANK', 'SBIN', 'RELIANCE', 'TITAN', 'INFY', 'TATAMOTORS', 'TCS',
             'NTPC', 'ITC', 'TATASTEEL', 'CIPLA', 'ADANIGREEN', 'TATAPOWER', 'POWERGRID', 'GAIL']
    stock_model=[]
    for i in stock:
        model=torch.jit.load(STD_PATH+'model_sell_10\\'+i)
        model.eval()
        stock_model.append(model)
    a=pd.read_pickle(STD_PATH+'stock\\'+name)
    a.set_index('time', inplace=True)
    a['time'] = a.index
    x = [1]
    # slope of close value of 2,4,16,32, days
    for i in x:
        a['cs' + str(i)] = tna.slope(a, 'close', i)
    sma = {'sma2': 2, 'sma4': 4, 'sma8': 8, 'sma16': 16, 'sma32': 32, 'sma64': 64, 'sma128': 128}
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
    a.drop(
        ['open', 'high', 'low', 'close', 'volume', 'sma2', 'sma4', 'sma8', 'sma16', 'sma32', 'sma64', 'sma128', 'date'],
        axis=1, inplace=True)
    a=a.dropna()
    start_index=a.index[0]
    sell=[]
    buy=[]
    nothing=[]
    for i in a.index:
        row=a.loc[i].values
        row_tensor=torch.Tensor(row)
        sell_t=0
        buy_t=0
        nothing_t=0
        # count=0
        for j in stock_model:
            output=j(row_tensor).tolist()
            sell_t=sell_t+output[0]
            buy_t=buy_t+output[1]
            nothing_t=nothing_t+output[2]
        count=stock_model.__len__()
        sell.append(sell_t/count)
        buy.append(buy_t/count)
        nothing.append(nothing_t/count)
    a = pd.read_pickle(STD_PATH + 'stock\\'+name)
    a.set_index('time', inplace=True)
    a['time'] = a.index
    a['date'] = a['time']
    a.set_index('time', inplace=True)
    a['ordinal'] = a['date'].apply(lambda x: x.toordinal())
    a.set_index('ordinal', inplace=True)
    a.drop(['date'],axis=1, inplace=True)
    a=a[a.index>=start_index]
    a['buy']=buy
    a['sell']=sell
    a['nothing']=nothing
    a.to_pickle(STD_PATH+'output_sell_10\\'+name)
    return






    # a = torch.from_numpy(a.to_numpy())



def trading_(a):
    # this function get file of stock and  store data in new file of profit above percentage

    a['close']=a['close'].shift(periods=-1)
    a=a.dropna()
    buy2_date=0
    buy1_date=0
    stock_hold=False
    buy_price=0
    sell_price=0
    buy_date=0
    sell_date=0
    flag=True
    x=pd.DataFrame({'buy_date':[],'sell_date':[],'buy_price':[],'sell_price':[],'no_of_days':[],'profit':[]})
    amt=100000
    invested=0
    invested_qty=0
    while flag:
        stock_hold = False
        for i in a.index:
            if i==buy2_date:
                flag=False
                break
            price=a.loc[i]['close']
            buy=a.loc[i]['buy']
            sell=a.loc[i]['sell']
            no_of_day=i-buy_date
            stoploss=0.02
            # nothing=a.loc[i]['nothing']
            if price<buy_price*(1-stoploss) and stock_hold and no_of_day>2:
                sell_date = i
                sell_price = price
                stock_hold = False
                print(buy_date, sell_date, buy_price, sell_price, sell_date - buy_date,
                      (sell_price - buy_price) / buy_price * 100)
                new_row = pd.DataFrame({'buy_date': [buy_date], 'sell_date': [sell_date], 'buy_price': [buy_price]
                                           , 'sell_price': [sell_price], 'no_of_days': [no_of_day],
                                        'profit': [(sell_price - buy_price) / buy_price * 100]})
                x = pd.concat([x, new_row], ignore_index=True)
                t = tax(sell_price, buy_price, invested_qty)
                amt = sell_price * invested_qty - t + amt
                print('total amt', amt, 'tax', t, 'qty', invested_qty)
                invested = 0
                invested_qty = 0

            if buy>.70 or sell>.70:
                if not stock_hold and buy >.7 and sell<.6 :
                    buy_date=i
                    if buy1_date==0:
                        buy1_date=i
                    elif buy2_date==0:
                        buy2_date=i
                    else:
                        pass
                    buy_price=price
                    stock_hold=True
                    invested_qty=int(amt/buy_price)
                    invested=invested_qty*buy_price
                    amt=amt-invested
                else:
                    if sell>.7 and no_of_day>2 and stock_hold and buy<.6:
                        sell_date=i
                        sell_price=price
                        stock_hold=False
                        print(buy_date,sell_date,buy_price,sell_price,sell_date-buy_date,(sell_price-buy_price)/buy_price*100)
                        new_row=pd.DataFrame({'buy_date':[buy_date],'sell_date':[sell_date],'buy_price':[buy_price]
                                                  ,'sell_price':[sell_price],'no_of_days':[no_of_day],
                                               'profit':[(sell_price-buy_price)/buy_price*100]})
                        x=pd.concat([x,new_row],ignore_index=True)
                        t=tax(sell_price,buy_price,invested_qty)
                        amt=sell_price*invested_qty-t+amt
                        print('total amt', amt, 'tax', t, 'qty', invested_qty)
                        invested=0
                        invested_qty=0


            else:
                pass
        a=a[1:]





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
# create_data()
# trading_()


# # Now, file_names is a list of file names
# # print(file_names)
# for i in file_names:
#     trading_test(i)
import os
directory_path = STD_PATH + '\\output_sell_10'
file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
for i in file_names:
    a=pd.read_pickle(STD_PATH+'\\output_sell_10\\'+i)
    trading_(a)
    print(i)