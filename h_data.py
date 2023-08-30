import nsepy
import pandas as pd
import datetime
import gc
def new_nse_data(stock_name,start_date, end_date,location):
    # this function store  data frame of o,h,l,c form as pickle in file in given location
    data = nsepy.get_history(symbol=stock_name, start=start_date, end=end_date)
    print(stock_name)
    # print(stock_name)
    data.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'},inplace=True)
    data=data[['open','high','low','close','volume']]
    data['i']=data.index
    data=data.drop_duplicates(subset='i')
    data.drop('i',axis=1,inplace=True)
    data.to_pickle(location+stock_name)#specipic stock complete O H L C V
    return
# new_nse_data('SBIN','')
def multi(stock_name_list,start_date, end_date,location):
    # return multiple data at once
    # give stock_name_list and start date ,end date,loation where needs to be stored
    # head of stock name should be name
    for i in stock_name_list:
        new_nse_data(i,start_date,end_date,location)
    return
def nse_open_gen(date):#date in string format
    # This function return true if nse open on the date
    # date="2021-01-26"
    off = ['2021-11-05', '2022-07-09','2022-07-15','2022-07-16','2022-07-31','2022-09-05','2022-09-24','2022-09-26','2022-11-08','2021-01-26', '2021-03-11', '2021-03-29', '2021-04-02', '2021-04-14', '2021-04-21', '2021-05-13',
           '2021-07-21', '2021-08-19', '2021-09-10', '2021-10-15', '2021-11-04', '2021-11-05', '2021-11-19','2022-05-03','2022-05-16','2022-04-01','2022-04-14','2022-04-15'
           '2022-03-01','2022-03-18','2022-01-26']
    d = date.split("-")
    d = datetime.date(int(d[0]), int(d[1]), int(d[2]))
    if str(date) in off or d.weekday() == 6 or d.weekday() == 5:
        return False
    else:
        return True
def nse_data(no_of_day,name_list,location):
    #no_of days=no of day of which data of any particular stock which is in the name list is to be fetched and stored
    #eg nse_data(100,['SBI','PNB'],'C:\\Users\\lenovo\\PycharmProjects\\intraday\\stock_history\\'  )
    try:
        new = []
        remove=[]
        add=[]
        stock_list =name_list
        date_list = []#keep date as date type object
        date=[]
        bhav_copy={}
        start = datetime.datetime.now().date() - datetime.timedelta(days=1)
# list of date -------total 200-----------
        while len(date_list) < no_of_day:
            if nse_open_gen(str(start)):
                date_list.append(start)
            start = start - datetime.timedelta(days=1)
        print(date_list[0], date_list[len(date_list) - 1])
        end_date = date_list[0]
        start_date = date_list[len(date_list) - 1]
#-------------------------------------getting  new name or some invalid names---------
        for i in stock_list:
            try:
                stock = pd.read_pickle(location+ i)
                if stock.shape[0]>=no_of_day:
                    date=stock.index
                else:
                    new.append(i)
                del stock
            except:
                new.append(i)# new stock added in list for trading
        gc.collect()
#-------------remove date and add adate finder-------
        if len(date)!=0:
            for i in date_list:
                if (i-date[-1]).days>0:#  i is recent [12,11,10,8] and date is [1,2,3,4]
                    add.append(i)# add=[12,11,10]
                if (i-date[0]).days<0:
                    add.append(i)
            for i in date:
                if (date_list[-1]-i).days>0:
                    remove.append(i)# remove=[12,11,10]
        del date_list,date
        gc.collect()
#--------------------------------------------------------
        if len(add)>10:
            renew=False# means more than 10 days are needed to fill in every stocks
        else:
            renew=True
#--------------storing daily bhav copy according to add date
        if renew:
            for i in add:
                print('getting bhav copy of '+str(i))
                bhav=nsepy.history.get_price_list(dt=i)
                bhav_copy[i]=bhav
                del bhav
                gc.collect()
#---------------------------
        for i in stock_list:
            if renew==False:
                new_nse_data(i, start_date, end_date,location)
                gc.collect()
                continue
            if i in new:
                continue
            stock=pd.read_pickle(location+i)
            if stock.shape[0]<no_of_day:#if any stock is
                print('\n'+str(i)+" not in full length,actual length is"+str(stock.shape[0])+'date:'+str(datetime.datetime.now().date()))
            for j in range(len(add)):
                temp=bhav_copy[add[j]]# bhav copy of that date
                try:
                    temp=temp[temp['SYMBOL']==i]
                    temp.rename(columns={'OPEN':'open','HIGH':'high','LOW':'low','CLOSE':'close','TOTTRDQTY':'volume'},inplace=True)
                    temp=temp[['open','high','low','close','volume']]
                    temp=temp.rename(index={temp.index[0]:add[j]})
                    stock = pd.concat([stock, temp])
                except:
                    print(str(i)+'does not traded on '+str(add[j]))
                    print('\n'+str(i)+'does not traded on '+str(add[j]))
                del temp
                gc.collect()
            for j in remove:
                stock=stock[stock.index!=j]
            stock.sort_index(inplace=True)
            stock['i'] = stock.index
            stock = stock.drop_duplicates(subset='i')
            stock.drop('i', axis=1, inplace=True)
            stock.to_pickle(location+i)
            del stock
            gc.collect()
        if renew:
            for i in new:
                new_nse_data(i,start_date,end_date,location)
        gc.collect()
        return
    except Exception as argument:
        print(argument,'error in nse data')
        return

def share(name,location='C:\\Users\\lenovo\\PycharmProjects\\intraday\\stock_history\\'):
    # it print and return stock data from given location if available
    a=pd.read_pickle(location+name)
    # print(a)
    return a
# a=share('SBIN')
def verify(stock_list,day,location):
    # it verifies if stocks of the given list updated till latest date at that
# location,if not returns list of only updated stocks and print list of not updated stock
    k=[]
    start = datetime.datetime.now().date() - datetime.timedelta(days=1)
    while True:
        if nse_open_gen(str(start)):
            last_date = start
            break
        start = start - datetime.timedelta(days=1)
    for i in stock_list:
        a = pd.read_pickle(location + i)
        if a.shape[0]<day:
            k.append(i)
            continue
        a = a.index
        a = a[-1]
        if last_date == a:
            pass
        else:
            k.append(i)
            print(i, 'is not updated last date =', a)
    for i in k:
        stock_list.remove(i)
    print('not updated',k)
    print('updated',stock_list)
    return stock_list
def no_of_days(date,c=str(datetime.datetime.now().date())):
    # it returns number of days nse open in between date
    # date format should be '2022-01-25' like no_of_days('2022-01-25','2022-03-22')
    # nse_off = get('nse_off')
    if type(date)!=str:
        date=str(date)
    temp = date.split('-')
    b=datetime.date(int(temp[0]),int(temp[1]),int(temp[2]))
    temp = c.split('-')
    c = datetime.date(int(temp[0]), int(temp[1]), int(temp[2]))
    off = 0
    for i in range((c - b).days):
        if nse_open_gen(str(b+datetime.timedelta(days=i))):
            off = off + 1

    return off
# a=share('SBI')
# print(a)
# print(tna.convert_timeframe(a,20))
# nse_data(100,['SBIN'],'C:\\Users\\lenovo\\PycharmProjects\\intraday\\stock_history\\')
# nse_data(100,['SBIN'],'C:\\Users\\lenovo\\PycharmProjects\\intraday\\stock_history\\')
