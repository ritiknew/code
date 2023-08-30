import pickle
import time

import pandas as pd
import datetime
import pyautogui as pt
STD_PATH='.\\'
def internet():
    import requests
    try:
        res = requests.get('https://www.google.com')
        return True
    except:
        return False
def rv(name,location=STD_PATH+'fy\\'):
    # this function retrieve any data from file given location having name
    f = open(location+name, 'rb')
    k = pickle.load(f)
    f.close()
    return k
def sv(data,name,location='C:\\Users\\lenovo\\PycharmProjects\\intraday\\fy\\' ):
    # this function store data in in inay folder given location and name and data
    f = open(location+name, 'wb')
    pickle.dump(data, f)
    f.close()
    return 0

def symbol_name(symbol):#convert symbol to name eg:-NSE:SBIN-EQ to SBIN
    symbol=symbol[symbol.find(':')+1:]
    symbol=symbol[:symbol.find('-')]
    # symbol=
    return symbol
def fy_auth():
    import time
    import urllib.parse as urlparse
    from selenium import webdriver
    from fyers_api import fyersModel
    from fyers_api import accessToken
    from selenium.webdriver.common.by import By
    #
    fyers_id="XR00456"
    password="Papa@000"
    pan_dob='DQGPR0571G'
    client_id = 'I2IEW27POR-100'
    secret_key = 'HH1AP7JEYD'
    redirect_uri = 'https://www.google.com'
    response_type = "code"
    grant_type = "authorization_code"
    state = "None"
    nonce = "private"
    browser = webdriver.Firefox()
    session=accessToken.SessionModel(client_id=client_id,secret_key=secret_key,redirect_uri=redirect_uri
                                     ,response_type=response_type,grant_type=grant_type)
    url=session.generate_authcode()
    browser.get(url)
    # -----------------
    # time.sleep(5)
    # # browser.find_element(by=By.XPATH, value="//input[@id='fy_client_id']").send_keys(fyers_id)
    # browser.find_element(by=By.XPATH, value="//input[@id='fy_client_id']").send_keys(fyers_id)
    # # browser.find_element_by_xpath("//input[@id='fy_client_id']").send_keys(fyers_id)
    # time.sleep(5)
    # browser.find_element(by=By.XPATH, value="//button[@id='clientIdSubmit']").click()
    # # browser.find_element_by_xpath("//button[@id='clientIdSubmit']").click()
    # time.sleep(25)
    # # browser.find_element(by=By.XPATH, value="//input[@id='fy_client_pwd']").send_keys(password)
    # # browser.find_element_by_xpath("//input[@id='fy_client_pwd']").send_keys(password)
    # time.sleep(5)
    # # browser.find_element(by=By.XPATH, value="//button[@id='loginSubmit']").click()
    # # browser.find_element_by_xpath("//button[@id='loginSubmit']").click()
    # time.sleep(5)
    # pt.typewrite('2')
    # pt.typewrite('2')
    # pt.typewrite('4')
    # pt.typewrite('1')
    # # browser.find_element_by_xpath("//input[@id='pancard']").send_keys(pan_dob)
    # browser.find_element(by=By.XPATH, value="//button[@id='verifyPinSubmit']").click()
    # # browser.find_element_by_xpath("//button[@id='verifyPinSubmit']").click()
    # time.sleep(20)
    # --------------------------------
    pt.alert()
    current_url=browser.current_url
    browser.close()
    print(current_url)
    parsed=urlparse.urlparse(current_url)
    auth_code=urlparse.parse_qs(parsed.query)['auth_code'][0]
    session.set_token(auth_code)
    response=session.generate_token()
    print(response)
    sv(response['access_token'],'tok')
    fyers = fyersModel.FyersModel(client_id=client_id, token=rv('tok'), log_path=".//fyers")
    sv(fyers,'fyers')
    sv(datetime.datetime.now().date(),'fyers_flag')
    return

def daily_data(name,candle_lenght,start=0,end=0):
    #return intraday data of particular length up to a particular time
    # k=datetime.datetime(2022, 6, 3,15,0)
    k=datetime.datetime.now()
    now=int(k.timestamp())
    hr=9
    minute=15
    start=datetime.datetime(k.date().year,
                            k.date().month,
                            k.date().day,hr,minute,0).timestamp()
    start=int(start)
    data = {"symbol": 'NSE:'+name+'-EQ', "resolution": candle_lenght, "date_format": 0, "range_from": str(start),
            "range_to": str(now), "cont_flag": 1}
    print(data)
    fyers=rv('fyers')
    d=fyers.history(data)
    time.sleep(1)
    d=pd.DataFrame(d['candles'], columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    d.set_index('time')
    return d
def xhistorical_data(name,start,end=datetime.datetime.now(),candle_lenght='D'):
    #return historical data of particular length up to a particular time
    # k=datetime.datetime(2022, 6, 3,15,0)

    # k=datetime.datetime.now().date()


    now=int(end.timestamp())
    # hr=9
    # minute=15
    # start=datetime.datetime(k.date().year,
    #                         k.date().month,
    #                         k.date().day,hr,minute,0).timestamp()
    start=int(start.timestamp())

    data = {"symbol": 'NSE:'+name+'-EQ', "resolution": candle_lenght, "date_format": 0, "range_from": str(start),
            "range_to": str(now), "cont_flag": 1}
    print(data)
    fyers=rv('fyers')
    d=fyers.history(data)
    time.sleep(1)
    if d==None:
        while internet()==False:
            time.sleep(.5)
        d = fyers.history(data)

    d=pd.DataFrame(d['candles'], columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    d.set_index('time')
    return d
def current(name):
    #return name='sbi' cureent condition of stock
    fyers = rv('fyers')
    data = {"symbols":'NSE:'+name+'-EQ'}
    data=fyers.quotes(data)
    time.sleep(1)
    if data['s']=='ok':
        return data
    else:
        raise Exception('status not ok check code')
def depth(name):
    data = {"symbol":'NSE:'+name+'-EQ', "ohlcv_flag": 1}
    fyers = rv('fyers')
    data=fyers.depth(data)
    time.sleep(1)
    if data['s']=='ok':
        return data
    else:
        raise Exception('status not ok check code')
# depth('SBIN')

def historical_data(name,start,end=datetime.datetime.now(),candle_lenght='D'):
    # return dataframe of stock containing o,h,l,c ,volume of name='SBI',start=date in datetime format,end=today date or in datetime format,candle length
    y=pd.DataFrame()
    while (end - start).days >= 365:
        new_start = end - datetime.timedelta(days=360)
        x = xhistorical_data(name, new_start, end)
        x['time']=x['time'].apply(lambda x:datetime.datetime.fromtimestamp(x).date())
        y=pd.concat([x,y])
        end=new_start-datetime.timedelta(days=1)
    if (end - start).days >0:
        x = xhistorical_data(name,start, end)
        x['time'] = x['time'].apply(lambda x: datetime.datetime.fromtimestamp(x).date())
        y = pd.concat([x,y])
    return y
start=datetime.datetime(2021,6,1)-datetime.timedelta(500*1.28)
fy_auth()
# a=historical_data('PNB',start=start)
# a.to_pickle(STD_PATH+'stock//PNB')
# print(historical_data('PNB',start=start))
# name=pd.read_csv('nse_list.csv')
# name=name['name'].to_list()
# name='SBIN'
# name=['RELIANCE','TCS','HINDUNILVR','KOTAKBANK','ICICIBANK','SBIN','WIPRO','AXISBANK','TATASTEEL','TATAMOTORS','ADANIENT','FACT']
# name=['HDFCBANK','ICICIBANK','SBIN','RELIANCE','TITAN','INFY','TATAMOTORS','TCS',
#        'NTPC','ITC','TATASTEEL','CIPLA','ADANIGREEN','TATAPOWER', 'POWERGRID','GAIL']
# for i in name:
#     a=historical_data(i,start=start)
#     a.to_pickle('.\\stock\\'+i)
# print(internet())

# def store_stock_data(name):
#     a=historical_data(name,start=start)
#     a.to_pickle(name)
# store_stock_data(name)
