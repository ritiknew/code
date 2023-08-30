import datetime
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
    open = 0
    manhour=0
    for i in range((c - b).days):
        temp=b+datetime.timedelta(days=i)
        if nse_open_gen(str(b+datetime.timedelta(days=i))):
            if temp.weekday()==5:
                open=open+280
            else:
                open = open + 520

    return open/60
def nse_open_gen(date):#date in string format
    # This function return true if nse open on the date
    # date="2021-01-26"
    off = ['2021-11-05', '2022-08-15','2022-10-02','2022-10-04','2022-10-05','2022-10-06','2022-10-25','2022-10-31','2022-11-08','2023-01-15','2023-01-26'
        ,'2023-03-11','2023-03-18','2023-03-18','2023-04-15','2023-05-03']
    d = date.split("-")
    d = datetime.date(int(d[0]), int(d[1]), int(d[2]))
    if str(date) in off or d.weekday() == 6 :#or d.weekday() == 5:
        return False
    else:
        return True
no_of_days('2022-06-19','2023-03-31')