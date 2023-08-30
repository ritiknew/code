#test other stocks data on trained model and check accuracy using given fuction
# input df of stock and convert data frame for each model and test accuracy as per function
import torch
import pickle
import other as ot
import pandas as pd
# import torch.nn as nn
# import tensor
# import ohlc
import os
def accuracy_test(pred,label):
    diff=pred-label
    diff=abs(diff/label)*100
    first=diff[:,:1].clone().view(1,-1)[0]
    second = diff[:, 1:2].clone().view(1, -1)[0]
    third = diff[:, 2:3].clone().view(1, -1)[0]
    fourth = diff[:, 3:4].clone().view(1, -1)[0]
    print('accuracy overall:',100-diff.sum()/diff.numel(),' other accuracy: ',100-first.sum()/first.numel(),
          100-second.sum()/second.numel(),100-third.sum()/third.numel(),100-fourth.sum()/fourth.numel())
def accuracy_r_g(y_true,y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    # Calculate accuracy, precision, recall, and F1 score
    rounded_predictions = torch.round(y_pred)
    y_pred = rounded_predictions.numpy().flatten()
    y_true = y_true.numpy().flatten()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
def in_out_split_r(a,drop_na=True):
    a=a.copy()
    if drop_na:
        a.dropna(inplace=True)
    out_feature = ['oo', 'ho', 'lo', 'co', 'red_o','green_o']
    out_feature = ['red_o', 'green_o']
    output = a[out_feature]
    # output.drop(['red_o','green_o'],axis=1,inplace=True)#this is for only regression code
    a.drop(out_feature, axis=1, inplace=True)
    return (output,a)
def in_out_split_ohlc(a,drop_na=True):
    a=a.copy()
    if drop_na:
        a.dropna(inplace=True)
    out_feature = ['oo', 'ho', 'lo', 'co', 'red_o','green_o']
    # out_feature = ['red_o', 'green_o']
    output = a[out_feature]
    output.drop(['red_o','green_o'],axis=1,inplace=True)#this is for only regression code
    a.drop(out_feature, axis=1, inplace=True)
    return (output,a)
def rv(name,location='C:\\Users\\lenovo\\PycharmProjects\\intraday\\fy\\'):
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
def save_model(model,class_name,model_name):
    sv({'model_dict':model.state_dict(),'class':class_name},model_name)
    return
def retrieve_model(model_name):
    model=rv(model_name)
    m=model['class']
    m.load_state_dict(model['model_dict'])
    return m

def test(name_stock):
# ready for red_green class
    a=pd.read_pickle('.\\stock\\'+name_stock)
    a=ot.ready_for_learning(a)
    a.drop(['hi', 'li','ci','oo','co','ho','lo'], axis=1,inplace=True)
    label,data=in_out_split_r(a)#output , data(input)
    # input_train,input_test=b,c
    train_data=torch.from_numpy(data.to_numpy())
    train_data=train_data.to(torch.float32)
    train_labels=torch.from_numpy(label.to_numpy())
    train_labels=train_labels.to(torch.float32)
    # model=tensor.retrieve_model('r_g')
    model = torch.jit.load('tensor_model.pt')
    model.eval()

    accuracy_r_g(train_labels,model(train_data).detach())
    # for regression data----------------------------
    a=pd.read_pickle('.\\stock\\'+name_stock)
    a=ot.ready_for_learning(a)
    # a.drop(['hi', 'li','ci','oo','co','ho','lo'], axis=1,inplace=True)
    label,data=in_out_split_ohlc(a)#output , data(input)
    # input_train,input_test=b,c
    train_data=torch.from_numpy(data.to_numpy())
    train_data=train_data.to(torch.float32)
    train_labels=torch.from_numpy(label.to_numpy())
    train_labels=train_labels.to(torch.float32)
    # model=ohlc.retrieve_model('ohlc')
    model = torch.jit.load('ohlc_model.pt')
    model.eval()

    accuracy_test(train_labels,model(train_data).detach())
dir_list = os.listdir('.\\stock')
for i in dir_list:
    print(i)
    test(i)
# model=retrieve_model('ohlc')
# rv('ritik')