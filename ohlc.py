# import tensorflow as tf
import pandas as pd
import torch
import other as ot
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy
import math
import pickle
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
a=pd.read_pickle('pnb')
a=ot.ready_for_learning(a)
# a.drop(['hi', 'li','ci','oo','co','ho','lo'], axis=1,inplace=True)
# a.drop(['red_o', 'green_o'], axis=1,inplace=True)
b,c=ot.in_out_split(a)
out_train,out_test,input_train,input_test=ot.train_test_split(b,c)
train_data=torch.from_numpy(input_train.to_numpy())
train_data=train_data.to(torch.float32)
train_labels=torch.from_numpy(out_train.to_numpy())
train_labels=train_labels.to(torch.float32)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Define the architecture of the neural network
        self.hidden1 = nn.Linear(27, 300)
        self.hidden2 = nn.Linear(300, 100)
        self.hidden3 = nn.Linear(100, 20)
        self.hidden4=nn.Linear(20,4)
        # self.output = nn.Linear(2, 2)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x=self.hidden4(x)
        return x


# Create an instance of the neural network
model = NeuralNetwork()


# Define the training loop
def train(model, train_data, train_labels, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in zip(train_data, train_labels):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print average loss per epoch
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {running_loss / len(train_data)}")

    print("Training finished!")
    return running_loss / len(train_data)


# Convert your training data and labels to PyTorch tensors
train_data = torch.tensor(train_data)
train_labels = torch.tensor(train_labels)

# Define hyperparameters for training
num_epochs =10
learning_rate = 0.0001

# Create an instance of the neural network
model = NeuralNetwork()

# Train the model
# train(model, train_data, train_labels, num_epochs, learning_rate)
# model=retrieve_model('ohlc')
#
# model_scripted = torch.jit.script(model) # Export to TorchScript
# model_scripted.save('ohlc_model.pt')

# ---------


def update_train(name):
    a = pd.read_pickle('.\\stock\\'+name)
    a = ot.ready_for_learning(a)
    # a.drop(['hi', 'li','ci','oo','co','ho','lo'], axis=1,inplace=True)
    b, c = ot.in_out_split(a)
    out_train, out_test, input_train, input_test = ot.train_test_split(b, c)
    train_data = torch.from_numpy(input_train.to_numpy())
    train_data = train_data.to(torch.float32)
    train_labels = torch.from_numpy(out_train.to_numpy())
    train_labels = train_labels.to(torch.float32)
    return (train_data,train_labels)
import os
# -----------------------------------------------------


# 'RELIANCE','TCS','HINDUNILVR','KOTAKBANK','ICICIBANK','SBIN','WIPRO',
dir_list = os.listdir('.\\stock')
dir_list=[ 'RELIANCE','TCS','HINDUNILVR','KOTAKBANK','ICICIBANK','SBIN','WIPRO','AXISBANK','TATASTEEL','TATAMOTORS','ADANIENT','FACT']
x=5
z=.0006
for i in dir_list:
    print(i)
    k = 0
    train_data, train_labels = update_train(i)
    x = train(model, train_data, train_labels, num_epochs, learning_rate)
    while x>z:
        # test(i)
        train_data,train_labels=update_train(i)
        x=train(model, train_data, train_labels, num_epochs, learning_rate)
    model_scripted = torch.jit.script(model)  # Export to TorchScript
    model_scripted.save('.\\model_regression\\'+i)


test_data=torch.from_numpy(input_test.to_numpy())
test_data=train_data.to(torch.float32)
test_labels=torch.from_numpy(out_test.to_numpy())
test_labels=train_labels.to(torch.float32)

# Define the test function
def test(model, test_data, test_labels):
    with torch.no_grad():
        outputs = model(test_data)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == test_labels).sum().item()
        total = test_labels.size(0)
        accuracy = correct / total
        print(f"Accuracy: {accuracy * 100:.2f}%")

# Convert your test data and labels to PyTorch tensors
test_data = torch.tensor(test_data)
test_labels = torch.tensor(test_labels)
 # Set the model to evaluation mode
model.eval()

# Make predictions on the test data
with torch.no_grad():
    predictions = model(test_data)

# Calculate the root mean squared error (RMSE)
mse = nn.MSELoss()
rmse = math.sqrt(mse(predictions, test_labels))

print("Root Mean Squared Error (RMSE):", rmse)
# from sklearn.metrics import r2_score
# score = r2_score( test_labels,predictions)
# print("The accuracy of our model is {}%".format(round(score, 2) *100))
def accuracy_test(pred,label):
    diff=pred-label
    diff=abs(diff/label)*100
    first=diff[:,:1].clone().view(1,-1)[0]
    second = diff[:, 1:2].clone().view(1, -1)[0]
    third = diff[:, 2:3].clone().view(1, -1)[0]
    fourth = diff[:, 3:4].clone().view(1, -1)[0]
    print('accuracy overall:',100-diff.sum()/diff.numel(),' other accuracy: ',100-first.sum()/first.numel(),
          100-second.sum()/second.numel(),100-third.sum()/third.numel(),100-fourth.sum()/fourth.numel())
accuracy_test(predictions,test_labels)
# # Load the trained model
# model = NeuralNetwork()
#
# # Load the saved weights of the trained model
# model.load_state_dict(torch.load('model_weights.pth'))
# sv(NeuralNetwork(),'ritik')
# Test the model
# test(model, test_data, test_labels)