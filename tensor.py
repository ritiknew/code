# import tensorflow as tf
import pandas as pd
import torch
import other as ot
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import gc
import numpy
STD_PATH='C:\\Users\\lenovo\\PycharmProjects\intraday\\'
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



# .\\stock\\BRITANNIA
name_list=['HDFCBANK','ICICIBANK','SBIN','RELIANCE','TITAN','INFY','TATAMOTORS','TCS',
           'NTPC','ITC','TATASTEEL','CIPLA','ADANIGREEN','TATAPOWER', 'POWERGRID','GAIL']
for stock_name in name_list:
    a=pd.read_pickle(STD_PATH+'\\stock\\'+stock_name)
    # a=ot.ready_for_learning(a)
    # a.drop(['hi', 'li','ci','oo','co','ho','lo'], axis=1,inplace=True)
    # b,c=ot.in_out_split(a)
    import check
    a=check.ready_to_machine_input(a,stock_name)
    output=['sell','buy','nothing']
    out_data=a[output]
    input=a.drop(output, axis=1)
    out_train,out_test,input_train,input_test=ot.train_test_split(out_data,input,90)
    train_data=torch.from_numpy(input_train.to_numpy())
    train_data=train_data.to(torch.float32)
    train_labels=torch.from_numpy(out_train.to_numpy())
    train_labels=train_labels.to(torch.float32)


    # Define a simple neural network model with Leaky ReLU activation
    class SimpleNN(nn.Module):
        def __init__(self, input_size, output_size):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, 20)  # 16 input nodes to a hidden layer with 64 nodes
            self.fc2 = nn.Linear(20, 20)
            # self.fc3=nn.Linear(128,256)
            # self.fc4=nn.Linear(256,256)
            # self.fc5 = nn.Linear(256, 128)
            self.fc6 = nn.Linear(20, 10)
            self.fc7=nn.Linear(10,output_size)
            # 64 hidden nodes to 3 output nodes
            self.leaky_relu = nn.LeakyReLU(0.1)  # Leaky ReLU activation with a small negative slope
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.leaky_relu(self.fc1(x))
            x=self.leaky_relu(self.fc2(x))
            # x = self.leaky_relu(self.fc3(x))
            # x = self.leaky_relu(self.fc4(x))
            # x = self.leaky_relu(self.fc5(x))
            x = self.leaky_relu(self.fc6(x))
            x = self.sigmoid(self.fc7(x))
            return x


    # Initialize the model
    input_size = 15
    output_size = 3
    model = SimpleNN(input_size, output_size)

    # Define the loss function (CrossEntropy for classification) and optimizer (Adam)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dummy data for training (you should replace this with your dataset)
    # num_samples = 1000
    X_train = train_data
    y_train = train_labels

    # Training loop
    loss_=100
    while loss_>0.01:
        num_epochs = 100
        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(X_train)

            # Calculate the loss
            loss = criterion(outputs, y_train)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the loss at every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
                loss_=loss.item()
    model_scripted = torch.jit.script(model)  # Export to TorchScript
    model_scripted.save(STD_PATH+'\\model_sell_10\\'+stock_name)
    gc.collect()

# Convert your training data and labels to PyTorch tensors
train_data = torch.tensor(train_data)
train_labels = torch.tensor(train_labels)

# Define hyperparameters for training
num_epochs =10000
learning_rate = 0.0001

# Create an instance of the neural network
# model = NeuralNetwork()
# model=retrieve_model('r_g')
# model_scripted = torch.jit.script(model) # Export to TorchScript
# model_scripted.save('tensor_model.pt')
# Train the model
# train(model, train_data, train_labels, num_epochs, learning_rate)


def update_train(name):
    a = pd.read_pickle('.\\stock\\'+name)
    a = ot.ready_for_learning(a)
    # a.drop(['hi', 'li','ci','oo','co','ho','lo'], axis=1,inplace=True)
    b, c = ot.in_out_split(a)
    out_train, out_test, input_train, input_test = ot.train_test_split(b, c)
    train_data = torch.from_numpy(input_train.to_numpy())
    train_data = train_data.to(torch.float32)
    train_labels = torch.from_numpy(out_train[['red_o', 'green_o']].to_numpy())
    train_labels = train_labels.to(torch.float32)
    return (train_data,train_labels)
import os
# -----------------------------------------------------


# 'RELIANCE','TCS','HINDUNILVR','KOTAKBANK','ICICIBANK','SBIN','WIPRO',
dir_list = os.listdir('.\\stock')
dir_list=[ 'WIPRO','AXISBANK','TATASTEEL','TATAMOTORS','ADANIENT','FACT']
x=5
z=.05
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
    model_scripted.save('.\\model_color\\'+i)

# ------------------------------------------------------------------

test_data=torch.from_numpy(input_test.to_numpy())
test_data=train_data.to(torch.float32)
test_labels=torch.from_numpy(out_test[['red_o','green_o']].to_numpy())
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

# Round the predictions to obtain the binary class labels
rounded_predictions = torch.round(predictions)

# # Convert the tensor to a numpy array
y_pred = rounded_predictions.numpy().flatten()
y_true = test_labels.numpy().flatten()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
# # Load the trained model
# model = NeuralNetwork()
#
# # Load the saved weights of the trained model
# model.load_state_dict(torch.load('model_weights.pth'))

# Test the model
# test(model, test_data, test_labels)