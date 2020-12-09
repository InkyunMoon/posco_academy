import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.init as init 
import torchvision.datasets as dataset 
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader
import numpy as np 
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device: gpu") if torch.cuda.is_available() else print("device: cpu")

# Hyper parameter setting
learning_rate = 1e-1 
epochs = 25

# batch_size = 60000 # gradient descent
batch_size = 1 # stochastic gradient descent
batch_size = 32 # mini-batch stochastic gradient descent act = nn.ReLU() h = 200 display_step = 5
act = nn.ReLU()
h = 200
display_step = 5


# Load data and pre-process data
# load data
train_data = dataset.MNIST("./", train = True, transform = transforms.ToTensor(), target_transform = None, download = True) 
test_data = dataset.MNIST("./", train = False, transform = transforms.ToTensor(), target_transform = None, download = True)

# check the data
print('len(train_data): ', len(train_data)) 
print('len(test_data): ', len(test_data))

x_train, y_train = train_data[0] 
print('original data shape: ', x_train.shape) 
print('label: ', y_train)

plt.figure() 
plt.imshow(x_train[0]) 
plt.show()

# Pre-process (batch, shuffle)
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 1, drop_last = True) 
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 1000, shuffle = True, num_workers = 1, drop_last = True)

# check the data
examples = enumerate(train_loader) 
batch_idx, (example_data, example_target) = next(examples)

print('processed data shape:', example_data.shape) 
print('label:', example_target)

plt.figure() 
plt.imshow(example_data[0][0]) 
plt.show()

# Multi Layer Logistic Regression
# Train and result (with mini-batch stochastic gradient descent)
# model
model = nn.Sequential( 
    nn.Linear(np.prod(x_train.shape[1:]),h), 
    act, 
    nn.Linear(h,10), 
    )

model = model.to(device)
model.train()

# loss and optimizer
loss_function = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

loss_array = [] 
iteration_loss_array = []

# train the model
for epoch in range(epochs): 
    for iteration, [data, label] in enumerate(train_loader): 
        optimizer.zero_grad()
    
        x = data.to(device)
        x = x.view(batch_size, -1)
        y = label.to(device)
    
        output = model(x)
    
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()
        iteration_loss_array.append(loss.cpu().detach().numpy())
    
    loss_array.append(loss.cpu().detach().numpy())

    if epoch % 5 == 0:
        print("Epoch:", epoch + 1, "\Loss:", loss)
# plot losses
plt.figure() 
plt.plot(loss_array) 
plt.show()

# plot iteration losses
plt.figure() 
plt.plot(iteration_loss_array) 
plt.show()

# test
model.eval() 
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 10000, shuffle = True, num_workers = 1, drop_last = True) 
correct = 0 
total = 0

prediction_list = [] 
label_list = []

with torch.no_grad(): 
    for data, label in test_loader:
        x = data.to(device) 
        x = x.view(-1, 784) 
        y = label.to(device)

        prediction = model(x)
        _, prediction_index = torch.max(prediction, 1)
    
        prediction_list.append(prediction_index)
        label_list.append(y)
    
        total += y.size(0)
        correct += (prediction_index == y).sum().float()
print('total', total) 
print('correct', correct) 
print('accuracy', correct/total)

# confusion matrix
from sklearn.metrics import confusion_matrix 
import numpy as np

prediction_array = np.array(prediction_list[0].cpu()) 
label_array = np.array(label_list[0].cpu())

print("prediction :", prediction_array.shape) 
print("true label :", label_array.shape)

confusion_matrix( 
    label_array, prediction_array) # y_pred

# Advanced: Weight initialization
def init_weights(m): 
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight) 
        m.bias.data.fill_(0.01)

h2 = 100

# model
model = nn.Sequential( 
    nn.Linear(np.prod(x_train.shape[1:]),h),
    act,
    nn.Linear(np.prod(h),h2),
    act,
    nn.Linear(h2,10),
    )

model.apply(init_weights) 
model = model.to(device)
model.train()

# loss and optimizer
loss_function = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

loss_array = [] 
iteration_loss_array = []

# train the model
for epoch in range(epochs):
    for iteration, [data, label] in enumerate(train_loader): 
        optimizer.zero_grad()

        x = data.to(device)
        x = x.view(batch_size, -1)
        y = label.to(device)
    
        output = model(x)
    
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()
        iteration_loss_array.append(loss.cpu().detach().numpy())

    loss_array.append(loss.cpu().detach().numpy())

    if epoch % 5 == 0:
        print("Epoch:", epoch + 1, "\Loss:", loss)
        
# plot losses
plt.figure()
plt.plot(loss_array) 
plt.show()

# plot iteration losses
plt.figure() 
plt.plot(iteration_loss_array) 
plt.show()

# test
model.eval() 
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 10000, shuffle = True, num_workers = 1, drop_last = True) 
correct = 0 
total = 0

prediction_list = [] 
label_list = []

with torch.no_grad(): 
    for data, label in test_loader: 
        x = data.to(device) 
        x = x.view(-1, 784) 
        y = label.to(device)

        prediction = model(x)
        _, prediction_index = torch.max(prediction, 1)
    
        prediction_list.append(prediction_index)
        label_list.append(y)
    
        total += y.size(0)
        correct += (prediction_index == y).sum().float()
        
print('total', total) 
print('correct', correct) 
print('accuracy', correct/total)

# confusion matrix
from sklearn.metrics import confusion_matrix
import numpy as np

prediction_array = np.array(prediction_list[0].cpu()) 
label_array = np.array(label_list[0].cpu())

print("prediction :", prediction_array.shape) 
print("true label :", label_array.shape)

confusion_matrix( label_array, prediction_array)