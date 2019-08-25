import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
import pickle

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data =  scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y =   test_data['test_data'], test_data['test_labels']
print(valid_x.shape)

max_iters = 300
# pick a batch size, learning rate
batch_size = 32
learning_rate = 1e-3
hidden_size = 64
train_acc_list = []
train_loss_list = []
valid_acc_list = []
valid_loss_list = []


batches = get_random_batches(train_x,train_y,batch_size)
print('train_x shape',train_x.shape)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(1024,64,params,'layer1')
initialize_weights(64,36,params,'output')

 
# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        h1 = forward(xb ,params,'layer1',activation=sigmoid)
        # print('h shape',h.shape)
        probs = forward(h1 , params, 'output', activation=softmax)


        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss , acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc

        # print(acc)

        # backward
        delta = probs - yb
        delta1 = backwards(delta,params,name='output',activation_deriv=linear_deriv)
        delta2 = backwards(delta1,params,name='layer1',activation_deriv=sigmoid_deriv)

        # apply gradient
        # print('blah', params['b' + 'output'].shape)
        # print('blahblah', params['grad_b' + 'output'].shape)
        params['W' + 'output'] -= learning_rate * params['grad_W' + 'output'] 
        params['b' + 'output'] -= learning_rate * params['grad_b' + 'output'] 
        params['W' + 'layer1'] -= learning_rate * params['grad_W' + 'layer1'] 
        params['b' + 'layer1'] -= learning_rate * params['grad_b' + 'layer1'] 


    total_acc = total_acc / len(batches)
    total_loss = total_loss / len(batches)

    train_acc_list.append(total_acc)
    train_loss_list.append(total_loss)


        # training loop can be exactly the same as q2!
        
    if itr % 40 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
# run on validation set and report accuracy! should be above 75%
    valid_acc = None
    h_valid = forward(valid_x ,params,'layer1',activation=sigmoid)
    valid_probs = forward(h_valid , params, 'output', activation=softmax)
    valid_loss , valid_acc = compute_loss_and_acc(valid_y, valid_probs)
    valid_acc_list.append(valid_acc)
    valid_loss_list.append(valid_loss)
    print('Validation accuracy: ',valid_acc)







# if False: # view the data
#     for crop in xb:
#         import matplotlib.pyplot as plt
#         plt.imshow(crop.reshape(32,32).T)
#         plt.show()
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


#Q 3.1.1 Plotting 

# Accuracy Plot
acc_fig = plt.figure(1)
ax = acc_fig.add_subplot(111)
ax.plot(np.arange(max_iters), train_acc_list, color='red', linewidth=2, label = 'Training Data')
ax.plot(np.arange(max_iters), valid_acc_list, color='lightblue', linewidth=2, label = 'Validation Data')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# # plt.savefig('../image_submission/accuracy_regular_lr.jpeg')


# #Loss Plot
loss_fig = plt.figure(2)
ax = loss_fig.add_subplot(111)
ax.plot(np.arange(max_iters), train_loss_list, color='red', linewidth=2, label = 'Training Data')
# ax.plot(np.arange(max_iters), valid_loss_list, color='lightblue', linewidth=2, label = 'Validation Data')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../image_submission/loss_10times_lr.jpeg')



#load saved params
with open('q3_weights.pickle', 'rb') as handle:
    saved_params = pickle.load(handle)



# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


# computed layer 1 
W_computed_layer1 = saved_params["Wlayer1"]
initialize_weights(1024,64,params,'Initial_layer1')
W_init_layer1 = params['WInitial_layer1']

# print(W_computed_layer1.shape)
# print(type(W_computed_layer1))

#image grid of the trained weight
trained_weight_fig = plt.figure(3)
grid = ImageGrid(trained_weight_fig, 111, (8,8))
for i in range(64):
    W_i = W_computed_layer1[:, i].reshape(32, 32)
    grid[i].imshow(W_i)
# plt.show()
# plt.savefig('../image_submission/trained_weight_fig.jpeg')



#image grid of the initialized weight
init_weight_fig = plt.figure(4)
grid = ImageGrid(init_weight_fig, 111, (8,8))
for i in range(64):
    W_i = W_init_layer1[:, i].reshape(32, 32)
    grid[i].imshow(W_i)
# plt.show()
# plt.savefig('../image_submission/init_weight_fig.jpeg')





confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

h1 = forward(valid_x, saved_params, 'layer1')
probs = forward(h1, saved_params, 'output', softmax)

grd_truth = np.argmax(valid_y, axis=1)
predicted = np.argmax(probs, axis=1)

num_samples = grd_truth.shape[0]

for i in range(num_samples):
    confusion_matrix[grd_truth[i], predicted[i]] += 1
# print(confusion_matrix)


# Visualization of the confusion matrix
import string
plt.figure(5)
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
# plt.show()
# plt.savefig('../image_submission/confusion_matrix.jpeg')



