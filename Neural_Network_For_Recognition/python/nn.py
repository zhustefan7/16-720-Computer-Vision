import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 2.1
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    
    lower = -1.0 * np.sqrt(6.0)  / np.sqrt(in_size + out_size)
    upper = np.sqrt(6.0) / np.sqrt(in_size + out_size)

    W = np.random.uniform(lower, upper, (in_size, out_size))
    b = np.zeros(out_size)


    params['W' + name] = W
    params['b' + name] = b



# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1 / (1+np.exp(-x))
    return res

# Q 2.2.2
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    # print("W", W.shape)
    # print('b', b.shape)
    # print('X forward', X.shape)
    # your code here
    pre_act = np.matmul(X,W)+b
    # print('pre_act',pre_act.shape)
    post_act= activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 2.2.2 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    # print('x',x)
    # print(x.shape)
    # print('x shape', x.shape)
    si = np.exp(x)
    # print('si', si.shape)
    # print( si)
    S = np.sum(si, axis=1 )[:,np.newaxis]
    # print('S shape',S.shape)
    # print(S)
    res = np.divide(si,S)
    # print('si',si)
    # print('S',S)
    # print('res',res)
    return res






# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None
    num_sample = y.shape[0]
    loss = -np.sum(np.multiply(y,np.log(probs))) 
    # print('losss', loss)
    # print('loss shape' , loss.shape)
    # print('probs',probs.shape)
    acc = np.sum(np.equal(np.argmax(probs,axis= 1),np.argmax(y,axis = 1)))/num_sample


    # print(np.argmax(y,axis = -1))
    # print(np.sum(np.equal(np.argmax(probs,axis=-1),np.argmax(y,axis = -1))))
    # print('y' , y.shape)
    # print('acc', acc)


    return loss, acc 




# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res





def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """ 
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name] 
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X
    # print("delta shape" , delta.shape)
    # print("W shape", W.shape)

    # print ("X shape", X.shape)
    delta = delta * activation_deriv(post_act)

    grad_X = np.matmul(delta, np.transpose(W))

    grad_W = np.matmul(np.transpose(X), delta)

    grad_b = np.dot(np.transpose(delta),np.ones((delta.shape[0],)))


    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b

    return grad_X




# # Q 2.4
# # split x and y into random batches1
# # return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    # print("batches x shape", x.shape)
    # print("batches y shape",y.shape)
    # print('batch size', batch_size)
    # print('x shape',x.shape)
    samples_num= x.shape[0]
    batch_num = round(samples_num / batch_size)  
    batches = []
    # print('x shape',x.shape)
    for i in range(batch_num):
        indx = np.random.permutation(np.arange(samples_num))[:batch_size]
        # indx = np.random.randint(0 , samples_num, batch_size)
        samle_x = x[indx,:]
        # print('sample x shape', samle_x.shape)
        sample_y = y[indx,:]
        batches.append((samle_x,sample_y))
    # print(batches)

    return batches


