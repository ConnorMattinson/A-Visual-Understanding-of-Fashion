# Writing a functional NN using only the numpy library
import numpy as np

# AIM: map these inputs to the corresponding outputs
# e.g. trying to get NN([1,1]) = [1]

inputs = np.array([[1,1],
                   [1,0],
                   [0,1],
                   [0,0]]).T

expected = np.array([[1],[0],[0],[1]]).T

# input size is 2, goes to a hidden layer size 2, then outputs 1 number as in figure 9
layer_sizes = [2,2,1]


#========FORWARD PROPAGATION========
# Get a bunch of weights for your network, all matrices with dimensions to map between layers
# Notice we add 1 to the second matrix dimension to accomodate a bias term (a parameter with no variable coefficient) 
# Numpy random normal distribution used to initialise parameters with mean 0 and std 1, works well when you normalise the input too
def initialise_network(layer_sizes):
    params = [np.random.randn(next_layer, prev_layer+1 ) for prev_layer, next_layer in zip(layer_sizes[:-1], layer_sizes[1:])]
    return params

params = initialise_network(layer_sizes)

# The logistic function is an example of a sigmoid (s-shaped) function but is often called sigmoid itself
# np.exp applies the function elementwise to any dimension of tensor
def logistic(z):
    return 1/(1+ np.exp(-z))

# This gets you from an arbitrary layer to the next
# This is really general on purpose so we can call it lots of times in 'forward_propagate'
def transfer(weights, inputs, activation):
    inputs = np.append(inputs,np.ones((1,inputs.shape[1])), axis = 0)
    outputs = activation(weights@inputs)
    return outputs

# Runs transfer multiple times to map from the start of the network to the end
# Outputs are still called inputs, it's just because it loops better that way
def forward_propagate(params, inputs, transfer, to_layer = len(params)):
    for i in range(to_layer):
        inputs = transfer(params[i], inputs, logistic)
    return inputs

#test
#output = forward_propagate(params, inputs, transfer)


#========BACKWARD PROPAGATION & TRAINING========
# Look at train_netowrk first, then update_params, then get_error 

# dLogistic/dz, used for backprop
# Handy shortcut to the derivative
def logistic_derivative(z, logistic):
    return logistic(z)*(1-logistic(z))

# These errors are acquired directly using the backprop matrix equations
# This is coded straight from the maths
# Matrix backprop equations are on my github
# It returns a set of matrices idential in shape to the parameters
# These matrices tell you how much to change each corresponding parameter to make the model more accurate
# Printing all the time is essential here, I commented them out for training
def get_error(params, expected, inputs, input_num):
    j = 0
    derivs = []
    for i in reversed(range(len(params))):
        if j == 0 :
            output = forward_propagate(params, np.array([inputs[:,input_num]]).T, transfer)
            #print('output:  ', output)
            dC_dy = output - np.array([expected[:,input_num]]).T
            #print('dC_dy:  ', dC_dy)
            last_hidden = forward_propagate(params, np.array([inputs[:,input_num]]).T, transfer, to_layer = i)
            last_hidden = np.append(last_hidden,np.ones((1,last_hidden.shape[1])), axis = 0)
            #print('last_hidden:  ', last_hidden)
            sig_deriv = logistic_derivative(params[i]@last_hidden, logistic)
            #print('sig_deriv:  ', sig_deriv)
            next_delta = dC_dy*sig_deriv
            #print('next_delta:  ', next_delta)
            dC_dparams = next_delta@last_hidden.T
            #print('dC_dparams:  ', dC_dparams)
            derivs.append(dC_dparams)
            j = 1
        else:
            prev_hidden = forward_propagate(params, np.array([inputs[:,input_num]]).T, transfer, to_layer = i)
            #print('prev_hidden:  ', prev_hidden)
            prev_hidden = np.append(prev_hidden,np.ones((1,prev_hidden.shape[1])), axis = 0)
            #print('prev_hidden:  ', prev_hidden)
            sig_deriv = logistic_derivative(params[i]@prev_hidden, logistic)
            #print('sig_deriv:  ', sig_deriv)
            delta_i = (params[i+1][:,:-1].T@next_delta)*sig_deriv
            #print('delta_i:  ', delta_i)
            dC_dparams = delta_i@prev_hidden.T
            #print('dC_dparams:  ', dC_dparams)
            next_delta = delta_i
            derivs.append(dC_dparams)
    derivs.reverse()
    return derivs


l_rate = 0.86 # how much you change the parameters each update

# Uses the change found from get_error to update the parameters
def update_params(params, deltas, l_rate):
    for i in range(len(params)):
        params[i] -= l_rate*deltas[i]
    return params
 
paramlist = []

# The training loop! 
# An epoch is an entire pass of the training set
# Cost function is the sum of how wrong the model currently is
# Notice the parameters are updated every epoch, this is 'batch gradient descent'
# Feel free to try and make this mini-batch gradient descent or stochastic GD!
def train_network(params, inputs, expected, transfer, n_epoch, l_rate):
    for epoch in range(n_epoch):
        outputs = forward_propagate(params, inputs, transfer)
        errors = 0.5*(outputs - expected)**2
        sum_errors = [sum(error) for error in errors]
        cost = sum(sum_errors)
        deltas = []
        for j in range(len(params)):
            delta = np.mean(np.array([get_error(params, expected, inputs, i)[j] for i in range(inputs.shape[1])]), axis = 0)
            deltas.append(delta)
        params = update_params(params, deltas, l_rate)
        paramlist.append([ params[0][0][1],params[0][1][0] ])
        print("epoch: {},   cost: {}".format(epoch, cost))
    print('expected:', expected)
    print('predicted:', outputs)
    return params

params = train_network(params, inputs, expected, transfer, 10000, l_rate)

trained_params = np.array([[[-6.32119493,  6.05276143, -3.22339192],
                           [-6.82582194,  6.84765277,  3.47960836]],
                          [[-11.1699101 ,  10.72136726,  -5.09849153]]]) #lr = 0.86, 100,000 epochs

#========FIGURE PLOTTING========

import matplotlib.pyplot as plt
#'''
# Inputs and outputs for the XOR function
x = np.array([[1,0,1,0],[1,1,0,0]])
y = np.array([1,0,0,1])

# Figure 7, Linear Approximation
fig = plt.figure(figsize = (30,30))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[0], x[1], y, alpha = 1)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')


point = [0,0,1]
normal = np.array([-0.5, -0.5, -1])
d = -np.dot(point,normal)
xx, yy = np.meshgrid(range(0,2), range(0,2))
z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
ax.plot_surface(xx, yy, z, color = 'orange',  alpha=0.5)

for i in range(4):
    xlin = [x[0,i],x[0,i] ]
    ylin = [x[1,i], x[1,i]]
    zlin = [y[i], 1-0.5*x[0,i]-0.5*x[1,i]]
    ax.plot(xlin,ylin,zlin,'r-',alpha=0.8, linewidth=1)

fig = plt.figure(figsize = (30,30))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[0], x[1], y, alpha = 1) # Add dataset points
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')

xx, yy = np.meshgrid([i/100 for i in range(100)], [i/100 for i in range(100)]) # Add surface
z = np.array([[ forward_propagate(trained_params, np.array([[ xx[i,j] ,  yy[i,j] ]]).T, transfer)[0] for i in range(100)] for j in range(100)])
z = z[:,:,0]
ax.plot_surface(xx, yy, z, color = 'orange',  alpha=0.5)


# + for rotate
ax.axis('off')
for angle in range(0, 360):
    ax.view_init(15, angle)
    plt.draw()
    plt.pause(.1)
#'''   
    
# Figure 13 
# comment out other figures using ''' then run whole script
# to add more paths, comment out the surface plot and run whole script again
res = 100 #resolution of the surface plot
propx0, propx1 = np.meshgrid(np.linspace(-13,8,res), np.linspace(-15,13,res))
outz = np.zeros([res,res]) # surface heights to populate in for loop
for i in range(res):
    for j in range(res):
        outz[i,j] = (1/4)*np.sum(
                  [
                      (forward_propagate(np.array(
                              [
                              [   [-6.32119493,  propx1[i,j], -3.22339192],   #only two weights are varied, the others are optimal
                                  [propx0[i,j],  6.84765277,  3.47960836]    ],
                              [   [-11.1699101 ,  10.72136726,  -5.09849153]]
                              ]),inputs, transfer)[0]- expected[0])**2
                  ])

#'''
fig = plt.figure(figsize = (30,30))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('$w_{2,1}$')
ax.set_ylabel('$w_{1,2}$')
ax.set_zlabel('C')
ax.plot_surface(propx0, propx1, outz, cmap = 'binary', alpha = 0.5)
#'''

subparams = np.array([[paramlist[100*i][0] for i in range(100)],[paramlist[100*i][1] for i in range(100)]])
subparams = np.array([subparams[0], subparams[1], np.zeros([100])])
for i in range(100):
        subparams[2][i] = (1/4)*np.sum(
                      (forward_propagate(np.array(
                              [
                              [   [-6.32119493,  subparams[1][i], -3.22339192],
                                  [subparams[0][i],  6.84765277,  3.47960836]    ],
                              [   [-11.1699101 ,  10.72136726,  -5.09849153]]
                              ]),inputs, transfer)[0]- expected[0])**2)
ax.scatter3D(subparams[0],subparams[1],subparams[2])







