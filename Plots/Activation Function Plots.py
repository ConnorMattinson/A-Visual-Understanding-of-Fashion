import numpy as np
import matplotlib.pyplot as plt

# Figure 8 (Common Activation Functions)
x = np.linspace(-2.5,2.5,50)
xmore = np.linspace(-2.5,2.5,5000) # used for step for higher precision
xrelu = np.linspace(-2.45,2.55,50) # helps stagger the makers on the figure
logistic = 1/(1+np.exp(-x)). # In isolation, gives Figure 12
relu = [np.max([0,i]) for i in xrelu]
leaky_relu = [np.max([0.1*i,i]) for i in x]
elu = [0.5*(np.exp(i)-1) if i<=0 else i for i in x]
step = [0 if i<=0 else 1 for i in xmore]
tanh = (np.exp(x) - np.exp(-x))/(np.exp(x)+ np.exp(-x))

plt.plot(xmore, step,color='#AADD88', linestyle='-')
plt.plot(x, leaky_relu,color='pink', linestyle='-')
plt.plot(x, logistic, color='#666666', linestyle='-')
plt.plot(x, tanh, color='#4444DD', linestyle='-')
plt.plot(xrelu, relu, color = '#6666DD', linestyle='', marker='1')
plt.plot(x, elu, color='red', linestyle='', marker='1')

x = np.linspace(-8,8,500)

plt.plot(x, logistic, color='#000000', linestyle='-')

plt.legend(['Step','Leaky ReLU', 'Logistic',  'Tanh' ,  'ReLU', 'Elu'])
