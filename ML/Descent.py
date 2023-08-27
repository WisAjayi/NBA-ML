import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm 
from math import log
from sympy import symbols, diff
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

DESCENT = None

### RUN Cost or Derivative function and then gradient descent function on Data ###


TEAM_NAME = 'Atlanta Hawks'
TEAM_ABBREV = 'ATL'

PLAYER_FIRST_NAME = "Trae"
PLAYER_LAST_NAME = "Young"

def init():

    Player = PLAYER_FIRST_NAME + " " + PLAYER_LAST_NAME
    return Player

PLAYER = init()

# Data #
data = pd.read_csv(f'../TEAMS/{TEAM_NAME}/GAMELOG/{PLAYER}.csv')


### Cost Function ###
def f(x):
    return x**2 + x + 1


### Derivatives ###
def df(x):
    return 2*x + 1


def g(x):
    return x**4 - 4*x**2 + 5


def dg(x):
    return 4*x**3 - 8*x


def h(x):
    return x**5 - 2*x**4 + 2

def dh(x):
    return 5*x**4 - 8*x**3


def mini(x, y):  # Minimise Function #
    r = 3**(-x**2 - y**2)
    return 1 / (r + 1)


def gradient_descent(derivative_func, initial_guess, multiplier=0.02, precision=0.001, max_iter=300):
    
    new_x = initial_guess
    x_list = [new_x]
    slope_list = [derivative_func(new_x)]

    for n in range(max_iter):
        previous_x = new_x
        gradient = derivative_func(previous_x)
        new_x = previous_x - multiplier * gradient

        step_size = abs(new_x - previous_x)
        x_list.append(new_x)
        slope_list.append(derivative_func(new_x))

        if step_size < precision:
            break
    
    #print('Local minimum occurs at:', new_x)
    return new_x, x_list, slope_list


def local_min():
    
    local_min, list_x, deriv_list = gradient_descent(dg, 0.5, 0.02, 0.001)
    print('Local min occurs at:', local_min)
    print('Number of steps:', len(list_x))


    local_min, list_x, deriv_list = gradient_descent(derivative_func=dg, initial_guess= -0.5, multiplier=0.01, precision=0.0001)
    print('Local min occurs at:', local_min)
    print('Number of steps:', len(list_x))


    local_min, list_x, deriv_list = gradient_descent(derivative_func=dg, initial_guess= -0.1)
    print('Local min occurs at:', local_min)
    print('Number of steps:', len(list_x))


#print( sys.float_info.max )







### THE LEARNING RATE ###
n = 100
low_gamma = gradient_descent(derivative_func=dg, initial_guess= 3,multiplier=0.0005, precision=0.0001, max_iter=n)
# Plot two more learning rates: mid_gamma (0.001) and high_gamma(0.002)
mid_gamma = gradient_descent(derivative_func=dg, initial_guess= 3, multiplier=0.001, precision=0.0001, max_iter=n)
high_gamma = gradient_descent(derivative_func=dg, initial_guess= 3,multiplier=0.002, precision=0.0001, max_iter=n)
# Experiment
insane_gamma = gradient_descent(derivative_func=dg, initial_guess= 1.9,multiplier=0.25, precision=0.0001, max_iter=n)

def visualise_learning_rate():
    
    # Plotting reduction in cost for each iteration
    plt.figure(figsize=[20, 10])

    plt.xlim(0, n)
    plt.ylim(0, 50)

    plt.title('Effect of the learning rate', fontsize=17)
    plt.xlabel('Nr of iterations', fontsize=16)
    plt.ylabel('Cost', fontsize=16)

    # Values for our charts
    # 1) Y Axis Data: convert the lists to numpy arrays
    low_values = np.array(low_gamma[1])

    # 2) X Axis Data: create a list from 0 to n+1
    iteration_list = list(range(0, n+1))

    # Plotting low learning rate
    plt.plot(iteration_list, g(low_values), color='lightgreen', linewidth=5)
    plt.scatter(iteration_list, g(low_values), color='lightgreen', s=80)

    # Plotting mid learning rate
    plt.plot(iteration_list, g(np.array(mid_gamma[1])), color='steelblue', linewidth=5)
    plt.scatter(iteration_list, g(np.array(mid_gamma[1])), color='steelblue', s=80)

    # Plotting high learning rate
    plt.plot(iteration_list, g(np.array(high_gamma[1])), color='hotpink', linewidth=5)
    plt.scatter(iteration_list, g(np.array(high_gamma[1])), color='hotpink', s=80)

    # Plotting insane learning rate
    plt.plot(iteration_list, g(np.array(insane_gamma[1])), color='red', linewidth=5)
    plt.scatter(iteration_list, g(np.array(insane_gamma[1])), color='red', s=80)


    plt.show()
    
    
    
    
    



### Partial Derivatives & Symbolic Computation ###

a, b = symbols('x, y')
print('Our cost function f(x, y) is: ', f(a, b))
print('Partial derivative wrt x is: ', diff(f(a, b), b))
print('Value of f(x,y) at x=1.8 y=1.0 is: ', f(a, b).evalf(subs={a:1.8, b:1.0})) # Python Dictionary
print('Value of partial derivative wrt x: ', diff(f(a, b), a).evalf(subs={a:1.8, b:1.0}))


### BATCH DESCENT WITH SYMPY ###

# Setup
multiplier = 0.1
max_iter = 500
params = np.array([1.8, 1.0]) # initial guess

for n in range(max_iter):
    gradient_x = diff(f(a, b), a).evalf(subs={a:params[0], b:params[1]})
    gradient_y = diff(f(a, b), b).evalf(subs={a:params[0], b:params[1]})
    gradients = np.array([gradient_x, gradient_y])
    params = params - multiplier * gradients
    
# Results
print('Values in gradient array', gradients)
print('Minimum occurs at x value of: ', params[0])
print('Minimum occurs at y value of: ', params[1])
print('The cost is: ', f(params[0], params[1]))


def fpx(x, y):
    r = 3**(-x**2 - y**2)
    return 2*x*log(3)*r / (r + 1)**2

def fpy(x, y):
    r = 3**(-x**2 - y**2)
    return 2*y*log(3)*r / (r + 1)**2



def mse(y, y_hat):

    mse_calc = np.average((y - y_hat)**2, axis=0)
    return mse_calc



# Setup
multiplier = 0.1
max_iter = 500
params = np.array([1.8, 1.0]) # initial guess

for n in range(max_iter):
    gradient_x = fpx(params[0], params[1])
    gradient_y = fpy(params[0], params[1])
    gradients = np.array([gradient_x, gradient_y])
    params = params - multiplier * gradients
    
# Results
print('Values in gradient array', gradients)
print('Minimum occurs at x value of: ', params[0])
print('Minimum occurs at y value of: ', params[1])
print('The cost is: ', f(params[0], params[1]))