from math import *

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

def h_star(t):
    h_star = exp(-5 * pow(t - 3, 2)) + 0.5 * exp(-100 * pow(t - 5, 2)) + 5 * exp(-100 * pow(t - 0.75, 2))
    return h_star


def lin_regression(d, Xi, Yi,debug = False):
    # d = degree of polynomial
    n = np.size(Xi)
    Yi= np.reshape(Yi,(15,1))

    # CONSTRUCT VANDERMONDE MATRIX
    V = np.array([[]])

    for c in range(d+1):
        Vcol = []
        for xi in range(n):
            Vcol.append(pow(Xi[xi],c))

        append_col = np.reshape(Vcol, (n, 1))
        if np.size(V)<1:
            V=append_col
        else:
            V = np.append(V, append_col,axis=1)


    V_T = np.transpose(V)
    #if debug: print(V_T.shape,V.shape)

    # CALCULATE w_erm USING METHOD FROM HW 2 PART 2.d
    # w_erm = (V_T*V)^{-1} * V_T*Y
    VTV_inv = inv(np.matmul(V_T, V))
    VY = np.matmul(V_T,Yi)
    w_erm =  np.matmul(VTV_inv,VY)
    if debug: print("w_erm = ", w_erm)
    return w_erm

def h(w, t):
    h = sum([w[i]*pow(t,i) for i in range(len(w))])
    return h


# PART 3.A #####################################################
n = 15
mu = 0.0
sigma = 0.1
domain = [0.,1.]
t = np.linspace(0, 1, 500)      # Higher resolution time interval for true function and regression

# CALCULATE UNIFORMLY RANDOM POINTS
Xi = np.sort(np.random.uniform(domain[0], domain[1], n))

# NOISE
Wi = np.random.normal(mu, sigma, n)

# CALCULATE TRUE FUNCTION OBSERVED DATA POINTS
#Y = [h_star(Xi[i]) for i in range(n)]           # True function
Y = [h_star(ts) for ts in t]           # True function
Yi = [h_star(Xi[i]) + Wi[i] for i in range(n)]  # Observed output

# PLOT
fig1, ax1 = plt.subplots()
ax1.plot(t, Y, color="r",linestyle=":")          # Plot true function
ax1.scatter(Xi, Yi, color='r')      # Plot noisy data
ax1.set_ylabel('Y')
ax1.set_xlabel('t')
labels = ["True function", "Noisy observations"]
plt.legend(labels)
plt.title("HW 2 Part 3.a Plot: h^*(t)")

# PART 3.B #####################################################
# CALCULATE VALUES FOR h(t)
w = [1, -2, 3]                  # Given constant coeff for polynomial
h_t = [h(w, ti) for ti in t]    # Calculate h for given coeff

# APPEND SUBPLOT
n_ax = len(fig1.axes)
for i in range(n_ax):
    fig1.axes[i].change_geometry(n_ax + 1, 1, i + 1)
ax1 = fig1.add_subplot(n_ax + 1, 1, n_ax + 1)

# PLOT NEW DATA
ax1.plot(t, Y, color="r",linestyle=":") # Plot true function
ax1.plot(t, h_t, color="g")
ax1.scatter(Xi, Yi, color='r')      # Plot noisy data
ax1.set_ylabel('Y')
ax1.set_xlabel('t')
labels = ["True function", "h(t)=1-2t+3t^2","Noisy observations"]
plt.legend(labels)
plt.title("HW 2 Question 3.b Plot: h(t)")

# PART 3.C #####################################################
# Set up Figure 2
fig2, ax2 = plt.subplots()
plt.title("HW 2 Question 3.b: Regression")

# Define poly orders to be fitted
fit_degrees = [2, 5, 15, 20]

# Calculate and plot regressions
for d in fit_degrees:
    print(f'plotting regression poly degree={d}')
    w_erm = lin_regression(d,Xi,Yi)
    #print(np.shape(w_erm))
    h_t = [h(w_erm, ti) for ti in t]

    # Plot True function and Noisy observations
    ax2.plot(t, Y, color="r",linestyle=":")  # Plot true function
    ax2.scatter(Xi, Yi, color='r')  # Plot noisy data

    # Plot Regression Line
    ax2.plot(t, h_t)

    # Configure Plot
    labels = ["True function", "Noisy observations",f'Poly Degree = {d}']
    ax2.legend(labels)
    plt.ylim((-3, 8))


    # APPEND SUBPLOT IF NOT LAST PLOT
    if d != fit_degrees[-1]:
        n_ax = len(fig2.axes)
        for i in range(n_ax):
            fig2.axes[i].change_geometry(n_ax + 1, 1, i + 1)
        ax2 = fig2.add_subplot(n_ax + 1, 1, n_ax + 1)




print("FINISHED ####################")
plt.show()
