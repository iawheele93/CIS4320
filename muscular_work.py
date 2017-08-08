from __future__ import print_function
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
import numpy as np
import sys


print(__doc__)

body_mass, work_level = [], []
heat_output = []

muscle1_df=pd.read_csv('muscle1.csv',encoding='utf8')
muscle1_df.apply(lambda x: pd.lib.infer_dtype(x.values))

# Read data from .csv file
def read_csv_file(file_path):
    file = open(file_path, "muscle1.csv")
    lines = file.readlines()
    file.close()
    for line in lines:
        body_mass.append(float(line.strip().split()[0]))
        work_level.append(float(line.strip().split()[1]))
        heat_output.append(float(line.strip().split()[2]))

    return np.array([body_mass, work_level]).T, heat_output


# Plot the figure
def plot_figs(fig_num, elev, azim, clf, a):
    fig = plt.figure(fig_num, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, elev=elev, azim=azim)

    ax.scatter(body_mass, work_level, heat_output, c='k', marker='+')

    X = np.arange(55, 85, 0.5)
    Y = np.arange(90, 180, 0.5)
    X, Y = np.meshgrid(X, Y)
    Z = a[0] + a[1] * X + a[2] * Y
    ax.plot_surface(X, Y, Z, alpha=.5, antialiased=True, rstride=200, cstride=100, cmap=plt.cm.coolwarm)

    ax.set_xlabel('BODY_MASS', color='b')
    ax.set_ylabel('WORK_LEVEL', color='b')
    ax.set_zlabel('HEAT_OUTPUT', color='b')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    ax.zaxis.set_major_locator(plt.LinearLocator(10))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.f'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.f'))
    ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.f'))


if __name__ == '__main__':
    x, y = read_csv_file('muscle1.csv')

    # Ordinary least squares Linear Regression
    lin_reg = linear_model.LinearRegression()

    # Fit linear model
    lin_reg.fit(x, y)

    # Estimated coefficients for the linear regression problem
    print('Coefficients: ', lin_reg.coef_)
    # Independent term in the linear model
    print('Independent term: ', lin_reg.intercept_)

    # Generate the three different figures from different views
    a = [lin_reg.intercept_, lin_reg.coef_[0], lin_reg.coef_[1]]
    elev = 43.5
    azim = -110
    plot_figs(1, elev, azim, lin_reg, a)
    plt.title('(1) Linear regression', color='g')

    elev = -.5
    azim = 0
    plot_figs(2, elev, azim, lin_reg, a)
    plt.title('(2) Linear regression', loc='left', color='g')

    elev = -.5
    azim = 90
    plot_figs(3, elev, azim, lin_reg, a)
    plt.title('(3) Linear regression', loc='right', color='g')

    plt.show()