'''
    Multivariate linear regression using batch vectors. Code inspired by the
    coursera course, machine learning with Andrew Ng. This program takes any
    number of parameters, including a single parameter, and outputs a projected
    pollution value. The linear model makes assumptions about the normality of
    the data, which may be violated, because we have outliers and
    collinearility.
'''

__author__ = 'Josh Malina'

import pandas as pd
import numpy as np
from pylab import *
import sys
sys.path.insert(0, '/home/utkarsh/projects/linear_and_logistic_regression/helpers')
import helpers
sys.path.insert(0, '/home/utkarsh/projects/linear_and_logistic_regression/interfaces')
import i_multivariate_linear_regression as interface
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics  import mean_squared_error


class MultivariteLinearRegression(interface.IMultivariateLinearRegression):

    def __init__(self):

        _alpha = 0.01
        _iters = 3000

        _xs, _ys = self.get_data()
        print(_xs.shape, _ys.shape)
        # print(_ysx)
        # print("hihello")
        _xs, X_test, _ys, y_test = train_test_split(_xs,_ys, test_size=0.3, random_state=42)

        self._xs, self._ys = _xs, _ys
        print(_xs.shape, _ys.shape)
        self.X_test, self.y_test = X_test, y_test

        self._thta, self._hist = self.gradient_decent(_xs, _ys, _iters,_alpha)
        self.pred = self.predict(X_test)
        print(self.pred.shape, y_test.shape)
        print("MSE: ", mean_squared_error(y_test, self.pred))
        plt.scatter(y_test,self.pred)
        plt.xlim(0,250)
        plt.ylim(0,250)
        plt.show()
        
        # self._theta = self.theta_maker(_xs, _ys, _alpha, _iters)

    def predict(self, x_vector):
        # print(x_vector)
        # print(self._theta)
        # print(x_vector.dot(self._theta))
        # thta = gradient_decent
        # print("hello",self._thta)
        # print("hello",self._theta)
        # epochs = [x for x in range(1,2501)]
        # plt.plot(epochs,self._hist)
        # plt.show()
        # plt.scatter(self._xs[:,3],self._ys)
        plt.show()
        return x_vector.dot(self._thta)

    def get_data(self):

        x_param_list = ['wind_speed_mph', 'temperature_f', 'pressure_mb', 'visibility_miles_max_10']
        # x_param_list = ['wind_speed_mph', 'visibility_miles_max_10']
        # print("hi")
        xs, ys = helpers.Helpers.get_data_2('../Data/', 'wp_remove_null_2014.csv', 'Value', x_param_list, True)
        
        # for i in range(xs.shape[1]):
        #     plt.scatter(xs[:,i],ys)
        #     plt.show()
        # plt.scatter(xs[:,2],ys)
        # plt.show()
        # print(xs)
        # print(ys)
        return xs, ys

    # def get_cost(self):

    #     # build a batch of all predictions
    #     all_results = self._xs.dot(self._theta).T

    #     # build a batch of all corresponding errors
    #     all_errors = (all_results - self._ys) ** 2

    #     # total error
    #     sum_err = sum(all_errors)
     
    #     # dividing by two "makes the math easier"
    #     # dividing by length gives us some kind of average error
    #     return sum_err / 2 / len(self._ys)

    # gradient descent algorithm for coming up with the right thetas
    def cost_function(self,X,y,B):
        m = X.shape[0]
        J = np.sum((np.dot(X,B)-y)**2)/(2*m)
        return J

    def gradient_decent(self,X,y,iterations,alpha):
        B = np.zeros(X.shape[1])
        history = [0]*iterations
        m = X.shape[0]
        # X = X[:,[0,1,2,3,4,5,6,7,10]]
        # print(X)
        # B = np.zeros(X.shape[1])
        for iteration in range(iterations):
            h = np.dot(X,B)
            loss = h - y
            # print(loss)
            der = np.dot(loss,X)/m
            # print(der)
            B = B - alpha*der
            cost = self.cost_function(X,y,B)
            # print(f"Iteration : {iteration}; Cost : {cost}")
            history[iteration] = cost

        # print(history[iterations-1])
        # print(B)
        return B,history
    
    # def theta_maker(self, xs, ys, step_size, when_stop):
    #     # print("ys=",ys)
    #     # print("xs=",xs)
    #     # ys = ys[:500]
    #     # xs = xs[:500,:]
    #     # initialize theta parameters according to how many features we are evaluating
    #     theta = np.zeros(shape=(xs.shape[1], 1))
    #     # print(len(theta))
    #     # print(theta)
    #     # print(theta)
    #     num_points = len(ys)
    #     num_thetas = len(theta)
    #     # print("xs=",xs)
    #     # print("ys=",ys)
    #     # print("step_size=",step_size)
    #     # print("when_stop=",when_stop)
    #     # stop at some arbitrary point when we think we've reached
    #     # the minimum
    #     # print(len(xs),",",xs.shape[1])
    #     # print(len(theta),",",theta.shape[1])
    #     # pred = xs.dot(theta).T
    #     # print(len(pred),",",pred.shape[1])
    #     # print(xs[0])
    #     # print(theta)
    #     # print(pred[0][0])
    #     # print(len(xs[:,1]))
    #     for i in range(when_stop):

    #         # build a vector of predictions for every x given theta
    #         # starts at theta == all 0s
    #         pred = xs.dot(theta).T

    #         # build a vector of errors for every prediction
    #         # initial errors should distance of points from 0
    #         e = pred - ys

    #         # for every theta term
    #         for j in range(0, num_thetas):

    #             # multiply error by corresponding x value           
    #             e_at_given_x = e.dot(xs[:, j])

    #             # update theta, i.e. step down if positive error / step up if neg error
    #             theta[j] -= step_size * sum(e_at_given_x) / num_points

    #         # print cost_f(xs, ys, theta)

    #     # print("theta=",theta[10])
    #     # x = np.linspace(0,10,10000)
    #     # fig = plt.figure()
    #     # ax = plt.axes()
    #     # ax.plot(x,theta[10][0]*x+theta[0][0])
    #     # plt.show()
    #     return theta    


g = MultivariteLinearRegression()
