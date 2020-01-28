import math
import numpy as np

class Tree_Perceptron:

    def __init__(self, n_class, n_tree, n_feature):

        self.learning_rate = 0
        self.n_class = n_class
        self.n_feature = n_feature
        self.n_tree = n_tree
        self.parameters = self.make_param_table(n_class, n_tree)
        self.instance_counter = 0


    def update_parameters(self, X, y, Prob_List): # y should be the class number (zero base??)
        for i in range(self.n_class):
            ai = self.get_ai(Prob_List, i)
            yp = 0
            if y == i: #This class was the correct class
                yp = 1
            self.run_ai(ai, i, X, yp)

        # Update Learning Rate
        self.instance_counter += 1
        self.learning_rate = 2 / (2 + self.instance_counter + self.n_feature)

    def get_ai(self, Prob_List, class_i):
        ci = []
        ai = []

        for P in Prob_List:
            cij = P[class_i]
            ci = ci.append(cij)
            ai = ai.append( math.log(cij / (1 - cij)) )
        return ai

    def run_ai(self, ai, class_i, X, yp):
        # update parameters
        theta = np.zeros(self.n_tree)
        for j in range(self.n_tree):
            theta[j] = self.parameters[class_i][j]
        theta, cost = self.logistic_regression(X, yp, self.learning_rate, 3, theta=None)

    def hypothesis(self, theta, X, n):
        h = np.ones((X.shape[0], 1))
        theta = theta.reshape(1, n + 1)
        for i in range(0, X.shape[0]):
            h[i] = 1 / (1 + np.exp(-float(np.matmul(theta, X[i]))))
        h = h.reshape(X.shape[0])
        return h

    def GD(self, theta, alpha, num_iters, h, X, y, n):
        cost = np.ones(num_iters)
        for i in range(0, num_iters):
            theta[0] = theta[0] - (alpha / X.shape[0]) * sum(h - y)
            for j in range(1, n + 1):
                theta[j] = theta[j] - (alpha / X.shape[0]) * sum((h - y) * X.transpose()[j])
            h = self.hypothesis(theta, X, n)
            cost[i] = (-1 / X.shape[0]) * sum(y * np.log(h) + (np.ones(len(y)) - y) * np.log(1 - h))
        theta = theta.reshape(1, n + 1)
        return theta, cost

    def logistic_regression(self, X, y, alpha, num_iters, theta=None):
        n = X.shape[1]
        if theta.all() == None:
            theta = np.zeros(n + 1)

        one_column = np.ones((X.shape[0], 1))
        X = np.concatenate((one_column, X), axis=1)
        h = self.hypothesis(theta, X, n)
        theta, cost = self.GD(theta, alpha, num_iters, h, X, y, n)
        return theta, cost

    def run_perceptron(self, ai, class_i):

        theta = np.zeros(self.n_tree)
        for j in range(self.n_tree):
            theta[j] = self.parameters[class_i][j]

        prob = 1 / (1 + np.exp(-float(np.matmul(theta, ai))))

        return prob

    def predict_proba(self, X, Prob_List):

        P = [0 for x in range(self.n_class)]

        for i in range(self.n_class):
            ai = self.get_ai(Prob_List, i)
            P[i] = self.run_perceptron(ai, i)

        P = [round(x / sum(P), 4) for x in P]

        return P


    def make_param_table(self, n_class, n_tree):
        #make a big table to keep the learning theta
        param_table = [[(1/n_class) for x in range(n_tree)] for y in range(n_class)]
        return param_table