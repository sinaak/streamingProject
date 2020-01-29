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
        self.n_counter = 0
        self.epsilon = 0.001


    def update_parameters(self, y, Prob_List):
        # update learning rate
        self.instance_counter += 1
        self.n_counter += 1
        self.learning_rate = 2 / (2 + self.n_counter + self.n_feature)

        # fixing zero frequency
        Prob_List = self.fix_zero(Prob_List)

        for i in range(self.n_class):
            ai = self.get_ai(Prob_List, i)
            yp = 0
            if y == i: yp = 1 # this class was the correct class
            self.run_ai(ai, i, yp)



    def fix_zero(self, Prob_List):
        new_Prob_List = []

        for P in Prob_List:

            if len(P) < self.n_class :# ?
                n = self.n_class - len(P)
                zero_row = np.zeros((n, 1))
                P = np.concatenate((zero_row, P), axis=1)

            P = P + self.epsilon
            P = P/P[0].sum()
            new_Prob_List.append(P)
        return new_Prob_List


    def get_ai(self, Prob_List, class_i):
        ci = []
        ai = []
        for P in Prob_List:
            cij = P.item(class_i)
            ci.append(cij)
            ai.append( math.log(cij / (1 - cij)) )
        return ai


    def run_ai(self, ai, class_i, yp):
        # update parameters
        theta = np.zeros(self.n_tree + 1)

        for j in range(self.n_tree + 1):
            theta[j] = self.parameters[class_i][j]

        theta = self.logistic_regression(np.array([ai]), yp, self.learning_rate, 3, theta)
        for j in range(self.n_tree + 1):
            self.parameters[class_i][j] = theta[0][j]


    def yhat(self, theta, X, n): # predict based on current parameters
        h = np.ones((X.shape[0], 1))
        theta = theta.reshape(1, n + 1)
        for i in range(0, X.shape[0]):
            h[i] = 1 / (1 + np.exp(-float(np.matmul(theta, X[i]))))
        h = h.reshape(X.shape[0])
        return h


    def GD(self, theta, alpha, num_iters, h, X, y, n):
        for i in range(0, num_iters):
            theta[0] = theta[0] - (alpha / X.shape[0]) * sum(h - y)
            for j in range(1, n + 1):
                theta[j] = theta[j] - (alpha / X.shape[0]) * sum((h - y) * X.transpose()[j])
            # h = self.yhat(theta, X, n)
        theta = theta.reshape(1, n + 1)
        return theta


    def logistic_regression(self, X, y, alpha, num_iters, theta):
        n = X.shape[1]
        one_column = np.ones((X.shape[0], 1))
        X = np.concatenate((one_column, X), axis=1)
        h = self.yhat(theta, X, n)
        theta = self.GD(theta, alpha, num_iters, h, X, y, n)
        return theta


    def run_perceptron(self, ai, class_i):
        theta = np.zeros(self.n_tree + 1)
        for j in range(self.n_tree + 1):
            theta[j] = self.parameters[class_i][j]

        n = ai.shape[1]
        one_column = np.ones((ai.shape[0], 1))
        x = np.concatenate((one_column, ai), axis=1)
        prob = self.yhat(theta, x, n)
        return prob[0]



    def predict_proba(self, Prob_List):
        Prob_List = self.fix_zero(Prob_List)
        P = [0 for x in range(self.n_class)]
        for i in range(self.n_class):
            ai = self.get_ai(Prob_List, i)
            P[i] = self.run_perceptron(np.array([ai]), i)
        P = [(x / sum(P)) for x in P] # makes it as probability vector

        return P


    def make_param_table(self, n_class, n_tree):
        # makes a 2D array to keep the learning theta
        param_table = [[(1/n_class) for x in range(n_tree+1)] for y in range(n_class)]
        return param_table

    def reset_learning_rate(self):
        self.n_counter = 0
        self.learning_rate = 2 / (2 + self.n_counter + self.n_feature)
