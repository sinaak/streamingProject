from skmultiflow.trees import HoeffdingTree
import itertools
import numpy as np
from perceptron import Tree_Perceptron
from skmultiflow.drift_detection.adwin import ADWIN



class RHT:
    def __init__(self, K=1, base_learner=HoeffdingTree()):
        self.K = K
        self.base_learner = base_learner
        self.firstTime = True
        self.featureSets = list()
        self.models = list()
        self.numberOfFeatures = None
        self.alreadySeenInstances = 0
        self.adwin = ADWIN()
        self.tree_adwins = []


    def partial_fit(self, X, y, classes=None, sample_weight=None):

        n_data, n_features = X.shape

        if self.firstTime:
            self.firstTime = False
            if not n_features > self.K:
                raise Exception('K cannot be bigger than the number of features')

            featureNames = list(range(0, n_features))
            self.featureSets = list(itertools.combinations(featureNames, self.K))
            self.models = [self.base_learner] * len(self.featureSets)
            self.numberOfFeatures = len(self.featureSets)

            # for i in range(len(self.featureSets)):
            #     self.tree_adwins.append(ADWIN(delta=0.0002))

            for i, tupleFeature in enumerate(self.featureSets):
                tmpTrainX = X[:, tupleFeature]
                self.models[i].partial_fit(tmpTrainX, y)

            self.perceptron = Tree_Perceptron(n_class=2, n_tree=len(self.featureSets), n_feature=self.numberOfFeatures)

        for i, x in enumerate(X):
            p = []
            choosenY = y[i]
            for i, tupleFeature in enumerate(self.featureSets):
                tmpTrainx = np.take(x, tupleFeature)
                tmpTrainx = tmpTrainx.reshape(1, tmpTrainx.shape[0])
                p.append(self.models[i].predict_proba(tmpTrainx))
                # tree_pred = self.models[i].predict(tmpTrainx)
                # self.add_adwin_tree(i, choosenY, tree_pred)

            # self.add_adwin(x, choosenY) # add adwin filter
            self.perceptron.update_parameters(choosenY, p)


        for i, tupleFeature in enumerate(self.featureSets):
            tmpTrainX = X[:, tupleFeature]
            self.models[i].partial_fit(tmpTrainX, y)

        self.alreadySeenInstances += n_data

        #print(self.alreadySeenInstances)

    def add_adwin_tree(self, i, y, y_hat):
        element = 0
        if y == y_hat : element = 1
        self.tree_adwins[i].add_element(element)
        if self.tree_adwins[i].detected_change():
            print('here!')
            self.perceptron.reset_tree_params(i)
            # reset the model as well!


    def add_adwin(self, x, y):
        element = 0
        y_hat = self.predict(np.array([x]))
        if y_hat == y: element = 1
        self.adwin.add_element(element)
        if self.adwin.detected_change():
            print('here!')
            self.perceptron.reset_learning_rate()


    def predict(self, X):
        N, D = X.shape

        res = self.predict_proba(X)

        return res
        #return res



    def predict_proba(self, X):
        N, D = X.shape

        res = np.zeros(N)

        for j, x in enumerate(X):

            Prob_List = []
            for i, tupleFeature in enumerate(self.featureSets):
                tmpTrainx = np.take(x, tupleFeature)
                tmpTrainx = tmpTrainx.reshape(1, tmpTrainx.shape[0])
                Prob_List.append(self.models[i].predict_proba(tmpTrainx))


            P = self.perceptron.predict_proba(Prob_List)
            res[j] = np.argmax(P)

        return res
