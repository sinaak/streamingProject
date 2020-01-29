from skmultiflow.trees import HoeffdingTree
import itertools
import numpy as np
from perceptron import Tree_Perceptron


class RHT:
    def __init__(self, K=1, base_learner=HoeffdingTree()):
        self.K = K
        self.base_learner = base_learner
        self.firstTime = True
        self.featureSets = list()
        self.models = list()
        self.numberOfFeatures = None
        self.alreadySeenInstances = 0

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

            self.perceptron.update_parameters(choosenY, p)

        for i, tupleFeature in enumerate(self.featureSets):
            tmpTrainX = X[:, tupleFeature]
            self.models[i].partial_fit(tmpTrainX, y)

        self.alreadySeenInstances += n_data

        #print(self.alreadySeenInstances)



    def predict(self, X):
        N, D = X.shape

        self.predict_proba(X)

        return np.zeros(N)



    def predict_proba(self, X):

        for i, x in enumerate(X):
            Prob_List = []
            for i, tupleFeature in enumerate(self.featureSets):
                tmpTrainx = np.take(x, tupleFeature)
                tmpTrainx = tmpTrainx.reshape(1, tmpTrainx.shape[0])
                Prob_List.append(self.models[i].predict_proba(tmpTrainx))

            #print(Prob_List)
            P = self.perceptron.predict_proba(Prob_List)
            print(P)
