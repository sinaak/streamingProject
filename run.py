from skmultiflow.meta import OzaBagging
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.trees import HoeffdingTree
from skmultiflow.data import HyperplaneGenerator, RandomTreeGenerator, FileStream


from RestrictedHofedingTree import RHT
stream = FileStream('data/elec.csv', n_targets=1, target_idx=-1)
stream.prepare_for_use()

# instantiate a classifier
rht = RHT(K=5)

# some dudes to compare with
h = [rht,HoeffdingTree()]

# prepare the evaluator
evaluator = EvaluatePrequential(pretrain_size=1000, max_samples=100000, show_plot=True,
                                metrics=['accuracy', 'kappa'], batch_size=10)

# run
evaluator.evaluate(stream=stream, model=h, model_names=['rht','HT'])