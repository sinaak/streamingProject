from skmultiflow.meta import OzaBagging
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.trees import HoeffdingTree
from skmultiflow.data import HyperplaneGenerator, RandomTreeGenerator, FileStream, LEDGenerator
from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator
from RestrictedHofedingTree import RHT
from skmultiflow.trees.hoeffding_adaptive_tree import HAT

stream = FileStream('data/elec.csv', n_targets=1, target_idx=-1)
stream.prepare_for_use()

stream_LED = LEDGenerator()
stream_LED.prepare_for_use()

# instantiate a classifier
rhtA = RHT(K=2, nc=2)
rhtB = RHT(K=2, nc=2,adwinEnabler=True)


h = [rhtA, rhtB]

# prepare the evaluator
evaluator = EvaluatePrequential(pretrain_size=100, max_samples=40000, show_plot=True, metrics=['accuracy', 'kappa'],
                                batch_size=1)

# run
evaluator.evaluate(stream=stream, model=h, model_names=['RHT','RHT ADWIN'])
