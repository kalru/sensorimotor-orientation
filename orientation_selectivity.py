from default_params import params
from union_path_integration_multi_column_convergence import MultipleExperiments
from plots import GeneratePlots
import os
import json
import time
import plotly.io as pio

pio.kaleido.scope.default_width = 750
pio.kaleido.scope.default_height = 480
pio.kaleido.scope.default_scale = 2

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# helper function
def load_file(path):
    with open(os.path.join(SCRIPT_DIR, path)) as f:
        return(json.load(f))

experiment = {
    'debug' : True,
    ##### grid cells
    # increase to 360 to make modules swappable
    'angle' : 360,
    'num_modules' : [25],
    'cells_per_axis' : [13], #13 because thats wehere transition started
    ##### general
    'repetitions' : 5,
    'num_learning_points' : 3,
    'threshold' : 8,
    'num_cortical_columns' : [2],
    # Max number of sensations to infer
    'num_sensations' : 10,
    'exp_name' : "Sensation by Columns",
    #orientationAlgo: 0 - No algo applied
    #orientationAlgo: 1 - Ideal algo applied
    #orientationAlgo: 2 - Converging algo applied
    'orientationAlgo': [2],
    # this param limits the maximum amount of samples from the randomly
    # generated objects that can be used for inference. Its useful for testing
    'objectSampleLimit': 2000,
    # this param specifies which object pool to sample from
    # It can be 'shifted', 'rotated', 'normal', a combination of them in a string i.e. 'normal and rotated'
    'objectSamplePool': 'rotated',
    ##### object generation
    'iterations' : [50],
    'num_features' : [40],
    'features_per_object' : [10],
    # this controls how many random orientations are generated for each object on inference
    'random_gen': 10, # generate 500 objects to make distplot
    #if rotations is 'None', do nothing. Else rotations is given by '10,12,50,..' as a string 
    # (since lists are used for multi experiments)
    # the rotations are sampled from for each object for random_gen times
    'rotations':'None', #allign it for 10 modules
    # seed from current time in seconds
    'seed': int(time.time())
}
params['experiment'] = experiment

focused_feature = "cells_per_axis"
savefile = "results/25_13_dist_result.json"
MultipleExperiments(savefile, focused_feature, params)
plots = GeneratePlots(load_file(savefile))
orientation_10_selectivity_25_13 = plots.generateOrientationDist(10)
orientation_18_selectivity_25_13 = plots.generateOrientationDist(18)
plots.generateOrientationDist(18, show_title=False).write_image("%s/%s.png" % (savefile.split('/')[0], 'orientation_18_selectivity_25_13'))
plots.generateOrientationDist(10, show_title=False).write_image("%s/%s.png" % (savefile.split('/')[0], 'orientation_10_selectivity_25_13'))
