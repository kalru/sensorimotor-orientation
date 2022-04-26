params = {}

general = {
    # Number of objects represented by number of iterations
    # Run one iteration per object.
    'iterations' : 40,
    # Objects Generation parameters
    'features_per_object' : 10,
    'object_width' : 4,
    'num_features' : 40,
    'feature_distribution' : "AllFeaturesEqual_Replacement",
    'num_active_minicolumns' : 10,

    # Max number of sensations to infer
    'num_sensations' : 9,

    # Number of times each sensation should be learned
    # not in defaults
    'num_learning_points' : 3,

    'num_cortical_columns' : 2,
}
params.update(general)

# L2 Parameters
# Adapted from htmresearch.frameworks.layers.l2_l4_inference.L4L2Experiment#getDefaultL2Params
l2_params = {
    "activationThresholdDistal": 20,
    "cellCount": 4096,
    "connectedPermanenceDistal": 0.5,
    "connectedPermanenceProximal": 0.5,
    "initialDistalPermanence": 0.51,
    "initialProximalPermanence": 0.6,
    "minThresholdProximal": 5,
    "sampleSizeDistal": 30,
    "sampleSizeProximal": 10,
    "sdrSize": 40,
    "synPermDistalDec": 0.001,
    "synPermDistalInc": 0.1,
    "synPermProximalDec": 0.001,
    "synPermProximalInc": 0.1}
params['l2_params'] = l2_params

# L4 Parameters
general = {
    'threshold' : 8
    }
l4_params = {
    "columnCount": 150,
    "cellsPerColumn": 16,
    "connectedPermanence": 0.6,
    "permanenceDecrement": 0.02,
    "permanenceIncrement": 0.1,
    "apicalPredictedSegmentDecrement": 0.0,
    "basalPredictedSegmentDecrement": 0.0,
    "initialPermanence": 1.0,
    "activationThreshold": general['threshold'],
    "minThreshold": general['threshold'],
    "reducedBasalThreshold": general['threshold'],
    "sampleSize": general['threshold'],
    "implementation": "ApicalTiebreak"}
params['l4_params_general'] = general
params['l4_params'] = l4_params

# L6a Parameters
general = {
    'num_modules' : 10,
    'scale' : 40,
    # not in defaults
    'angle' : 60,
    'cells_per_axis' : 10,
}
l6a_params = {
    "moduleCount": general['num_modules'],
    "cellsPerAxis": general['cells_per_axis'],
    "activationThreshold": 8,
    "initialPermanence": 1.0,
    "connectedPermanence": 0.5,
    "learningThreshold": 8,
    "sampleSize": 10,
    "permanenceIncrement": 0.1,
    "permanenceDecrement": 0.0,
    "bumpOverlapMethod": "probabilistic"}
params['l6a_params_general'] = general
params['l6a_params'] = l6a_params
###########################################