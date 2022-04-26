import time
import collections
import json
import os
import random

import matplotlib
from matplotlib.ticker import MaxNLocator

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from htm.advanced.frameworks.location.location_network_creation import L246aNetwork
# local copy
# from location_network_creation import L246aNetwork
from htm.advanced.support.object_generation import generateObjects
from htm.advanced.support.expsuite import PyExperimentSuite
from htm.advanced.support.register_regions import registerAllAdvancedRegions, registerAdvancedRegion
from htm.bindings.engine_internal import Network

#logging
from rich.console import Console, OverflowMethod
console = Console()
from flatten_dict import flatten
import plotly.graph_objects as go

#local
from utils import Object_Augmentation
from plots import GeneratePlots

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


class Experiment():    
    def __init__(self, params):
        self.params = params

    def get_param(self, param, default_key=None):
        """
        Returns the most relevant param. Experiment specific params override
        default params. If no default key is provided then the default param
        will be retrieved from the root of the param dict

        Args:
            param (str): param key to get
            default_key (str): default key to get it at if its not defined in
                                the experiment

        Returns:
            str: the most relevant version of the requested param
        """
        if param in self.params['experiment']:
            return self.params['experiment'][param]
        elif default_key is not None:
            return self.params[default_key][param]
        else:
            return self.params[param]

    def reset(self):
        """
        Take the steps necessary to reset the experiment before each repetition:
            - Make sure random seed is different for each repetition
            - Create the L2-L4-L6a network
            - Generate objects used by the experiment
            - Learn all objects used by the experiment
        """
        console.print(self.get_param('exp_name'), ":", "repitition number")

        self.debug = self.get_param('debug')
        # this is to scale the amount of samples generated randomly per object
        self.random_gen = self.get_param('random_gen')

        L2Params = self.get_param('l2_params')
        L4Params = self.get_param('l4_params')
        L6aParams = self.get_param('l6a_params')

        # fix num_modules=moduleCount. they have different names but mean the same
        L6aParams['moduleCount'] = self.get_param('num_modules', 'l6a_params_general')
        # also fix cells_per_axis=cellsPerAxis. they have different names but mean the same
        L6aParams['cellsPerAxis'] = self.get_param('cells_per_axis', 'l6a_params_general')


        # Make sure random seed is different for each repetition
        # seed = 42
        seed = self.get_param('seed')
        print('seed is ', str(seed))
        # seed = np.random.randint(0, 1000)
        np.random.seed(seed)
        random.seed(seed)
        L2Params["seed"] = seed
        L4Params["seed"] = seed
        L6aParams["seed"] = seed

        # Configure L6a params
        numModules = self.get_param('num_modules', 'l6a_params_general')
        L6aParams["scale"] = [self.get_param('scale', 'l6a_params_general')] * numModules
        # old orientations... replaced with new (360 based ones)
        # ang = self.get_param('angle', 'l6a_params_general') // numModules
        # orientation = list(range(ang // 2, ang * numModules, ang))
        # from previous work (no columns grid cells)
        perModRange = float(self.get_param('angle', 'l6a_params_general') / float(numModules))
        #hierdie distribute die orientations evenly (NIE random nie)
        orientation = [(float(i) * perModRange) + (perModRange / 2.0) for i in range(numModules)]
        L6aParams["orientation"] = np.radians(orientation).tolist()

        # Create multi-column L2-L4-L6a network
        self.numColumns = self.get_param('num_cortical_columns')
        self.network = L246aNetwork(numColumns=self.numColumns, 
                                    L2Params=L2Params,
                                    L4Params=L4Params, 
                                    L6aParams=L6aParams,
                                    repeat=0,
                                    logCalls=self.debug)

        # Use the number of iterations as the number of objects. This will allow us
        # to execute one iteration per object and use the "iteration" parameter as
        # the object index
        numObjects = self.get_param('iterations')

        # Generate feature SDRs
        numFeatures = self.get_param('num_features')
        numOfMinicolumns = L4Params["columnCount"]
        numOfActiveMinicolumns = self.get_param('num_active_minicolumns')
        self.featureSDR = [{
            str(f): sorted(np.random.choice(numOfMinicolumns, numOfActiveMinicolumns))
            for f in range(numFeatures)
        } for _ in range(self.numColumns)]

        # Generate objects used in the experiment
        self.objects = generateObjects(numObjects=numObjects,
                                            featuresPerObject=self.get_param('features_per_object'),
                                            objectWidth=self.get_param('object_width'),
                                            numFeatures=numFeatures,
                                            distribution=self.get_param('feature_distribution'))

        #shift all objects up by 20 to fit them nicely inside positive
        #array indicies. currently they have top @ 0
        aug = Object_Augmentation(self.objects)
        for obj in self.objects:
            aug.shift(obj['name'], [[20, 0]], replace=True)
        shifts = [
            [10,10],
            [20,20],
            ]
        # shifted_augmentations = []
        shifted_augmentations = aug.shift(str(0), shifts)
        # diff = 360/numModules
        # print(diff, numModules)
        # rotations = np.random.randint(24, size=10)*diff

        #if rotations is 'None', do nothing. Else rotations is given by '10,12,50,..' as a string 
        # (since lists are used for multi experiments)
        # the rotations are sampled from for each object for random_gen times
        rr = self.get_param('rotations')
        if rr != 'None':
            fixed_rotations = [int(i) for i in rr.split(',')]
        else:
            fixed_rotations = None
        # rotations = [
        #     # 15,
        #     # 30,
        #     45,
        #     # 60,
        #     # 75,
        #     # 105,
        #     # 135,
        #     150,
        #     # 180,
        #     210,
        # ]
        rotated_augmentations = []
        for obj in self.objects:
            if fixed_rotations:
                if self.random_gen > len(fixed_rotations):
                    rotations = random.choices(fixed_rotations, k=self.random_gen)
                else:
                    rotations = random.sample(fixed_rotations, self.random_gen)
            else:
                rotations = np.random.randint(360, size=self.random_gen).tolist()
            # rotations = np.random.randint(numModules, size=2)*diff
            # rotations = np.random.randint(360, size=10)
            rotated_augmentations.extend(aug.rotate(str(obj['name']), rotations))
        
        #add augmented objects (shifted)
        # self.objects.extend(shifted_augmentations)
        self.shifted_augmentations = shifted_augmentations
        #add augmented objects (rotated)
        # self.objects.extend(rotated_augmentations)
        # #temporary only infer augmented
        # self.objects = rotated_augmentations
        self.rotated_augmentations = rotated_augmentations

        self.sdrSize = L2Params["sdrSize"]

        # Learn objects
        self.numLearningPoints = self.get_param('num_learning_points')
        self.numOfSensations = self.get_param('num_sensations')
        # # if its larger make it the same as features... (to account for max possible)
        # if numFeatures < self.numOfSensations:
        #     self.numOfSensations = numFeatures
        self.learnedObjects = {}
        self.learn()

    def iterate(self, params, repetition, iteration):
        """
        [infer original objects...]
        For each iteration try to infer the object represented by the 'iteration'
        parameter returning the number of touches required to unambiguously
        classify the object.
        :param params: Specific parameters for this iteration. See 'experiments.cfg'
                                     for list of parameters
        :param repetition: Current repetition
        :param iteration: Use the iteration to select the object to infer
        :return: number of touches required to unambiguously classify the object
        """
        objectToInfer = self.objects[iteration]
        stats = collections.defaultdict(list)
        touches = self.infer(objectToInfer, stats)
        results = {'touches': touches}
        results.update(stats)

        return results

    def infer_set(self, objects):
        """Infer a list of objects. This could be any type of object in the specified
        format. (augmented or original)

        Args:
            objects (list): set of object descriptions to infer
        """
        results = []
        for obj in objects:
            stats = collections.defaultdict(list)
            touches = self.infer(obj, stats)
            stats['touches'] = touches
            # save learned objects. convert sets to lists. and int64 to int
            temp = {}
            for obj_name in self.network.learnedObjects:
                temp[obj_name] = []
                for act in self.network.learnedObjects[obj_name]:
                    temp[obj_name].append([int(i) for i in list(act)])
            stats['learnedObjects'] = temp
            results.append(stats)
            stats['object'] = obj
        return results


    def learn(self):
        """
        Learn all objects on every column. Each column will learn all the features
        of every object and store the the object's L2 representation to be later
        used in the inference stage
        """
        self.setLearning(True)

        for obj in self.objects:
            self.sendReset()

            previousLocation = [None] * self.numColumns
            displacement = [0., 0.]
            features = obj["features"]
            numOfFeatures = len(features)

            # Randomize touch sequences
            touchSequence = np.random.permutation(numOfFeatures)

            for sensation in range(numOfFeatures):
                for col in range(self.numColumns):
                    # Shift the touch sequence for each column
                    colSequence = np.roll(touchSequence, col)
                    feature = features[colSequence[sensation]]
                    # Move the sensor to the center of the object
                    locationOnObject = np.array([feature["top"] + feature["height"] / 2., feature["left"] + feature["width"] / 2.])
                    # Calculate displacement from previous location
                    if previousLocation[col] is not None:
                        displacement = locationOnObject - previousLocation[col]
                    previousLocation[col] = locationOnObject
                    # wierd workaround fix json decoding error
                    if type(displacement) == np.ndarray:
                        displacement = displacement.tolist()
                    # learn each pattern multiple times
                    activeColumns = self.featureSDR[col][feature["name"]]
                    for _ in range(self.numLearningPoints):
                        # Sense feature at location
                        self.network.motorInput[col].executeCommand('addDataToQueue', displacement)
                        self.network.sensorInput[col].executeCommand('addDataToQueue', activeColumns, False, 0)
                        # Only move to the location on the first sensation.
                        displacement = [0, 0]

            self.network.network.run(numOfFeatures * self.numLearningPoints)

            # update L2 representations for the object
            self.learnedObjects[obj["name"]] = self.getL2Representations()
            self.network.learnedObjects[obj["name"]] = self.getL2Representations()###asdf

    def log_inference_stats(self, stats):
        """Log inference results to neptune

        Args:
            stats (list): list of regions sdr's, overlaps and classification status
                            for each column on each inferred object
        """
        #log learned object representations for each column
        numCells = self.get_param('cellCount', 'l2_params')
        total_objects = self.get_param('iterations')
        for icol in range(self.get_param('num_cortical_columns')):
            im = np.empty((numCells, 0))
            for iobj in range(total_objects):
                temp = np.zeros((numCells, 1))
                temp[list(self.network.learnedObjects[str(iobj)][icol])] = 1
                im = np.hstack((im, temp))
            fig = go.Figure()
            fig.add_heatmap(z=im,
                            xgap=1,
                            colorscale=[[0, 'white'], [1.0, 'black']],
                            showscale=False
                            )
            fig.update_layout(title='L2 Object Representations in C' + str(icol),
                                height=numCells,
                                yaxis_title="Activated Cells",
                                xaxis_title="Object #") 

        #assume same keys across objects. i.e. all objects was learned
        for key in stats[0]:#use first obj to get keys
            if type(stats[0][key]) is list and 'SDR' not in key and 'orientational_firing_rate' not in key:#sensation based metrics, leave SDR's and orientational_firing_rate out for now
                key_values = [] #objects x sensations
                for obj in stats:
                    temp = []
                    for sensation in obj[key]:
                        temp.append(sensation)
                    key_values.append(temp)
                average_key_values = np.average(np.array(key_values), 0).tolist()
            elif 'SDR' in key and 0:#cell activity graphs, block for now...
                for iobj, obj in enumerate(stats):
                    #set the correct size for the sdr
                    if 'L2' in key:  
                        numCells = self.get_param('cellCount', 'l2_params') 
                        im = np.empty((numCells, 0))
                    else:
                        print('unknown sdr')
                    for sensation in obj[key]:
                        temp = np.zeros((numCells, 1))
                        temp[sensation] = 1
                        im = np.hstack((im, temp))

                    inferred_step = obj['touches']
                    vis = True
                    if inferred_step == None:
                        vis = False
                        inferred_step = 0

                    fig = go.Figure()
                    fig.add_heatmap(z=im,
                                    x=np.array(range(len(obj[key])))+1,#starts from 1, thus needs to be offset
                                    xgap=1,
                                    colorscale=[[0, 'white'], [1.0, 'black']],
                                    showscale=False
                                    ).add_shape(
                                                visible=vis,
                                                type='rect',
                                                x0=inferred_step - 0.5, x1=inferred_step + 1 - 0.5, y0=0 - 0.5, y1=numCells - 0.5,
                                                line_color='red'
                                            )
                    if obj['object'].get('augmentation', None) is not None:
                        aug = ' ' + obj['object']['augmentation']['type'] + '@' + str(obj['object']['augmentation']['amount'])
                    else:
                        aug = ''
                    fig.update_layout(title=key + ' Cell Activity for Object ' + str(iobj) + '(' + obj['object']['name'] + aug + ')',
                                        height=numCells,
                                        yaxis_title="Activated Cells",
                                        xaxis_title="Sensations") 


    def check_best_orientations(self, touchSequence, features):
        # assume that there haven't been any sensations so far
        # only look certain depth in sensations
        max_depth = self.numOfSensations
        previousLocation = [None] * self.numColumns
        displacement = [0., 0.]
        # assume cols have the same number of modules
        orientations = json.loads(self.network.L6aRegions[0].executeCommand("getOrientations"))
        for i, orientation in enumerate(orientations):
            if np.degrees(orientation) < np.degrees(orientations[i+1]):
                diff = np.degrees(orientations[i+1]) - np.degrees(orientation)
                break
        # make a 3d array with orientations x cols x sensations 
        orientational_firing_rate = np.ndarray((len(orientations),self.numColumns,max_depth))
        for orientation in range(len(orientations)):
            # first rotate the modules
            [self.network.L6aRegions[col].executeCommand('rotate_sensory_connections', orientation, False) for col in range(self.numColumns)]
            for sensation in range(max_depth):
                for col in range(self.numColumns):
                    # Shift the touch sequence for each column
                    colSequence = np.roll(touchSequence, col)
                    feature = features[colSequence[sensation]]
                    # Move the sensor to the center of the object
                    locationOnObject = np.array([feature["top"] + feature["height"] / 2., feature["left"] + feature["width"] / 2.])
                    # Calculate displacement from previous location
                    if previousLocation[col] is not None:
                        displacement = locationOnObject - previousLocation[col]
                    previousLocation[col] = locationOnObject
                    # wierd workaround fix json decoding error
                    if type(displacement) == np.ndarray:
                        displacement = displacement.tolist()
                    # Sense feature at location
                    # self.network.motorInput[col].executeCommand('addDataToQueue', displacement)
                    # self.network.sensorInput[col].executeCommand('addDataToQueue', self.featureSDR[col][feature["name"]], False, 0)
                    # adding traversals
                    for _ in range(2):#range(self.numLearningPoints):
                        # Sense feature at location
                        self.network.motorInput[col].executeCommand('addDataToQueue', displacement)
                        self.network.sensorInput[col].executeCommand('addDataToQueue', self.featureSDR[col][feature["name"]], False, 0)
                        # Only move to the location on the first sensation.
                        displacement = [0, 0]
                self.network.network.run(1* 2)#self.numLearningPoints)
                activations = self.network.getL6aSensoryAssociatedCells()#getL6aSensoryAssociatedCells()#getL6aRepresentations
                # save cell activations
                for c in range(self.numColumns):
                    rate = len(activations[c])
                    if len(activations[c])==0:
                        rate = 1000
                    orientational_firing_rate[orientation][c][sensation] = 1/(rate)
            # console.print('resetting orientation and network in new sensation')
            for col in range(self.numColumns):
                self.network.L6aRegions[col].executeCommand("reset_sensory_connection_rotation")
            self.sendReset()
        
        lowest_steps = np.argmax(np.sum(np.sum(orientational_firing_rate, axis=2), axis=1))

        # compare random orientation selections to ideal ones... make a graph... also make one for orientation selectivity?
        # also fix only using sum to compare... mybe remove 0's?

        # for ss in range(max_depth):
        #     sense = orientational_firing_rate[0][:,ss]
        #     print(np.min(sense)) if not np.isin(0.0, sense) else print('nope')

        # for o in range(len(orientations)):
        #     print(np.argwhere(orientational_firing_rate[o]==0.0).shape[0])

        # may not contain 0's
        if np.isin(0.0, orientational_firing_rate[lowest_steps]):
            console.print("Optimal orientation (%d steps) contains 0 active cells!" % lowest_steps, style="bold red")
        console.print("Optimal Orientation is %d steps (%d degrees). Rotating..." % (lowest_steps, lowest_steps*diff))
        [self.network.L6aRegions[col].executeCommand('rotate_sensory_connections', lowest_steps, False) for col in range(self.numColumns)]
        return orientational_firing_rate, int(lowest_steps)
        

    def infer(self, objectToInfer, stats=None):
        """
        Attempt to recognize the specified object with the network. Randomly move
        the sensor over the object until the object is recognized.
        """
        if objectToInfer.get('augmentation', None) is not None:
            aug = '(' + objectToInfer['augmentation']['type'] + '@' + str(objectToInfer['augmentation']['amount']) + ')'
        else:
            aug = ''
        style = "bold green"
        console.print("Inferring Object %s %s" % (objectToInfer['name'], aug), style=style)

        self.setLearning(False)
        self.sendReset()

        # it seems that this is not neccesary anymore
        console.print('resetting orientation before inference...')
        for col in range(self.numColumns):
            self.network.L6aRegions[col].executeCommand("reset_sensory_connection_rotation")
        self.sendReset()

        touches = None
        previousLocation = [None] * self.numColumns
        displacement = [0., 0.]
        features = objectToInfer["features"]
        objName = objectToInfer["name"]
        numOfFeatures = len(features)

        orientational_firing_rate = None
        # initialize as 0 for no rotations
        delta_steps = 0

        angle_ground_truth = 0 
        #orientationAlgo: 0 - No algo applied
        #orientationAlgo: 1 - Ideal algo applied
        #orientationAlgo: 2 - Converging algo applied


        # first calculate ideal orientation
        orientations = json.loads(self.network.L6aRegions[col].executeCommand("getOrientations"))
        for i, orientation in enumerate(orientations):
            if np.degrees(orientation) < np.degrees(orientations[i+1]):
                diff = np.degrees(orientations[i+1]) - np.degrees(orientation)
                break
        
        if objectToInfer.get('augmentation', None) is not None:
            if objectToInfer['augmentation']['type'] == 'rotation':
                #assume that it devides evenly
                delta_steps = (objectToInfer['augmentation']['amount']/diff) - int(self.network.L6aRegions[col].executeCommand("getStepsRotated"))
            else:
                if self.get_param('orientationAlgo') == 1:
                    # if its only shifted ideal case is 0
                    stats['chosen_orientation'] = 0

        if delta_steps != int(delta_steps):
            console.print("steps to next augmentation orientation is not a clean int. rounding from %f to %d" % (delta_steps, round(delta_steps)))
            delta_steps = round(delta_steps)
        assert delta_steps == int(delta_steps), 'augmentation amount needs to be a multiple of module orientation resolution, i.e. divide cleanly'
        delta_steps = int(delta_steps)
        
        if self.debug:
            stats['ideal_orientation'] = delta_steps
            console.print('Ideal steps is  %d steps (%d degrees)' % (delta_steps, delta_steps*diff))
            if self.get_param('orientationAlgo') == 0:
                stats['chosen_orientation'] = 0

        # Randomize touch sequences
        touchSequence = np.random.permutation(numOfFeatures)
        if len(touchSequence) > self.numOfSensations:
            touchSequence = touchSequence[:self.numOfSensations]
        if len(touchSequence) < self.numOfSensations:
            temp = touchSequence.tolist()
            temp.extend(random.choices(temp, k=self.numOfSensations-len(touchSequence)))
            touchSequence = np.array(temp)
        
        # for debugging ordered sequence
        # touchSequence = list(range(len(objectToInfer["features"])))

        # base algo
        # if self.get_param('orientationAlgo') == 0:
        #     for sensation in range(self.numOfSensations):
        #         # Add sensation for all columns at once
        #         for col in range(self.numColumns):
        #             # Shift the touch sequence for each column
        #             colSequence = np.roll(touchSequence, col)
        #             feature = features[colSequence[sensation]]
        #             # Move the sensor to the center of the object
        #             locationOnObject = np.array([feature["top"] + feature["height"] / 2., feature["left"] + feature["width"] / 2.])
        #             # Calculate displacement from previous location
        #             if previousLocation[col] is not None:
        #                 displacement = locationOnObject - previousLocation[col]
        #             previousLocation[col] = locationOnObject
        #             # wierd workaround fix json decoding error
        #             if type(displacement) == np.ndarray:
        #                 displacement = displacement.tolist()
        #             # Sense feature at location
        #             self.network.motorInput[col].executeCommand('addDataToQueue', displacement)
        #             self.network.sensorInput[col].executeCommand('addDataToQueue', self.featureSDR[col][feature["name"]], False, 0)
        #         self.network.network.run(1)
        #         if self.debug:
        #             self.network.updateInferenceStats(stats, objectName=objName)

        #         if touches is None and self.network.isObjectClassified(objName, minOverlap=30):
        #             touches = sensation + 1
        #             neptune.log_metric('touches', touches)
        #             neptune.log_text('inferred_status', 'object inferred')
        #             if not self.debug:
        #                 return touches
        # # ideal case (rotate precisely)
        # elif self.get_param('orientationAlgo') == 1:
        #     console.print('111111111')
        
        if objectToInfer.get('augmentation', None) is not None:
            aug = ' ' + objectToInfer['augmentation']['type'] + '@' + str(objectToInfer['augmentation']['amount'])
            if objectToInfer['augmentation']['type'] == 'rotation':
                for col in range(self.numColumns):
                    console.print('object %s rotation augmentation @: %s degrees (%s steps currently in col_%d)' % \
                        (objectToInfer['name'], objectToInfer['augmentation']['amount'], self.network.L6aRegions[col].executeCommand("getStepsRotated"), col))
                    angle_ground_truth = objectToInfer['augmentation']['amount']

                    #ideal, rotate precisely
                    if self.get_param('orientationAlgo') == 1:
                        # for col in range(self.numColumns):
                        # orientations = json.loads(self.network.L6aRegions[col].executeCommand("getOrientations"))
                        # for i, orientation in enumerate(orientations):
                        #     if np.degrees(orientation) < np.degrees(orientations[i+1]):
                        #         diff = np.degrees(orientations[i+1]) - np.degrees(orientation)
                        #         break

                        # #assume that it devides evenly
                        # delta_steps = (objectToInfer['augmentation']['amount']/diff) - int(self.network.L6aRegions[col].executeCommand("getStepsRotated"))
                        # if delta_steps != int(delta_steps):
                        #     console.print("steps to next augmentation orientation is not a clean int. rounding from %f to %d" % (delta_steps, round(delta_steps)))
                        #     delta_steps = round(delta_steps)
                        # assert delta_steps == int(delta_steps), 'augmentation amount n eeds to be a multiple of module orientation resolution, i.e. divide cleanly'
                        # delta_steps = int(delta_steps)

                        if self.debug:
                            stats['chosen_orientation'] = delta_steps

                        self.network.L6aRegions[col].executeCommand('rotate_sensory_connections', delta_steps, False)
                        # self.steps_rotated += delta_steps
                        console.print('Obj %s ideal oriented modules col_%d ' % (objectToInfer['name'], col), 
                                diff*int(self.network.L6aRegions[col].executeCommand("getStepsRotated")) == angle_ground_truth, 
                                diff*int(self.network.L6aRegions[col].executeCommand("getStepsRotated")), angle_ground_truth)

        # first orient modules
        if self.get_param('orientationAlgo') == 2:
            
            orientational_firing_rate, lowest = self.check_best_orientations(touchSequence, features)
            # logging
            if self.debug:
                stats['orientational_firing_rate'] = orientational_firing_rate.tolist()
                stats['chosen_orientation'] = lowest

        for sensation in range(self.numOfSensations):
            # Add sensation for all columns at once
            for col in range(self.numColumns):
                # Shift the touch sequence for each column
                colSequence = np.roll(touchSequence, col)
                feature = features[colSequence[sensation]]
                # Move the sensor to the center of the object
                locationOnObject = np.array([feature["top"] + feature["height"] / 2., feature["left"] + feature["width"] / 2.])
                # Calculate displacement from previous location
                if previousLocation[col] is not None:
                    displacement = locationOnObject - previousLocation[col]
                previousLocation[col] = locationOnObject
                # wierd workaround fix json decoding error
                if type(displacement) == np.ndarray:
                    displacement = displacement.tolist()
                # Sense feature at location
                self.network.motorInput[col].executeCommand('addDataToQueue', displacement)
                self.network.sensorInput[col].executeCommand('addDataToQueue', self.featureSDR[col][feature["name"]], False, 0)
            self.network.network.run(1)
            if self.debug:
                self.network.updateInferenceStats(stats, objectName=objName)
                # also add full L6 sdr here, because i dont want to edit the network. this can be updated later if needed
                for i, rep in enumerate(self.network.getL6aRepresentations()):
                    stats["Full L6 SDR C" + str(i)].append(sorted([int(c) for c in rep]))

            if touches is None and self.network.isObjectClassified(objName, minOverlap=30):
                touches = sensation + 1
                style = "bold blue"
                console.print("[bold blue]Inferred Object %s %s in %d touches" % (objectToInfer['name'], aug, touches), style=style)
                if not self.debug:
                    return touches

        if touches is None:
            style = "bold red"
            console.print("[bold red]Could not Infer Object %s %s in %d touches" % (objectToInfer['name'], aug, self.numOfSensations), style=style)
        return touches
        # return self.numOfSensations if touches is None else touches

    def setLearning(self, learn):
        """
        Set all regions in every column into the given learning mode
        """
        self.network.setLearning(learn)

    def sendReset(self):
        """
        Sends a reset signal to all regions in the network.
        It should be called before changing objects.
        """
        self.network.sendReset()

    def getL2Representations(self):
        """
        Returns the active representation in L2.
        """
        return self.network.getL2Representations()


class MultipleExperiments:
    def __init__(self, resultName,focused_feature, params):
        self.focused_feature = focused_feature
        #list of (experiment params, results)
        experiments = []
        #first setup parameters:
        #get the max list len first:
        num_exp = max([len(values) for key, values in params["experiment"].items() if type(values) is list])
        # do it like this for correct reference links
        for i in range(num_exp):
            experiments.append({'params':{},'results':{}})
            #add defaults
            for key, value in params.items():
                if key != "experiment":
                    experiments[i]['params'][key] = value
                else:
                    experiments[i]['params'][key] = {}
        for key, values in params["experiment"].items():
            if type(values) is list:
                for iExperiment in range(len(experiments)):
                    #add first values as default for many experiments
                    if iExperiment > len(values)-1:
                        val = values[0]
                    else:
                        val = values[iExperiment]
                    #add param to relevant exp
                    experiments[iExperiment]['params']["experiment"][key] = val
            else:
                #set param in all experiments
                #and fill in the remaining spots
                for i in range(len(experiments)):
                    experiments[i]['params']["experiment"][key] = values
        #now get results
        for ex in experiments:
            ex["focused_feature"] = focused_feature
            style = "purple"
            console.rule("[bold yellow]%s : %s" % (focused_feature, ex['params']["experiment"][focused_feature]), style=style, align='center')
            ex['results'] = self.doExperiment(ex['params'])
            print('done experiment')
        
        console.print("Logging aggregate plots...")
        plots = GeneratePlots(experiments)
        # log_chart(name='L6a SensoryAssociatedCells Distribution', chart=plots.generateDensityRidge('L6a SensoryAssociatedCells'))
        # log_chart(name='L6a LearnableCells Distribution', chart=plots.generateDensityRidge('L6a LearnableCells'))
        # log_chart(name='L6a Representation Distribution', chart=plots.generateDensityRidge('L6a Representation'))
        # # also log density plots for location representations
        # log_chart(name='L6a SensoryAssociatedCells Density', chart=plots.generateDensity('L6a SensoryAssociatedCells', normalization_key='Location'))
        # log_chart(name='L6a LearnableCells Density', chart=plots.generateDensity('L6a LearnableCells', normalization_key='Location'))
        # log_chart(name='L6a Representation Density', chart=plots.generateDensity('L6a Representation', normalization_key='Location'))        
        # log_chart(name='L4 Representation Density', chart=plots.generateDensityRidge('L4 Representation'))
        # log_chart(name='L4 Predicted Density', chart=plots.generateDensityRidge('L4 Predicted'))
        # log_chart(name='L2 Representation Density', chart=plots.generateDensityRidge('L2 Representation'))
        # log_chart(name='Overlap L2 with object Density', chart=plots.generateDensityRidge('Overlap L2 with object'))
        # log_chart(name='Sensation Convergence', chart=plots.generateTouchesHist())
        # log_chart(name='Inference Error', chart=plots.generateInferredDist())
        # log_chart(name='Directional Error', chart=plots.generateDirectionalError())
        # # get max number of modules (only applies when the number of modules is varied)
        # max_mods = max([ex['params']['experiment']['num_modules'] for ex in experiments])
        # [log_chart(name='Directional Selectivity for direction %d' % i, chart=plots.generateDirectionalSelectivityStacked(i)) for i in range(max_mods)]
        

        #save to file
        with open(os.path.join(SCRIPT_DIR, resultName),"w") as f:
            print("Writing results to {}".format(resultName))
            json.dump(experiments,f)

        # return experiments

    def doExperiment(self, params):
        # neptune
        flattened_params = flatten(params, reducer='path')

        # limit the amount of objects evaluated during inference up to a maximum
        objectSampleLimit = params["experiment"]['objectSampleLimit']

        # this param specifies which object pool to sample from
        # It can be 'shifted', 'rotated', 'normal', a combination of them in a string i.e. 'normal and rotated'
        objectSamplePool = params["experiment"]['objectSamplePool']

        test = Experiment(params)
        test.reset()

        console.print("Learned objects")
        ll = []
        if 'shifted' in objectSamplePool:
            ll.extend(test.shifted_augmentations)
        # ll = test.rotated_augmentations[10:]
        # ll.extend(test.objects)
        if 'rotated' in objectSamplePool:
            ll.extend(test.rotated_augmentations)
        if 'normal' in objectSamplePool:
            ll.extend(test.objects)

        if objectSampleLimit < len(ll):
            object_pool = random.sample(ll, objectSampleLimit)
        else:
            object_pool = ll
        inferred = test.infer_set(object_pool)
        # old way
        # inferred = []
        # for i in range(test.get_param('iterations')):
        #     inferred.append(test.iterate('params', 'repitition', i))
        #     neptune.log_metric('classified (minOverlap=30)', test.network.isObjectClassified(str(i), minOverlap=30))
        console.print("Inferred objects")
        #log stats to neptune.ai across all objects (average)

        test.log_inference_stats(inferred)
        console.print("Logged Progress")
        return inferred


# registerAllAdvancedRegions()
# need to register all induvidualldy because grid region is local
for regionName in [
                    "RawValues",
                    "RawSensor",
                    # "GridCellLocationRegion",
                    "ApicalTMPairRegion", 
                    "ApicalTMSequenceRegion",
                    "ColumnPoolerRegion",
                    ]:
    registerAdvancedRegion(regionName)
# register local grid region
registerAdvancedRegion("GridCellLocationRegion", "GridCellLocationRegion")

# multi = MultipleExperiments("results/result.json", "iterations", params)

# test = Experiment(params)
# test.reset()

# print(dir(test.network.network)) 

# print("Learned objects")
# inferred = []
# for i in range(test.get_param('iterations')):
#     inferred.append(test.iterate('params', 'repitition', i))
#     neptune.log_metric('classified (minOverlap=30)', test.network.isObjectClassified(str(i), minOverlap=30))
# print("Inferred objects")
# #log stats to neptune.ai across all objects (average)

# test.log_inference_stats(inferred)
# print("Logged Progress")