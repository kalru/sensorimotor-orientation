import numpy as np
from scipy import ndimage

class Object_Augmentation():
    """This class is for augmenting created objects by either rotating or shifting them
       from their original positions
    """
    def __init__(self, objects):
        """Initialize Class with generated objects

        Args:
            objects (list): generated objects
        """
        self.objects = objects

    def shift(self, name, shifts, append=False, replace=False):
        """Shift a specified object in given directions

        Args:
            name (str): object name
            shifts (list): list of shifts to apply [[shift1_y, shift1_x],[shift2_y, shift2_x]...]
            append (bool, optional): Append to original list. Defaults to False.

        Returns:
            list: newly augmented copies of the original object
        """
        #offset = [row, column] (x, y)
        obj = [obj for obj in self.objects if obj['name'] == name]
        assert len(obj) == 1, 'Somehow there\'s duplicate object names... This should not happen'
        obj = obj[0]
        if replace:
            assert len(shifts) == 1, "Only one augment can be applied to replace object"

        augmentations = []
        for s in shifts:
            features = []
            for f in obj['features']:
                features.append({
                    "left": f['left'] + s[1],
                    "top": f['top'] + s[0],
                    "width": f['width'],
                    "height": f['height'],
                    "name": f['name']
                })
            augmentations.append({
                'features': features,
                'name' : obj['name'],
                'augmentation': {
                    'type': 'shift',
                    'amount': s
                }
            })

        if append:
            return self.objects.extend(augmentations)
        elif replace:
            for o in self.objects:
                if o['name'] == obj['name']:
                    o['features'] = augmentations[0]['features']
                    break
        else:
            return augmentations

    def rotate(self, name, rotations):
        #rotation = degrees
        obj = [obj for obj in self.objects if obj['name'] == name]
        assert len(obj) == 1, 'Somehow there\'s duplicate object names... This should not happen'
        obj = obj[0]
        #reassign features to temporary values (index of list + 1, cuz 0 is taken)
        features_original = [int(f['name']) for f in obj['features']]
        features_temp = np.array(range(1, len(features_original)+1))
        #check that there is no duplicate features
        assert len(np.unique(features_temp)) == len(features_original), 'Some features are duplicated in this object... Rotations do not account for it'
        augmentations = []
        for r in rotations:
            features = [] 
            z = np.zeros((60, 60))
            for i_f, f in enumerate(obj['features']):
                #pad until features fit
                total_width = f['left'] + f['width']
                if total_width > z.shape[1]:
                    z = np.pad(z, ((0,0), (0,total_width - z.shape[1])))
                total_height = f['top']
                if total_height > z.shape[0]:
                    z = np.pad(z, ((0,total_height - z.shape[0]), (0,0)))
                #debug, for plotting real values
                # z[f['top']-f['height']:f['top'], f['left']:f['left']+f['width']] += int(f['name'])
                z[f['top']-f['height']:f['top'], f['left']:f['left']+f['width']] += features_temp[i_f]

            rotated = ndimage.rotate(z, r, reshape=True, order=0)
            #find bounding box of all features...
            for f in features_temp:
                box = np.where(rotated==f)
                # calculate rotated features (bounding box will have same midpoint(used for learning/inference) as rotated feature)
                features.append({
                    "left": int(np.min(box[1])),
                    "top": int(np.max(box[0])),
                    "width": int(np.max(box[1]) - np.min(box[1])),
                    "height": int(np.max(box[0]) - np.min(box[0])),
                    "name": str(features_original[f-1])
                })
            # #debug, this will show bounding boxes if plotted
            # import matplotlib
            # matplotlib.use('TkAgg')
            # import matplotlib.pyplot as plt
            # # rotated[np.min(box[0]): np.max(box[0]), np.min(box[1]): np.max(box[1])] += 1
            # plt.imshow(rotated)
            # plt.title(str(r))
            # plt.gca().invert_yaxis()
            # plt.show()
            augmentations.append({
                'features': features,
                'features_original': obj['features'],
                'name' : obj['name'],
                'augmentation': {
                    'type': 'rotation',
                    'amount': r
                }
            })

        return augmentations

    # sit padding by generated objects, sodat dit gerotate kan word...
    # nee... jy moet net die rigtings reg update instead?...
    # 1) try eers met offsets sonder rotation, en dan met offset en rotations later