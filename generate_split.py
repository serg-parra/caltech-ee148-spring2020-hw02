import numpy as np
import os
import json

np.random.seed(2020)  # to ensure you always get the same train/test split

preds_path = '../../HW1/hw01_preds'
data_path = '../RedLights2011_Medium'
gts_path = '../hw02_annotations'
split_path = '../hw02_splits'
os.makedirs(preds_path, exist_ok=True)  # create directory if needed

split_test = True  # set to True and run when annotations are available

train_frac = 0.85

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

# split file names into train and test

# select random test and train indices
test_frac = int(np.around((1-train_frac)*len(file_names)))  # number of images needed from the number of images supplied

array_indices = np.arange(len(file_names))
random_selection = np.random.choice(array_indices, size=test_frac, replace=False)  # samples without replacement

train_indices = np.delete(array_indices, random_selection)  # erases the selected testing indices

file_names_test = [file_names[i] for i in random_selection]
file_names_train = [file_names[i] for i in train_indices]

assert (len(file_names_train) + len(file_names_test)) == len(file_names)
assert len(np.intersect1d(file_names_train,file_names_test)) == 0

np.save(os.path.join(split_path,'file_names_train.npy'),file_names_train)
np.save(os.path.join(split_path,'file_names_test.npy'),file_names_test)

if split_test:
    with open(os.path.join(gts_path, 'formatted_annotations_students.json'),'r') as f:
        gts = json.load(f)
    
    # Use file_names_train and file_names_test to apply the split to the
    # annotations
    
    gts_train = {}
    gts_test = {}
    
    '''
    Your code below. 
    '''
    gts_filenames = list(gts.keys())  # converts filenames to list items
    gts_fnames_train = {gts_filenames[i] for i in train_indices}
    gts_fnames_test = {gts_filenames[i] for i in random_selection}
    
    gts_train = {key: gts[key] for key in gts.keys() & gts_fnames_train}
    gts_test = {key: gts[key] for key in gts.keys() & gts_fnames_test}
        
    with open(os.path.join(gts_path, 'annotations_train.json'),'w') as f:
        json.dump(gts_train,f)
    
    with open(os.path.join(gts_path, 'annotations_test.json'),'w') as f:
        json.dump(gts_test,f)
