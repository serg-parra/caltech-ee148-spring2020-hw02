import os
import numpy as np
import json
import copy
import warnings
from PIL import Image

def compute_convolution(I, T):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''

    all_dots =[]  # initialize list for collecting dot products from each kernel
    all_boxes = []  # initialize list for collecting boxes from each kernel
    output =[]  # initialize list for output (NOTE: see #2 on HW assignment)
    padding = [] # initialize list for collecting padding information
    kernel_sizes = (0.50, 1.0)
    stride = 2

    Io_dims = I.shape  # saves the dimensions of the original image
    T_image = Image.fromarray(T)  # converts the sampled kernel NumPy array back to an image for resampling

    heatmap_averages = np.empty((Io_dims[0], Io_dims[1],1), dtype=np.float16) #initialize heatmap array storage
    heatmap_averages[:] = np.NaN # makes all empty values NaNs

    for f in range(len(kernel_sizes)):  # for all kernels, do the following
    
        np_dot = []  # initialize store dot products
        boxes = []  # initialize bounding box coordinate storage
    
        T_dims = T.shape  # saves the dimensions of the original kernel image
        h = np.rint(T_dims[0]*kernel_sizes[f])  # Some % of the original kernel size
        w = h*(T_dims[1]/T_dims[0])

        Tf = np.array(T_image.resize((int(h), int(w)),Image.BILINEAR))
    
        T_dims = Tf.shape  # saves the dimensions of the re-sized kernel image
        p_h = int(np.ceil((T_dims[0]-1)/2))  # produce a padding of the ceiling of kernel height -1 (output same height)
        p_w = int(np.floor((T_dims[1]-1)/2))   # produce a padding of the ceiling of kernel width -1 (output same width)
    
        padding.append([p_h, p_w]) #  save the padding parameters used
    
        # Flattening the kernel image and normalizing the resulting vector

        Tf = Tf.ravel()  # flattens the 3D NumPy array into a 1D array
        l_2 = (np.linalg.norm(Tf, 2))  # calculates the l2 normalization value (as suggested in class), 2 is the order
        # l2 means summing the square of all the vector elements equals 1

        if l_2 == 0:
            l_2 = 1  # prevents division by zero

        Tf = Tf / l_2  # normalizes the 1D array

        # Image Padding

        I_pad = np.zeros([Io_dims[0]+(2*p_h), Io_dims[1]+(2*p_w), Io_dims[2]])  # creates a padded empty array
        I_pad[(0+p_h):(Io_dims[0]+p_h), (0+p_w):(Io_dims[1]+p_w), 0:Io_dims[2]] = I  # replaces the correct values of the image
        I_pad_dims = I_pad.shape  # saves the dimensions of the padded image
    
        # Image Flattening and Normalization

        for r in range(int(np.floor((Io_dims[0] + p_h - T_dims[0] + stride) / stride))):
            for c in range(int(np.ceil((Io_dims[1] + p_w - T_dims[1] + stride) / stride))):

                box_coords = [(r*stride), (c*stride), ((r*stride) + T_dims[0]), ((c*stride) + T_dims[1])]  # stores the bounding box coordinates in the specified format
                I_subsample = I_pad[(r*stride):(r*stride) + T_dims[0],  (c*stride):(c*stride) + T_dims[1], :]  # sub-samples a frame from the padded image
                I_subsample = I_subsample.ravel()  # flattens the 3D NumPy array into a 1D array
                I_l2 = (np.linalg.norm(I_subsample, 2))  # calculates the l2 normalization value, 2 is the order

                if I_l2 == 0:
                    I_l2 = 1  # prevents division by zero
                I_subsample = I_subsample / I_l2  # normalizes the 1D array

                # Kernel Convolution

                dotprod = np.dot(I_subsample, Tf)  # produces the dot product of image and kernel

                if dotprod > 0.92:
                    np_dot.append(dotprod)  # stores the dot product
                    boxes.append(box_coords)  # stores the box coordinates

        meancube = np.empty((I_pad_dims[0], I_pad_dims[1], len(np_dot)), dtype=np.float16)  # copies the padded array for each kernel to be produced
        meancube[:] = np.NaN  # makes all empty values NaNs
    
        for c in range(len(np_dot)):
            box_coords = boxes[c]  # extracts each set of coordinates
            dotprod_kernel = np.ones([T_dims[0], T_dims[1]]) * np_dot[c]  # creates a kernel-sized array of the dot product
            meancube[box_coords[0]:box_coords[2], box_coords[1]:box_coords[3],c] = dotprod_kernel  # stores the array in the correct mean cube layer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            meancube = np.nanmean(meancube, axis=2, dtype=np.float32)  # calculates the mean convolution product for each pixel, converts remaining NaNs to zeros
    
        all_dots.append(np_dot)  # stores the dot product
        all_boxes.append(boxes)  # stores the box coordinates
    
        # Creating arrays to average dot products and create heat map
    
        canvas = np.empty((I_pad_dims[0], I_pad_dims[1], f+1), dtype=np.float16)
        canvas[:] = np.NaN
    
        h_sz = heatmap_averages.shape
        c_sz = canvas.shape
    
        canvas[int((c_sz[0]-h_sz[0])/2):int(h_sz[0]+((c_sz[0]-h_sz[0])/2)), int((c_sz[1]-h_sz[1])/2):int(h_sz[1]+((c_sz[1]-h_sz[1])/2)),:] = heatmap_averages  # because we are stacking heatmaps of different sizes
    
        heatmap_averages = np.dstack([canvas, meancube])  # stores the meancube for each filter

    # Assign Convolution Products to Heat Map

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        heatmap_averages = np.nan_to_num(np.nanmean(heatmap_averages, axis=2))  # calculates the mean convolution products across each filter

    heatmap = np.stack([heatmap_averages,heatmap_averages, heatmap_averages], axis=2)

    #NOTE: Will need to return last padding values as well
    
    return heatmap, all_boxes, all_dots, padding

def predict_boxes(all_boxes, all_dots, padding):
    
    '''
    This function takes returns the bounding boxes and associated
    confidence scores.
    '''
    output = [] # initializes output array

    for i in range(len(all_boxes)):
        for j in range(len(all_boxes[i])):
            kernel_v = all_boxes[i]
            kernel_c = all_dots[i]
            current = kernel_v[j] #selects one of the bounding boxes
            conf = kernel_c[j] #selects the confidence value
            tl_r = current[0] - padding[i][0] #pulls out the top left corner, corrects for padding
            tl_c = current[1] - padding[i][1] #pulls out the top left corner, ""
            br_r = current[2] - padding[i][0] #pulls out the bottom right corner, ""
            br_c = current[3] - padding[i][1] #pulls out the bottom right corner, ""
            
            output.append([tl_r, tl_c, br_r, br_c, conf])

    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    owd = os.getcwd() #saves original working directory

    os.chdir('../RedLights2011_Medium')
    train_img = Image.open('RL-010.jpg')  # imports image with PIL functionality
    np_train_img = np.array(train_img)  # converts PIL image to NumPy array
    os.chdir(owd)

    T = np_train_img[26:52,321:350,:]

    heatmap, all_boxes, all_dots, padding = compute_convolution(I, T)
    output = predict_boxes(all_boxes, all_dots, padding)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../RedLights2011_Medium'

# load splits: 
split_path = '../hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
