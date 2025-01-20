# JigsawHSI.py
# Copyright (c) 2022, 2025, Jaime Moraga (jmoraga@mines.edu) 
# This code is a modified version of the JigsawHSI code provided by the author as a Jupyter Notebook.
# The original code is available at https://github.com/jmoraga/JigsawHSI 
# The original paper is available at https://arxiv.org/abs/2206.02327
# Results are published in https://paperswithcode.com/paper/jigsawhsi-a-network-for-hyperspectral-image


## GLOBAL VARIABLES
# dataset = 'PU_100'
dataset = 'IP_99.74'
# dataset = 'PU_100'
# dataset = 'SA_100'


import os
# Use this before loading tensorflow
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Block INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Block INFO and WARNING messages
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Block INFO, WARNING and ERROR messages
# https://www.tensorflow.org/api_docs/python/tf/autograph/set_verbosity
os.environ["AUTOGRAPH_VERBOSITY"] = "0"

# Import learning libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
 
from sklearn.decomposition import FactorAnalysis, NMF
from sklearn.decomposition import PCA, TruncatedSVD
from operator import truediv

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import spectral

# Define functions
## JigsawHSI definition
# Import Tensorflow GPU libraries
import tensorflow as tf
tf.get_logger().setLevel("WARNING")
max_gpus = len(tf.config.list_physical_devices('GPU'))

from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Concatenate, GlobalAveragePooling2D, Multiply
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adadelta, SGD, Adam, Nadam, Adagrad, Adamax

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import utils as np_utils

from skimage.transform import rotate
from tensorflow.keras.models import clone_model
from tensorflow.keras.utils import plot_model

# Initialize all random functions with same seed
import random
import numpy as np


def random_seed():
    return os.urandom(42)

def reset_seeds(random_state = 42):
    try:
        tf.keras.utils.set_random_seed(random_state) # This resets all
        return 0
    except:
        random.seed(random_state)
        np.random.seed(random_state)
        tf.random.set_seed(random_state) # Tensorflow 2.9+
    try:
        from tensorflow import set_random_seed # Tensorflow 1.x
        set_random_seed(random_state)
        return 2
    except:
        pass
    return 1

"""
Reset all random seeds
"""
r = reset_seeds(345)
del r



# Use MirroredStrategy for multi-GPU training
strategy = tf.distribute.MirroredStrategy()

def se_block(input_tensor, reduction_ratio=16):
    """Squeeze-and-Excitation block for channel attention"""
    channels = input_tensor.shape[-1]
    
    # Squeeze
    x = GlobalAveragePooling2D()(input_tensor)
    
    # Excitation
    x = Dense(channels // reduction_ratio, activation='relu')(x)
    x = Dense(channels, activation='sigmoid')(x)
    
    # Scale
    x = Multiply()([input_tensor, x])
    return x

def jigsaw_m2(input_net, first_layer=None, internal_size=13):
    """Improved JigsawHSI module with dynamic kernels and attention"""
    kernels = []
    
    # 1x1 convolution
    x_1x1 = Conv2D(256, (1,1), padding='same', kernel_regularizer=l2(0.002))(input_net)
    x_1x1 = BatchNormalization()(x_1x1)
    x_1x1 = Activation('relu')(x_1x1)
    kernels.append(x_1x1)
    
    # Dynamically generate larger kernels
    for k in range(3, internal_size + 1, 2):
        # Reduction layer
        x_reduce = Conv2D(96 if k == 3 else 16, (1,1), padding='same', kernel_regularizer=l2(0.002))(input_net)
        x_reduce = BatchNormalization()(x_reduce)
        x_reduce = Activation('relu')(x_reduce)
        
        # Main convolution
        x = Conv2D(128 if k <= 7 else 64, (k,k), padding='same', kernel_regularizer=l2(0.002))(x_reduce)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        kernels.append(x)
    
    # Pooling branch
    pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(input_net)
    pool_proj = Conv2D(32, (1,1), padding='same', kernel_regularizer=l2(0.002))(pool)
    pool_proj = BatchNormalization()(pool_proj)
    pool_proj = Activation('relu')(pool_proj)
    kernels.append(pool_proj)
    
    # Add first layer if provided
    if first_layer is not None:
        x_first = Conv2D(96, (1,1), padding='same', kernel_regularizer=l2(0.002))(first_layer)
        x_first = BatchNormalization()(x_first)
        x_first = Activation('relu')(x_first)
        kernels.append(x_first)
    
    # Concatenate all branches
    output = Concatenate(axis=-1)(kernels)
    
    # Add SE attention
    output = se_block(output)
    
    return output

def jigsaw_m_end(input_net, num_classes, first_layer = None):
    avg_pooling = AveragePooling2D(pool_size=(3,3), strides=(1,1), name='avg_pooling')(input_net)
    flat = Flatten()(avg_pooling)
    flat = Dense(16, kernel_regularizer=l2(0.002))(flat)
    flat = Dropout(0.4)(flat)
    if first_layer is not None:
        input_pixel = Flatten()(first_layer)
        input_pixel = Dense(16, kernel_regularizer=l2(0.002))(input_pixel)
        input_pixel = Dropout(0.2)(input_pixel)
        input_pixel = Dense(16, kernel_regularizer=l2(0.002))(input_pixel)
        input_pixel = Dropout(0.2)(input_pixel)
        flat = Concatenate(axis = -1)([input_pixel, flat])
    flat = Dense(32, kernel_regularizer=l2(0.002))(flat)
    avg_pooling = Dropout(0.4)(flat)
    loss3_classifier = Dense(num_classes, kernel_regularizer=l2(0.002))(avg_pooling)
    loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)
    return loss3_classifier_act


# Builds model
def build_jigsawHSI(internal_size=13, num_classes=2, image_dim = (19, 19, 7), dimension_filters = None, verbose=1):
    with strategy.scope():
        my_input = Input( shape=image_dim )
        
        # Not needed for SA
        if ((dimension_filters is None) or (dimension_filters < 1)):
            conv1 = None
        else:
            conv1 = Conv2D(dimension_filters, (1,1), padding='same', activation = 'relu',
                          kernel_regularizer = l2(0.002), name='spectral_filter')(my_input)
        if(verbose>0):
            print(f"*** Building Jigsaw with up to {internal_size}x{internal_size} kernels")
        # One jigsaw module(s)
        jigsaw_01 = jigsaw_m2( my_input if conv1 is None else conv1, internal_size = internal_size )
        # For SA, the next two lines must be uncommented
        # jigsaw_01 = jigsaw_m2( jigsaw_01, first_layer=my_input, internal_size = internal_size )
        # jigsaw_01 = jigsaw_m2( jigsaw_01, internal_size = internal_size )
        
        # Attaches end to jigsaw modules, returns class within num_classes
        loss3_classifier_act = jigsaw_m_end(jigsaw_01,
                                        num_classes = num_classes,
                                        first_layer = my_input ) # testing num_classes
        model3 = Model( inputs = my_input, outputs = loss3_classifier_act )
    model3.compile(loss='binary_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
    return model3


class Kernel3D:
    def __init__(self, rows=3, cols=3, shape='rect', radius=None, no_value=np.nan):
        if shape == 'circle':
            self.rows = 2*radius+1
            self.cols = 2*radius+1
            self.mask = self.round_mask(radius)
            self.row_buffer = radius
            self.col_buffer = radius
        else:
            self.rows = rows
            self.cols = cols
            self.mask = np.ones((rows, cols))
            self.row_buffer = int((rows-1)/2)
            self.col_buffer = int((cols-1)/2)
        self.mask = self.mask[np.newaxis, :, :]
        self.no_value = no_value
        assert((rows%2) == 1)
        assert((cols%2) == 1)

    def round_mask(self, radius):
        diameter = 2*radius+1
        mask = np.empty((diameter, diameter,))
        mask[:] = self.no_value
        sq_radius = radius**2
        for i in range(diameter):
            for j in range(diameter):
                if ((i-radius)**2+(j-radius)**2) <= sq_radius:
                    mask[i, j] = 1
        return mask

    def getSubset(self, matrix, row, column):
        m_rows = matrix.shape[1]
        assert (row >= self.row_buffer), f"Out of bounds row {row}, from {m_rows}"
        assert (row < (m_rows-self.row_buffer)), f"Out of bounds row {row}, from {m_rows}"
        m_cols = matrix.shape[2]
        assert((column >= self.col_buffer) and (column < (m_cols-self.col_buffer))), f"Out of bounds column {column}, from {m_cols}"
        row_start = row-self.row_buffer
        row_end = row+self.row_buffer
        column_start = column-self.col_buffer
        column_end = column+self.col_buffer
        small_matrix = matrix[:, row_start:row_end+1, column_start:column_end+1]
        return small_matrix*self.mask

class GeoTiffSlicer(object):
    def __init__(self, land_matrix, kernel_rows=None, kernel_cols=None,
                 kernel_shape='rect', kernel_radius=0, no_value = np.nan):
        # (w, h, d) input tiff expected
        # (d, h, w) input tiff from rasterio must be transposed before calling this class
        if kernel_cols is None:
            kernel_cols = kernel_rows
        assert(kernel_cols < land_matrix.shape[2])
        assert(kernel_rows < land_matrix.shape[1])
        assert((kernel_shape == 'rect') or (kernel_shape == 'circle'))
        assert(kernel_radius>=0)
        if kernel_shape == 'rect':
            self.kernel = Kernel3D(rows=kernel_rows, cols=kernel_cols)
        else:
            self.kernel = Kernel3D(radius=kernel_radius,
                                   shape=kernel_shape,
                                   no_value=no_value)
            kernel_rows = kernel_cols = 2*kernel_radius+1
        self.kernel_rows = kernel_rows
        self.kernel_cols = kernel_cols
        self.land_matrix = land_matrix
        self.land_matrix_channels, self.land_matrix_cols, self.land_matrix_rows = land_matrix.shape
        self.land_matrix_cols = land_matrix.shape[2]
        self.land_matrix_rows = land_matrix.shape[1]
        self.land_matrix_channels = land_matrix.shape[0]
        self.small_row_min = self.kernel.row_buffer
        self.small_row_max = self.land_matrix_rows - self.small_row_min
        self.small_column_min = self.kernel.col_buffer
        self.small_column_max = self.land_matrix_cols - self.small_column_min

    def apply_mask(self, row, column):
        return self.kernel.getSubset(self.land_matrix, row=row, column=column)

## Decompostion functions: reduce dimensionality
# Dimensionality reduction algorithms
def applyPCA(X, numComponents=75, random_state=0):
    newX = np.reshape(X, (-1, X.shape[2])) # Reshape to columns for each band
    pca = PCA(n_components=numComponents, whiten=True, random_state=random_state)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, 'pca'

def applyFA(X, numComponents=75, random_state=0):
    newX = np.reshape(X, (-1, X.shape[2])) # Reshape to columns for each band
    fa = FactorAnalysis(n_components=numComponents, random_state=random_state)
    newX = fa.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, 'fa'

def applySVD(X, numComponents=75, random_state=0):
    newX = np.reshape(X, (-1, X.shape[2])) # Reshape to columns for each band
    svd = TruncatedSVD(n_components=numComponents, random_state=random_state)
    newX = svd.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, 'svd'

def applyNMF(X, numComponents=75, random_state=0):
    newX = np.reshape(X, (-1, X.shape[2])) # Reshape to columns for each band
    nmf = NMF(n_components=numComponents, random_state=random_state)
    newX = nmf.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, 'nmf'

def applyNone(X, numComponents=75, random_state=0):
    return X, 'None'

## Utilitary functions
def readData(dataset, data_path='../../data/HSI'):
    data_dict = {
        'IP': ('Indian_pines_corrected.mat', 'indian_pines_corrected', 'Indian_pines_gt.mat', 'indian_pines_gt'),
        'SA': ('Salinas_corrected.mat', 'salinas_corrected', 'Salinas_gt.mat', 'salinas_gt'),
        'PU': ('PaviaU.mat', 'paviaU', 'PaviaU_gt.mat', 'paviaU_gt')
    }
    (X_1, X_2, y_1, y_2) = data_dict.get(dataset[0:2].upper())
    X = sio.loadmat(os.path.join(data_path, X_1))[X_2]
    y = sio.loadmat(os.path.join(data_path, y_1))[y_2]
    
    return (X, y)

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

# Padding functions
def padWithZeros(X, margin=2):
    newX = np.pad(X, pad_width=((margin, margin),(margin, margin),(0, 0)), constant_values = 0)
    return newX

def padSymmetric(X, margin=2):
    newX = np.pad(X, pad_width=((margin, margin),(margin, margin),(0, 0)), mode = 'symmetric')
    return newX

def createImageCubes(X, y, window_size=8, removeZeroLabels = True):
    margin = int((window_size-1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], window_size, window_size, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1 , c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

# Load parameters from config file (config.ini)
## Sample config.ini section (not used in this code)
"""
    [IP_test]
    # Test configuration
    ; This is also a comment
    test_ratio   = 0.7      # Test ratio of total samples
    window_size  = 27       # Window size: (n x n x b) images
    filter_size  = 9        # Maximum filter size in jigsaw
    batch_size   = 106      # Batch size for training purposes
    max_epochs   = 200      # Maximum number of training epochs
    decomp_func  = fa       # Decomposition function to reduce dimensionality
    num_channels = 9        # Number of bands for image after decomposition
    optimizer_fn = Adadelta # Optimizer for training
    optimizer_lr = 0.1      # Learning rate for optimizer
    max_patience = 10       # Patience for loss early stop 
    output_units = 16       # Number of classes 
"""
import configparser
config = configparser.ConfigParser(inline_comment_prefixes=';#')
config.read_file(open('config.ini'))
config = config[dataset]

# Parse parameters and hyper-parameters from config file

test_ratio   = config.getfloat('test_ratio', 0.9)
window_size  = config.getint('window_size', 25)
num_channels = config.getint('num_channels', 3)
output_units = config.getint('output_units', 16)

filter_size  = config.getint('filter_size', 13)

batch_size   = config.getint('batch_size', 30)
max_epochs   = config.getint('max_epochs', 100)

decomp_func  = config.get('decomp_func', 'pca').lower()
optimizer_fn = config.get('optimizer_fn', 'sgd').lower()
optimizer_lr = config.getfloat('optimizer_lr', 0.01)
max_patience = config.getint('max_patience', 10)

hsi_filters  = config.get('hsi_filters', 'none')
hsi_filters  = None if (hsi_filters.lower() in ['none', '']) else int(hsi_filters)
print(hsi_filters)


## Select proper optimizer and decomposition functions
dict_reduction={
    'fa' : (lambda X, numComponents: applyFA(X, numComponents=numComponents)),
    'nmf' : (lambda X, numComponents: applyNMF(X, numComponents=numComponents)),
    'pca': (lambda X, numComponents: applyPCA(X, numComponents=numComponents)),
    'svd': (lambda X, numComponents: applySVD(X, numComponents=numComponents)),
    'none': (lambda X, numComponents: applyNone(X, numComponents=numComponents))
}

DimReduction=dict_reduction.get(decomp_func)

dict_optimizer = {
    'sgd'     : SGD(learning_rate=optimizer_lr, momentum=0.9, nesterov=False),
    'adadelta': Adadelta(learning_rate=optimizer_lr, rho=0.95, epsilon=1e-07),
    'adam'    : Adam(learning_rate=optimizer_lr, epsilon=1e-07),
    'nadam'   : Nadam(learning_rate=optimizer_lr, epsilon=1e-07),
    'adamax'  : Adamax(learning_rate=optimizer_lr, epsilon=1e-07),
    'adagrad' : Adagrad(learning_rate=optimizer_lr, epsilon=1e-07)
}

FuncOptimizer = dict_optimizer.get(optimizer_fn)

# Define names of output files
best_model          = dataset + '-best-model.keras'
loss_curve          = dataset + '-loss-curve.png'
acc_curve           = dataset + '-acc-curve.png'
classification_file = dataset + '-classification_report.txt'
predictions_img     = dataset + '-predictions.png'
architecture_img    = dataset + '-architecture.png'

HSI, HSI_y = readData(dataset[0:2].upper())

print(HSI.shape, HSI_y.shape)
num_channels = np.min([HSI.shape[2], num_channels])
DRI, dim_reduction = DimReduction(HSI,numComponents=num_channels)

num_channels = DRI.shape[2]
print(DRI.shape)

X, y = createImageCubes(DRI, HSI_y, window_size=window_size)

print(X.shape, y.shape)

Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, test_ratio)

print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

Xtrain = Xtrain.reshape(-1, window_size, window_size, num_channels) #, 1)
print(Xtrain.shape)

ytrain = np_utils.to_categorical(ytrain)
print(ytrain.shape)

# Summarize configuration
config_txt  = f'Configuration for dataset [{dataset}]:\n\n'
config_txt += f'Test Set Ratio: {test_ratio*100}% of samples\n'
config_txt += f'Window Size   : {window_size} pixels per side\n'
config_txt += f'Dim. Reduction: {dim_reduction} function\n'
config_txt += f'Num channels  : {num_channels} bands after {dim_reduction}\n'
config_txt += '# Network design\n'
config_txt += f'Input shape   : ({window_size}x{window_size}x{num_channels})\n'
config_txt += f'HSI Filters   : {hsi_filters} filters in first layer\n'
config_txt += f'Internal Size : ({filter_size}x{filter_size}) maximum network filter size\n'
config_txt += '# Training hyperparameters\n'
config_txt += f'Optimizer     : {optimizer_fn}\n'
config_txt += f'Learning rate : {optimizer_lr}\n'
config_txt += f'Batch Size    : {batch_size}\n'
config_txt += f'Num Epochs    : {max_epochs}\n' 
config_txt += f'Patience      : {max_patience}\n'
config_txt += '# Training GPUs\n'
config_txt += f'GPU Maximum   : {max_gpus}\n'
print(config_txt)

# import garbage collector
import gc

# Clear any existing sessions/models
tf.keras.backend.clear_session()
gc.collect()

# Optional: verify GPU is available and memory is freed
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Memory usage:", tf.config.experimental.get_memory_info('GPU:0'))

# Enable mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Before building the network, check if trained model exists
if os.path.exists(best_model):
    print(f"Loading existing model from {best_model}")
    model = tf.keras.models.load_model(best_model)
    model.compile(loss='categorical_crossentropy', optimizer=FuncOptimizer, metrics=['accuracy'])
else:
    # Build network
    input_shape =  (window_size, window_size, num_channels)
    model = clone_model(build_jigsawHSI(internal_size = filter_size,
                        num_classes = output_units,
                        verbose=1,
                        dimension_filters=hsi_filters, # Was None,
                        image_dim = input_shape))

    model.summary()
    plot_model(model)
    plot_model(model, to_file=architecture_img)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=FuncOptimizer, metrics=['accuracy'])

# Saves the best model, based on accuracy
checkpoint = ModelCheckpoint(best_model, monitor='accuracy', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

# If no early stopping desired, skip this cell
early_stop = EarlyStopping( monitor = 'loss',
                        min_delta=0.0002,
                        mode='auto',
                        verbose=1, patience=max_patience)
callbacks_list = [checkpoint, early_stop]

# Fit the model keeping the history
history = model.fit(x=Xtrain, y=ytrain, batch_size=batch_size, epochs=max_epochs, callbacks=callbacks_list)

# Validation
# Save the history charts without displaying
fig = plt.figure(figsize=(7,7)) 
plt.ioff()  # Ensure interactive mode is off
plt.grid() 
plt.plot(history.history['loss'])
plt.ylabel('Loss') 
plt.xlabel('Epochs') 
plt.legend(['Training','Validation'], loc='upper right') 
plt.savefig(loss_curve) 
plt.close(fig)

fig = plt.figure(figsize=(7,7)) 
plt.ioff()
plt.ylim(np.min(history.history['accuracy']),1.05) 
plt.grid() 
plt.plot(history.history['accuracy'])
plt.ylabel('Accuracy') 
plt.xlabel('Epochs') 
plt.legend(['Training','Validation']) 
plt.savefig(acc_curve) 
plt.close(fig)

# load best weights
model.load_weights(best_model)
model.compile(loss='categorical_crossentropy', optimizer=FuncOptimizer, metrics=['accuracy'])

# Create test sets
Xtest = Xtest.reshape(-1, window_size, window_size, num_channels) #, 1)
print(Xtest.shape)

Ytest = np_utils.to_categorical(ytest)
print(Ytest.shape)

# Predict
Y_pred_test = model.predict(Xtest)
y_pred_test = np.argmax(Y_pred_test, axis=1)

print(y_pred_test.shape) 
classification = classification_report(np.argmax(Ytest, axis=1), y_pred_test)
print(classification)

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_row_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_row_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def get_targets(name):
    targets_dict = {
        'IP': ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
               'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill', 'Soybean-clean', 'Wheat', 'Woods', 
               'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers'],
        'SA': ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow', 'Fallow_smooth', 'Stubble',
               'Celery','Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk',
               'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk', 'Vinyard_untrained',
               'Vinyard_vertical_trellis'],
        'PU': ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen', 'Self-Blocking Bricks',
               'Shadows']
    }
    targets = targets_dict.get(name)
    return (targets)


print(', '.join(get_targets(dataset[0:2].upper())))

def reports (model, X_test, y_test, name, y_pred = None):
    #start = time.time()
    if (y_pred is None):
        Y_pred = model.predict(X_test)
        y_pred = np.argmax(Y_pred, axis=1)
    #end = time.time()
    #print(end - start)
    target_names = get_targets(name[0:2].upper())
    print("Producing report")
    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
    oa = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)
    score = model.evaluate(X_test, y_test, batch_size=32)
    Test_Loss =  score[0]*100
    Test_accuracy = score[1]*100
    
    return classification, confusion, Test_Loss, Test_accuracy, oa*100, each_acc*100, aa*100, kappa*100, target_names

(classification, confusion, Test_loss, Test_accuracy, 
 oa, each_acc, aa, kappa, target_names) = reports(model, Xtest, Ytest, dataset[0:2], y_pred=y_pred_test)

target_performance = 'Accuracy by target:\n'
for (a, b) in zip(target_names, each_acc):
    target_performance += f'{b:8.4f} : {a}\n'
print(target_performance)


import seaborn as sns
cf_matrix = np.asarray(confusion)
sns_plot=sns.heatmap(cf_matrix, annot=False, xticklabels=target_names, yticklabels=target_names)
sns_plot.figure.savefig(str(dataset)+'-heatmap.png', dpi=600)

classification = str(classification)
confusion = str(confusion)
confusion=confusion.replace('\n', '')
confusion=confusion.replace('] [', ']\n[')
confusion=confusion.replace('][', ']\n[')
dim_reduction='pca'
c_summary = 'Classification Summary\n'
c_summary += f'{Test_loss:7.3f} Test loss (%)\n'
c_summary += f'{Test_accuracy:7.3f} Test accuracy (%)\n'
c_summary += f'{kappa:7.3f} Kappa accuracy (%)\n'
c_summary += f'{oa:7.3f} Overall accuracy (%)\n'
c_summary += f'{aa:7.3f} Average accuracy (%)\n\n'

c_summary += f'{classification}\n\n'
c_summary += f'{confusion}\n\n'

model_summary = []
model.summary(print_fn=lambda x: model_summary.append(x)) # line_length=70,
model_summary = '\n'.join(model_summary)
model_summary += '\n\n'

print(c_summary)
print(target_performance+'\n\n')
print(model_summary)
print(config_txt)

with open(classification_file, 'w') as cs_file:
    cs_file.write(c_summary)
    cs_file.write(target_performance+'\n\n')
    cs_file.write(model_summary)
    cs_file.write(config_txt)

def Patch(data, height_index, width_index, PATCH_SIZE):
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    
    return patch

height = HSI_y.shape[0]
width  = HSI_y.shape[1]

X = padWithZeros(DRI, window_size//2)
print(X.shape)

from tqdm.notebook import tqdm
# calculate the predicted image

def extract_patches(X, window_size, valid_indices):
    """Efficiently extract patches for all valid pixels at once"""
    half_window = window_size // 2
    patches = np.zeros((len(valid_indices[0]), window_size, window_size, X.shape[2]))
    
    for idx in range(len(valid_indices[0])):
        i, j = valid_indices[0][idx], valid_indices[1][idx]
        patches[idx] = X[i:i+window_size, j:j+window_size, :]
    
    return patches

# Get indices of non-zero pixels
valid_indices = np.where(HSI_y > 0)
outputs = np.zeros((height, width))

# Extract all patches at once
X = padWithZeros(DRI, window_size//2)
patches = extract_patches(X, window_size, valid_indices)

# Make predictions in batches
batch_size = 256  # Adjust based on your GPU memory
num_patches = len(patches)
predictions = []

for i in tqdm(range(0, num_patches, batch_size), desc="Predicting batches..."):
    batch = patches[i:min(i+batch_size, num_patches)]
    batch_predictions = model.predict(batch, verbose=0)
    predictions.append(batch_predictions)

# Combine predictions
predictions = np.vstack(predictions)
predictions = np.argmax(predictions, axis=1) + 1

# Assign predictions back to output array
outputs[valid_indices] = predictions

# Create ground truth image 
ground_truth = spectral.imshow(classes = HSI_y,figsize =(7,7))

# Create prediction image
predict_image = spectral.imshow(classes = outputs.astype(int),figsize =(7,7))

from matplotlib.colors import ListedColormap, NoNorm

cm = ListedColormap(np.array(spectral.spy_colors)/255.0)
delta = (np.abs(outputs.astype(int) - HSI_y)>0)*1
print('Misclassified pixels: ', np.sum(np.asarray(delta)>0), "/", delta.shape[0]*delta.shape[1])
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,8))
ax1.set_title("Ground truth")
ax2.set_title("Prediction")
ax3.set_title("Delta")
ax1.imshow(HSI_y) #, cmap=cm, norm=NoNorm())
ax2.imshow(outputs.astype(int)) #, cmap=cm, norm=NoNorm())
ax3.imshow(delta, cmap=cm)
plt.tight_layout()

spectral.save_rgb(str(dataset)+"-ground_truth.png", HSI_y, colors=spectral.spy_colors)
spectral.save_rgb(str(dataset)+"-delta.png", delta, colors=spectral.spy_colors)
spectral.save_rgb(predictions_img, outputs.astype(int), colors=spectral.spy_colors)

# Reset the system and release GPU memory
#del model
#print("Keras Backend RESET")  # optional
#import keras
#import gc
#keras.backend.clear_session()
#tf.keras.backend.clear_session()
#gc.collect()

