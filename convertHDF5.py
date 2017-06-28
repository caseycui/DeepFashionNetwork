import h5py, os
import caffe
import numpy as np
from itertools import islice
import sys

SIZE = 300 # fixed size to all images
with open( 'bbox_train2.txt', 'r' ) as T :
    lines = T.readlines()


X = np.zeros( (len(lines), 3, SIZE, SIZE), dtype='f4' ) 
y1 = np.zeros( (1,len(lines)), dtype='f4' )
y2 = np.zeros( (1,len(lines)), dtype='f4' )
y3 = np.zeros( (1,len(lines)), dtype='f4' )
y4 = np.zeros( (1,len(lines)), dtype='f4' )
label = np.zeros( (1,len(lines)), dtype='f4' )

for i,l in enumerate(lines):
    sp = l.split(' ')
    img = caffe.io.load_image( sp[0] )
    img = caffe.io.resize( img, (3, SIZE, SIZE) ) # resize to fixed size
    # you may apply other input transformations here...
    # Note that the transformation should take img from size-by-size-by-3 and transpose it to 3-by-size-by-size
    X[i] = img
    y1[0] = float(sp[1])
    y2[0] = float(sp[2])
    y3[0] = float(sp[3])
    y4[0] = float(sp[4])
    label[0] = sp[5]
    if (i % 1000 == 0):
        print ("1000 lines")

with h5py.File('train2.h5','w') as H:
    H.create_dataset( 'X', data=X ) # note the name X given to the dataset!
    H.create_dataset( 'y1', data=y1 ) # note the name y given to the dataset!
    H.create_dataset( 'y2', data=y2 )
    H.create_dataset( 'y3', data=y3 )
    H.create_dataset( 'y4', data=y4 )
    H.create_dataset( 'label', data=label )
with open('train_h5_list.txt','w') as L:
    L.write( 'train2.h5' ) # list all h5 files you are going to use
