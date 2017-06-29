import h5py, os
import caffe
import numpy as np
from itertools import islice
import sys

SIZE = 300 # fixed size to all images
with open( 'minidata_hdf5.txt', 'r' ) as T :
    lines = T.readlines()


X = np.zeros( (len(lines), 3, SIZE, SIZE), dtype='f4' ) 
y1 = np.zeros( (len(lines),1,1,1), dtype='f4' )
y2 = np.zeros( (len(lines),1,1,1), dtype='f4' )
y3 = np.zeros( (len(lines),1,1,1), dtype='f4' )
y4 = np.zeros( (len(lines),1,1,1), dtype='f4' )
label = np.zeros( (len(lines),1,1,1), dtype='f4' )


#y1 = np.zeros( (1,len(lines)), dtype='f4' )
#y2 = np.zeros( (1,len(lines)), dtype='f4' )
#y3 = np.zeros( (1,len(lines)), dtype='f4' )
#y4 = np.zeros( (1,len(lines)), dtype='f4' )
#label = np.zeros( (1,len(lines)), dtype='f4')

for i,l in enumerate(lines):
    sp = l.split(' ')
    img = caffe.io.load_image( sp[0] )
    img = caffe.io.resize( img, (3, SIZE, SIZE) ) # resize to fixed size
    # you may apply other input transformations here...
    # Note that the transformation should take img from size-by-size-by-3 and transpose it to 3-by-size-by-size
    X[i] = img
    y1[1] = float(sp[1])/300.0
    y2[1] = float(sp[2])/300.0
    y3[1] = float(sp[3])/300.0
    y4[1] = float(sp[4])/300.0
    label[1] = sp[5]
    if (i % 1000 == 0):
        print ("1000 lines")

#    X=X.reshape(len(lines),3,SIZE,SIZE)
#    y1=y1.reshape(1,len(lines))
#    y2=y2.reshape(1,len(lines))
#    y3=y3.reshape(1,len(lines))
#    y4=y4.reshape(1,len(lines))

with h5py.File('minidata.h5','w') as H:
    H.create_dataset( 'data', data=X ) # note the name X given to the dataset!
    H.create_dataset( 'x1', data=y1 ) # note the name y given to the dataset!
    H.create_dataset( 'y1', data=y2 )
    H.create_dataset( 'x2', data=y3 )
    H.create_dataset( 'y2', data=y4 )
    H.create_dataset( 'category', data=label )
with open('minidata_hdf5_path.txt','w') as L:
    L.write( 'minidata.h5' ) # list all h5 files you are going to use
