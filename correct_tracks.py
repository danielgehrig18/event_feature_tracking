import os
import numpy as np
import copy

offsets = dict(#bicycles=1492701551.845469593,
    #rocks=1492701407.308714594,
               checkerboard=1447249152.014655732)
    #shapes=1468939993.067416019,
    #boxes=1468941032.229165635,
#poster=1468940293.840967273)

for dataset, off in offsets.items():
    path = "/home/dani/final/report/tracks/alex_2/alex/%s/tracks.csv" % dataset
    data  = np.genfromtxt(path, delimiter=",")
    data[:,1] = data[:,1] + off
    np.savetxt(path, data, delimiter=",")

