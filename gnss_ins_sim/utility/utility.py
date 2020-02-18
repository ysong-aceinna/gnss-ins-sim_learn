# -*- coding: utf-8 -*
# Filename: utility.py

"""
Created on 2020-02-12
@author: Ocean
"""

import os
import time
import math
import numpy as np
from enum import IntEnum


def save_motion_data(input, set_of_input, file=None):
    """
    Save input data to csv file. Units:
        accel: g, gyro: deg/sec, mag: Gauss

    Args:
        input: tells gnss-ins-sim what data the algorithm need. 
            'input' is a list of strings, such as:
            input = ['fs', 'gyro', 'accel', 'mag']
        
        set_of_input: a list consistent with input.
            unit:
                fs: Hz
                gyro: rad/s
                accel: m/s^2
                mag: uT
                others refer to "https://github.com/Aceinna/gnss-ins-sim"

        file: csv output file. Both absolute and relative paths are okay.
              eg "/home/aceinna_vg.csv", "./motion_data/aceinna_vg.csv"
              if 'file' is NOT None, please make sure the parent paths have exist already.
              if 'file' is None, save file under with timestamp eg './virtual_data/2020-02-12-19-54-25.csv'.

    Returns:
        True if success, raise Error if error.
    """

    class Idx(IntEnum):
        XACCEL = 0
        YACCEL = 1
        ZACCEL = 2
        XRATE = 3
        YRATE = 4
        ZRATE = 5
        XMAG = 6
        YMAG = 7
        ZMAG = 8
        # XATEMP = 9
        # YATEMP = 10
        # ZATEMP = 11
        # XRTEMP = 12
        # YRTEMP = 13
        # ZRTEMP = 14
        # BTEMP = 15
        # N_RAW_SENS = 16

    R2D = 180.0 / math.pi
    col = 16 #16 column in csv file.
    # get n from set_of_input[1] but not set_of_input[0]
    # since set_of_input[0] is always 'fs' but not sensor data.
    n = set_of_input[1].shape[0]
    outs = np.zeros([n, col], dtype = float)

    try:
        idx = input.index('accel')
        accel = set_of_input[idx]/9.80665
    except ValueError as e:
        accel = np.zeros([n, 3], dtype = float)
    outs[0:, Idx.XACCEL:Idx.ZACCEL+1] = accel

    try:
        idx = input.index('gyro')
        gyro = set_of_input[idx]*R2D
    except ValueError as e:
        gyro = np.zeros([n, 3], dtype = float)
    outs[0:, Idx.XRATE:Idx.ZRATE+1] = gyro

    try:
        idx = input.index('mag')
        mag = set_of_input[idx]/100.0
    except ValueError as e:
        mag = np.zeros([n, 3], dtype = float)
    outs[0:, Idx.XMAG:Idx.ZMAG+1] = mag

    if file:
        this_dir = os.path.dirname(file)
        if not os.path.exists(this_dir) and this_dir != '':
            try:
                os.makedirs(this_dir)
            except:
                raise IOError('Cannot create dir: %s.'% this_dir)
        np.savetxt(file, outs, delimiter=',')
    else: # if 'file' is None, create './virtual_data/timestamp.csv'
        data_dir = os.path.abspath('./virtual_data')
        if not os.path.exists(data_dir):
            try:
                os.makedirs(data_dir)
            except:
                raise IOError('Cannot create dir: %s.'% data_dir)
        data_dir = os.path.join(data_dir, time.strftime('%Y-%m-%d-%H-%M-%S.csv', time.localtime()) )
        np.savetxt(data_dir, outs, delimiter=',')

    return True


if __name__ == '__main__':
    pass