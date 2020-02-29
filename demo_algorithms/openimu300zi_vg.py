# -*- coding: utf-8 -*-
# Fielname = openimu300zi_vg.py

"""
simulate OpenIMU300ZI VG algo.
Created on 2020-02-13
@author: Ocean
"""

# import
import os
import sys
import struct
import platform
import math
import numpy as np
from ctypes import *
from gnss_ins_sim.utility import utility
import _ctypes

# globals
VERSION = '1.0'

R2D = 180.0 / math.pi

# used to fetch sim result from algorithm.
class EKF_STATE(Structure):
    '''
    Return EFK state in this structure
    '''
    _fields_ = [("timeStep", c_uint32),
                ("kfPosN", c_float*3),
                ("kfVelN", c_float*3),
                ("kfQuat", c_float*4),
                ("kfRateBias", c_float*3),
                ("kfAccelBias", c_float*3),
                ("kfCorrectedRateB", c_float*3),
                ("kfCorrectedAccelB", c_float*3),
                ("algoFilteredYawRate", c_float),
                ("kfEulerAngles", c_float*3),
                ("algoState", c_int),
                ("algoTurnSwitch", c_ushort),
                ("algoLinAccelSwitch", c_uint8),
                ("algoAMag", c_float),
                ("algoAFiltN", c_float * 3),]

class OpenIMU300ZISim(object):
    '''
    A wrapper form DMU380 algorithm offline simulation.
    '''
    def __init__(self, config_file):
        '''
        Args:
            config_file: a configuration file
        '''
        # platform
        if sys.platform.startswith('win'):
            self.ext = '.dll'
            if struct.calcsize("P") == 8:
                self.ext = '-x64' + self.ext
            else:
                self.ext = '-x86' + self.ext
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            self.ext = '.so'
        elif sys.platform.startswith('darwin'):
            self.ext = '.dylib'
        else:
            raise EnvironmentError('Unsupported platform')

        # algorithm description
        self.input = ['fs', 'gyro', 'accel', 'mag']
        self.output = ['algo_time', 'att_euler', 'wb']
        self.batch = True
        self.results = None
        # algorithm vars
        this_dir = os.path.dirname(__file__)
        self.sim_lib = os.path.join(this_dir, 'dmu380_sim_lib/libdmu380_algo_sim' + self.ext)
        if not (os.path.exists(self.sim_lib)):
            if not self.build_lib():
                raise OSError('Shared libs not found.')
        self.config_file = config_file
        self.sim_engine = cdll.LoadLibrary(self.sim_lib)
        self.sim_engine.parseConfigFile(c_char_p(self.config_file.encode('utf-8')))
        # initialize algorithm
        self.sim_engine.SimInitialize()

    def run(self, set_of_input):
        '''
        main procedure of the algorithm
        Args:
            set_of_input is a tuple or list consistent with self.input
                fs: sample frequency, Hz
                gyro: numpy array of size (n,3), rad/s
                accel: numpy array of size (n,3), m/s/s
                mag: numpy array of size (n,3), Gauss
        '''
        # get input
        # utility.save_motion_data(self.input, set_of_input)
        # np.save("motion_data.npy", set_of_input)
        # return
        # set_of_input = np.load( "ahrs-90deg_noise.npy" )

        fs = set_of_input[0]
        gyro = set_of_input[1]
        accel = set_of_input[2]
        if 'mag' in self.input:
            mag = set_of_input[3]
        n = accel.shape[0]
        # algo output
        time_step = np.zeros((n,))
        euler_angles = np.zeros((n, 3))
        rate_bias = np.zeros((n, 3))
        # run
        ekf_state = EKF_STATE()
        output_len = 0
        for i in range(0, n):
            sensor_data = np.zeros((15,))
            sensor_data[3:6] = gyro[i, :]*R2D
            sensor_data[0:3] = accel[i, :]/9.80665
            if 'mag' in self.input:
                sensor_data[6:9] = mag[i, :]/100.0
            sensorReadings = sensor_data.ctypes.data_as(POINTER(c_double))
            new_results = self.sim_engine.SimRun(sensorReadings)
            # get output
            if new_results == 1:
                self.sim_engine.GetEKF_STATES(pointer(ekf_state))
                # time_step[output_len] = ekf_state.timeStep / fs
                time_step[output_len] = i / fs
                # Euler angles order is [roll pitch yaw] in the algo
                # We use yaw [pitch roll yaw] order in the simulation
                euler_angles[output_len, 0] = ekf_state.kfEulerAngles[2]
                euler_angles[output_len, 1] = ekf_state.kfEulerAngles[1]
                euler_angles[output_len, 2] = ekf_state.kfEulerAngles[0]
                rate_bias[output_len, 0] = ekf_state.kfRateBias[0]
                rate_bias[output_len, 1] = ekf_state.kfRateBias[1]
                rate_bias[output_len, 2] = ekf_state.kfRateBias[2]
                output_len += 1
                # print("{0}:{1:0.5f}, {2:0.5f}, {3:0.5f}".format(output_len,
                #     ekf_state.kfEulerAngles[0],ekf_state.kfEulerAngles[1],ekf_state.kfEulerAngles[2]))
        # results
        self.results = [time_step[0:output_len],\
                        euler_angles[0:output_len, :],\
                        rate_bias[0:output_len, :]]

    def update(self, gyro, acc, mag=np.array([0.0, 0.0, 0.0])):
        '''
        Mahony filter for gyro, acc and mag.
        Args:
        Returns:
        '''
        pass

    def get_results(self):
        '''
        Returns:
            algorithm results as specified in self.output
                algorithm time step, sec
                Euler angles [yaw pitch roll], rad
        '''
        return self.results

    def reset(self):
        '''
        Reset the fusion process to uninitialized state.
        '''
        # free library ref: https://www.programcreek.com/python/example/56939/_ctypes.FreeLibrary
        _ctypes.dlclose(self.sim_engine._handle)
        self.sim_engine = cdll.LoadLibrary(self.sim_lib)
        self.sim_engine.parseConfigFile(c_char_p(self.config_file.encode('utf-8')))
        self.sim_engine.SimInitialize()

    def build_lib(self, dst_dir=None, src_dir=None):
        '''
        Build shared lib
        Args:
            dst_dir: dir to put the built libs in.
            src_dir: dir containing the source code.
        Returns:
            True if success, False if error.
        '''
        if self.ext == '.dll':
            print("Automatic generation of .dll is not supported.")
            return False
        this_dir = os.path.dirname(__file__)
        # get dir containing the source code
        if src_dir is None:
            src_dir = os.path.join(this_dir, 
                    '/Users/songyang/project/code/github/dmu380_sim_src')
        if not os.path.exists(src_dir):
            print('Source code directory ' + src_dir + ' does not exist.')
            return False
        # get dir to put the libs in
        if dst_dir is None:
            dst_dir = os.path.join(this_dir, './/dmu380_sim_lib//')
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        algo_lib = 'libdmu380_algo_sim' + self.ext 
        sim_utilities_lib = 'libsim_utilities' + self.ext 
        # get current workding dir
        cwd = os.getcwd()
        # create the cmake dir
        cmake_dir = src_dir + '//cmake//'
        if not os.path.exists(cmake_dir):
            os.mkdir(cmake_dir)
        else:
            os.system("rm -rf " + cmake_dir + "*")
        # call cmake and make to build the libs
        os.chdir(cmake_dir)
        ret = os.system("cmake ..")
        ret = os.system("make")
        algo_lib = os.path.join(cmake_dir, 'sim', algo_lib )
        # algo_lib = cmake_dir + 'sim//' + algo_lib
        sim_utilities_lib = cmake_dir + 'SimUtilities//' + sim_utilities_lib
        # if os.path.exists(algo_lib) and os.path.exists(sim_utilities_lib):
        if os.path.exists(algo_lib):
            os.system("mv " + algo_lib + " " + dst_dir)
            # os.system("mv " + sim_utilities_lib + " " + dst_dir)

        # restore working dir
        os.chdir(cwd)
        return True
