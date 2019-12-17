# -*- coding: utf-8 -*-
# Fielname = inclinometer_acc.py

"""
odometer simulation
Created on 2019-12-17
@author: Ocean
"""

# import
import numpy as np
from gnss_ins_sim.attitude import attitude
from gnss_ins_sim.geoparams import geoparams
import math

# globals
VERSION = '1.0'

class OdometerSim(object):
    '''
    Integrate gyro to get attitude, double integrate linear acceleration to get position.
    '''
    def __init__(self, ini_pos_vel_att, earth_rot=True):
        '''
        Args:
            ini_pos_vel_att: 9x1 numpy array containing initial position, velocity and attitude.
                3x1 position in the form of LLA, units: [rad, rad, m];
                3x1 velocity in the body frame, units: m/s;
                3x1 Euler anels [yaw, pitch, roll], rotation sequency is ZYX, rad.
            earth_rot: Consider the Earth rotation or not. Only used when ref_frame=0.
        '''
        # algorithm description
        self.input = ['ref_frame', 'fs', 'gyro', 'accel', 'gps', 'gps_visibility']
        self.output = ['att_euler', 'pos', 'vel']
        self.earth_rot = earth_rot
        self.batch = True
        self.results = None
        # algorithm vars
        self.ref_frame = 1
        self.dt = 1.0
        self.att = None
        self.pos = None
        self.vel = None
        self.vel_b = None
        # ini state
        self.set_of_inis = 1        # number of sets of inis
        self.run_times = int(0)     # algo run times. If run_times <= set_of_inis, the i-th run
                                    # uses the i-th set of initial states. Otherwise, the first
                                    # set of initial states will be used
        # only one set of inis is provided, multiple runs have the same inis.
        if ini_pos_vel_att.ndim == 1:
            self.set_of_inis = 1
            ini_pos_vel_att = ini_pos_vel_att.reshape((ini_pos_vel_att.shape[0], 1))
        # multiple set of inis is provided, multiple runs can have different inis.
        elif ini_pos_vel_att.ndim == 2:
            self.set_of_inis = ini_pos_vel_att.shape[1]
        else:
            raise ValueError('Initial states should be a 1D or 2D numpy array, \
                              but the dimension is %s.'% ini_pos_vel_att.ndim)
        self.r0 = ini_pos_vel_att[0:3]
        self.v0 = ini_pos_vel_att[3:6]
        self.att0 = ini_pos_vel_att[6:9]
        self.gravity = None
        if len(ini_pos_vel_att) > 9:
            self.gravity = ini_pos_vel_att[9]

    def run(self, set_of_input):
        '''
        main procedure of the algorithm
        Args:
            set_of_input is a tuple or list consistent with self.input
        '''
        self.run_times += 1
        # get input
        if set_of_input[0] == 0:
            self.ref_frame = 0
        self.dt = 1.0 / set_of_input[1]
        gyro = set_of_input[2]
        accel = set_of_input[3]
        gps = set_of_input[4]
        n = accel.shape[0]
        # Free IMU and odometer integration
        self.att = np.zeros((n, 3))
        self.pos = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))     # NED vel
        self.vel_b = np.zeros((n, 3))   # body vel
        c_bn = np.eye((3))
        if self.ref_frame == 1:
            # which set of initial states to use
            idx = self.run_times - 1
            if self.run_times > self.set_of_inis:
                idx = 0
            # Earth gravity
            if self.gravity is None:
                earth_param = geoparams.geo_param(self.r0)    # geo parameters
                g_n = np.array([0, 0, earth_param[2]]) # r0[LLA]处的g_n
            else:
                g_n = np.array([0, 0, self.gravity[idx]])
            for i in range(n):
                #### initialize
                if i == 0:
                    # 实际应用中，取GPS信号丢失前，系统的att, pos, vel作为初始状态，开始IMU+odometer组合导航的递推。
                    self.att[i, :] = self.att0[:, idx]
                    self.pos[i, :] = geoparams.lla2ecef(self.r0[:, idx])
                    self.vel_b[i, :] = self.v0[:, idx]
                    c_bn = attitude.euler2dcm(self.att[i, :]) #c_bn: body frame 到 nav frame的旋转矩阵。
                    self.vel[i, :] = c_bn.T.dot(self.vel_b[i, :])
                    continue
                #### propagate Euler angles
                self.att[i, :] = attitude.euler_update_zyx(self.att[i-1, :], gyro[i-1, :], self.dt)
                # self.att[i, :] = np.zeros((1, 3))
                #### cal vel_b
                gps_vel_N = gps[i-1][3]
                gps_vel_E = gps[i-1][4]
                # gps_vel_D = gps[i-1][5]
                # self.vel_b[i][0] = math.sqrt(pow(gps_vel_N, 2) + pow(gps_vel_E, 2) + pow(gps_vel_D, 2))
                self.vel_b[i][0] = math.sqrt(pow(gps_vel_N, 2) + pow(gps_vel_E, 2))
                self.vel_b[i][1] = 0
                self.vel_b[i][2] = 0

                # c_bn (i)
                c_bn = attitude.euler2dcm(self.att[i, :])
                self.vel[i, :] = c_bn.T.dot(self.vel_b[i, :])   # velocity in navigation frame
                self.pos[i, :] = self.pos[i-1, :] + self.vel[i-1, :] * self.dt
        else:
            pass
        # results
        self.results = [self.att, self.pos, self.vel_b]

    def get_results(self):
        '''
        return algorithm results as specified in self.output
        '''
        return self.results

    def reset(self):
        '''
        Reset the fusion process to uninitialized state.
        '''
        pass
