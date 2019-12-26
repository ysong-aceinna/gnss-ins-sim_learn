# -*- coding: utf-8 -*-
# Filename: algo_test.py

"""
An algorithm for test.
Created on 2019-12-17
@author: Ocean
"""

import numpy as np
import math
from gnss_ins_sim.attitude import attitude
from gnss_ins_sim.geoparams import geoparams

class MahonyTest(object):
    '''
    Mahony filter for calculating (yaw if mag sensor data is available) pitch and roll.
    '''
    def __init__(self):
        '''
        Args:
            ini_pos_vel_att: 9x1 numpy array containing initial position, velocity and attitude.
                3x1 position in the form of LLA, units: [rad, rad, m];
                3x1 velocity in the body frame, units: m/s;
                3x1 Euler anels [yaw, pitch, roll], rotation sequency is ZYX, rad.
            earth_rot: Consider the Earth rotation or not. Only used when ref_frame=0.
        '''
        # algorithm description
        self.input = ['fs', 'gyro', 'accel'] #, 'mag'
        self.output = ['att_quat'] # ['att_euler'] ['att_quat']
        self.results = None
        # algorithm vars
        self.Kp = 0.8  #Kp越大，acc权重越大。
        self.Ki = 0.001
        self.exInt = 0
        self.eyInt = 0
        self.ezInt = 0
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.qs = None
        self.att = None
        
        self.ref_frame = 1
        self.dt = 1.0
        # ini state
        self.set_of_inis = 1        # number of sets of inis
        self.run_times = int(0)     # algo run times. If run_times <= set_of_inis, the i-th run
                                    # uses the i-th set of initial states. Otherwise, the first
                                    # set of initial states will be used
        self.gravity = None

    def run(self, set_of_input):
        '''
        main procedure of the algorithm
        Args:
            set_of_input is a tuple or list consistent with self.input
        '''
        self.run_times += 1
        # get input
        self.dt = 1.0 / set_of_input[0]
        gyro = set_of_input[1]
        accel = set_of_input[2]
        # mag = set_of_input[3]
        n = accel.shape[0]
        self.qs = np.zeros((n, 4))
        self.att = np.zeros((n, 3))
        # update attitude by Mahony filter.

        for i in range(n):
            self.update_vg3(accel[i] ,gyro[i])
            self.att[i] = attitude.quat2euler(self.q) # * 180/math.pi
            self.qs[i] = self.q

        # results
        self.results = [self.qs]
        # self.results = [self.att]

    # ref:https://www.bilibili.com/video/av13035245?from=search&seid=18058484693769908689
    # 误差特别大
    def update_vg1(self, accel, gyro):
        halfT = self.dt/2

        if 0 != accel[0] or 0 != accel[1] or 0 != accel[2]:
            acc_norm = math.sqrt(accel[0]*accel[0] + accel[1]*accel[1] + accel[2]*accel[2])
            ax = accel[0]/acc_norm
            ay = accel[1]/acc_norm
            az = accel[2]/acc_norm

            vx = 2*(self.q[1]*self.q[3] - self.q[0]*self.q[2])
            vy = 2*(self.q[0]*self.q[1] + self.q[2]*self.q[3])
            vz = self.q[0]**2 - self.q[1]**2 - self.q[2]**2 + self.q[3]**2

            ex = ay*vz - az*vy
            ey = az*vx - ax*vz
            ez = ax*vy - ay*vx

            self.exInt += ex * self.Ki
            self.eyInt += ey * self.Ki
            self.ezInt += ez * self.Ki

            gyro[0] += self.Kp*ex + self.exInt
            gyro[1] += self.Kp*ey + self.eyInt
            gyro[2] += self.Kp*ez + self.ezInt

        self.q[0] += (-self.q[1]*gyro[0] - self.q[2]*gyro[1] - self.q[3]*gyro[2]) * halfT
        self.q[1] += ( self.q[0]*gyro[0] + self.q[2]*gyro[2] - self.q[3]*gyro[1]) * halfT
        self.q[2] += ( self.q[0]*gyro[1] - self.q[1]*gyro[2] + self.q[3]*gyro[0]) * halfT
        self.q[3] += ( self.q[0]*gyro[2] + self.q[1]*gyro[1] - self.q[2]*gyro[0]) * halfT

        self.q = attitude.quat_normalize(self.q)

    # update_vg2和update_vg3都是参考官方的代码，但是误差很大。
    # 二者区别是update_vg2没有用twoKp，twoKi,
    # 其实二者本质上是一样的，对同一组数据的结果完全相同
    def update_vg2(self, accel, gyro):
        halfT = self.dt/2

        if 0 != accel[0] or 0 != accel[1] or 0 != accel[2]:
            acc_norm = math.sqrt(accel[0]*accel[0] + accel[1]*accel[1] + accel[2]*accel[2])
            ax = accel[0]/acc_norm
            ay = accel[1]/acc_norm
            az = accel[2]/acc_norm

            vx = 2*(self.q[1]*self.q[3] - self.q[0]*self.q[2])
            vy = 2*(self.q[0]*self.q[1] + self.q[2]*self.q[3])
            vz = self.q[0]**2 - self.q[1]**2 - self.q[2]**2 + self.q[3]**2

            ex = ay*vz - az*vy
            ey = az*vx - ax*vz
            ez = ax*vy - ay*vx

            if(self.Ki > 0): #误差的积分项
                self.exInt += ex * self.Ki * self.dt
                self.eyInt += ey * self.Ki * self.dt
                self.ezInt += ez * self.Ki * self.dt
                gyro[0] += self.exInt
                gyro[1] += self.eyInt
                gyro[2] += self.ezInt
            else:
                self.exInt = 0
                self.eyInt = 0
                self.ezInt = 0

            gyro[0] += self.Kp*ex
            gyro[1] += self.Kp*ey
            gyro[2] += self.Kp*ez

        gyro[0] *= halfT
        gyro[1] *= halfT
        gyro[2] *= halfT

        self.q[0] += (-self.q[1]*gyro[0] - self.q[2]*gyro[1] - self.q[3]*gyro[2]) * 1
        self.q[1] += ( self.q[0]*gyro[0] + self.q[2]*gyro[2] - self.q[3]*gyro[1]) * 1
        self.q[2] += ( self.q[0]*gyro[1] - self.q[1]*gyro[2] + self.q[3]*gyro[0]) * 1
        self.q[3] += ( self.q[0]*gyro[2] + self.q[1]*gyro[1] - self.q[2]*gyro[0]) * 1

        self.q = attitude.quat_normalize(self.q)

    def update_vg3(self, accel, gyro):
        twoKp = 2*self.Kp
        twoKi = 2*self.Ki
        halfT = self.dt/2

        if 0 != accel[0] or 0 != accel[1] or 0 != accel[2]:
            acc_norm = math.sqrt(accel[0]*accel[0] + accel[1]*accel[1] + accel[2]*accel[2])
            ax = accel[0]/acc_norm
            ay = accel[1]/acc_norm
            az = accel[2]/acc_norm

            halfvx = self.q[1]*self.q[3] - self.q[0]*self.q[2]
            halfvy = self.q[0]*self.q[1] + self.q[2]*self.q[3]
            halfvz = 0.5*(self.q[0]**2 - self.q[1]**2 - self.q[2]**2 + self.q[3]**2)

            halfex = ay*halfvz - az*halfvy
            halfey = az*halfvx - ax*halfvz
            halfez = ax*halfvy - ay*halfvx

            if(twoKi > 0):
                self.exInt += twoKi * halfex * self.dt
                self.eyInt += twoKi * halfey * self.dt
                self.ezInt += twoKi * halfez * self.dt
                gyro[0] += self.exInt
                gyro[1] += self.eyInt
                gyro[2] += self.ezInt
            else:
                self.exInt = 0
                self.eyInt = 0
                self.ezInt = 0

            gyro[0] += twoKp*halfex
            gyro[1] += twoKp*halfey
            gyro[2] += twoKp*halfez

        gyro[0] *= halfT
        gyro[1] *= halfT
        gyro[2] *= halfT

        self.q[0] += (-self.q[1]*gyro[0] - self.q[2]*gyro[1] - self.q[3]*gyro[2])
        self.q[1] += ( self.q[0]*gyro[0] + self.q[2]*gyro[2] - self.q[3]*gyro[1])
        self.q[2] += ( self.q[0]*gyro[1] - self.q[1]*gyro[2] + self.q[3]*gyro[0])
        self.q[3] += ( self.q[0]*gyro[2] + self.q[1]*gyro[1] - self.q[2]*gyro[0])

        # self.q = attitude.quat_normalize(self.q)
        q_norm = math.sqrt(self.q[0]*self.q[0] + self.q[1]*self.q[1] + self.q[2]*self.q[2] + self.q[3]*self.q[3])
        self.q = self.q / q_norm

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
