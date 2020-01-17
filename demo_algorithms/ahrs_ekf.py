# -*- coding: utf-8 -*-
# Filename: ahrs_ekf.py

"""
An algorithm for ahrs ekf test.
Created on 2020-01-11
@author: Ocean

选取状态量：[yaw, pitch, roll, wb_x, wb_y, wb_z]
选取观测量：[yaw, pitch, roll]
"""

import numpy as np
import math
from gnss_ins_sim.attitude import attitude
from gnss_ins_sim.geoparams import geoparams

class AHRSEKFTest(object):
    '''
    calculate yar, pitch, roll by EKF.
    假设磁偏角为0.
    '''
    def __init__(self):
        '''
        Args:
        '''
        # algorithm description
        self.input = ['fs', 'gyro', 'accel', 'mag']
        self.output = ['att_euler'] #  ['att_quat'];  ['att_euler'] units: ['rad', 'rad', 'rad']
        self.results = None

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
                fs: sample frequency, Hz
                gyro: numpy array of size (n,3), rad/s
                accel: numpy array of size (n,3), m/s/s
                mag: numpy array of size (n,3), μT
        '''
        D2R = math.pi/180
        self.run_times += 1
        # get input
        self.dt = 1.0 / set_of_input[0]
        gyro = set_of_input[1] # rad/sec
        accel = set_of_input[2] # m/s^2
        mag = set_of_input[3] # μT  # 1Gauss = 100μT,  1μT = 0.01Gauss 

        n = accel.shape[0]
        self.att = np.zeros((n, 3))
        self.w_bias = np.ones(3) * 0.000001

        # init start
        init_n = 10 #use init_n accel datas to cal init attitude.
        if (n < init_n): 
            raise OSError('input data is too short!')

        acc_mean = np.mean(accel[0:init_n], axis=0) # cal mean value for acc.
        mag_mean = np.mean(mag[0:init_n], axis=0) # cal mean value for mag.
        self.euler = self.cal_attitude_by_accel_and_mag(acc_mean, mag_mean) #calculate init attitude by accels and mag
        print("init yaw:{0:0.3f} ,pitch:{1:0.3f}, roll:{2:0.3f}".format(self.euler[0]/D2R, self.euler[1]/D2R, self.euler[2]/D2R))

        # for i in range(n): # 因为感觉生成的mag数据有问题，所以结合accel看以下全程的yaw是否和motion file对应。下边得到的headin是对的。
        #     euler = self.cal_attitude_by_accel_and_mag(accel[i], mag[i]) #calculate init attitude by accels and mag
        #     euler = euler / D2R
        #     print("yaw:{0:0.3f} ,pitch:{1:0.3f}, roll:{2:0.3f}".format(euler[0], euler[1], euler[2]))

        for i in range(init_n): #前init_n个姿态用q来填充。
            self.att[i] = self.euler
    
        P = np.identity(6) * 0.0003 #init P  [6x6], 假设角度误差为1 deg, 即 0.01745 rad, 平方-> 0.0003
        R = np.identity(3) * 0.0003 #init R  [3x3]
        # init done

        for i in range(init_n, n): # i is from init_n+1 to n.
            # accel normalize.
            accel[i] /= np.linalg.norm(accel[i], ord=2)

            # pred_states, f shape [6x1]
            pred_states = self.update_f(gyro[i], self.euler, self.dt)
            F = self.update_F(gyro[i], self.euler, self.dt) # shape [6x6]f

            pred_euler = pred_states[0:4] #shape [3x1]
            pred_wb = pred_states[4:7] #shape [3x1]

            bi = 6/3600  # Bias Instability of IMU381. 6 deg/hr
            arw = 0.3/60 # Angle Random Walk of IMU381. 0.3 deg/sqr(hr)
            Q = self.update_Q(self.euler, self.dt, bi, arw) #shape [6x6]

            #Covariance Estimate.
            # P_ = F * P * F.T + Q #shape [6x6]
            P_ = np.dot(np.dot(F, P), F.T) + Q #shape [6x6]
            # update Measurement by predict states.
            # 因为选取[yaw, pitch, roll]作为测量值，则观测矩阵H就是单位阵。
            H = np.zeros((3, 6))#shape [3x6], 后边3列对wb求偏导的部分为0
            H[0:3, 0:3] = np.identity(3)
            hk = np.dot(H, pred_states)

            # h = self.update_h(pred_states, accel[i], mag[i]) #shape [3x1]
            # H = self.update_H(pred_states, mag[i]) #shape [3x6], 后边3列对wb求偏导的部分为0

            # cal Kalman gain
            S = np.dot(np.dot(H, P_), H.T) + R   #shape [6x6]
            K = np.dot(np.dot(P_, H.T), np.linalg.inv(S)) #shape [6x3]
            # K = np.zeros((6, 3))

            # update
            # zk = np.zeros((6,1)) # measurement from sensor. [6x1]
            # zk[0:3] = accel[i].reshape(3,1)   # accel measurement from accel sensor. [3x1]
            # zk[3:6] = mag[i].reshape(3,1)     # mag measurement from mag sensor. [3x1]
            zk = (self.cal_attitude_by_accel_and_mag(accel[i], mag[i])).reshape(3, 1)
            update_states = pred_states + np.dot(K, (zk - hk)) # update states 

            # self.euler 的 shape 有问题，再调下
            self.euler = np.squeeze(update_states[0:3]) #shape [3x1]
            self.w_bias = np.squeeze(update_states[3:6]) #shape [3x1]
            self.att[i] = self.euler
            # print(self.w_bias)
            
        # results
        self.results = [self.att]

    def cal_pitch_roll_by_accel(self, acc):
        '''
        Calculate pitch and roll by given acc.
        Args:
            acc: acc measurement, numpy array of 3x1.
        Returns:
            [pitch, roll]
        '''
        a_norm = np.linalg.norm(acc,ord=2) #Normalize 
        pitch = math.asin(acc[0]/a_norm)
        roll = math.atan2(-acc[1], -acc[2])

        return np.array([pitch, roll])

    def cal_attitude_by_accel_and_mag(self, acc, mag):
        '''
        Calculate pitch and roll by given acc data.
        Args:
            acc: acc measurement, numpy array of 3x1.
            mag: mag measurement, numpy array of 3x1.
        Returns:
            euler: [yaw, pitch, roll], unit rad/s
        '''
        [pitch, roll] = self.cal_pitch_roll_by_accel(acc)

        s_pitch = math.sin(pitch)
        c_pitch = math.cos(pitch)
        s_roll = math.sin(roll)
        c_roll = math.cos(roll)

        mag = mag / math.sqrt(np.dot(mag, mag))
        hx = mag[0]*c_pitch + mag[1]*s_pitch*s_roll + mag[2]*s_pitch*c_roll
        hy = mag[1]*c_roll - mag[2]*s_roll
        yaw = math.atan2(-hy, hx)

        return np.array([yaw, pitch, roll])

    def update_f(self, w, euler, dt):
        '''
        update f, the state-transition matrix f(x,u).
        Args:
            euler: [yaw, pitch, roll], rad
            w: angular velocity, rad/s
            dt: delta time.
        Returns:
            f, the state-transition matrix.
        '''
        [wx, wy, wz] = w - self.w_bias

        psi   = euler[0] # yaw   at (t-1)
        theta = euler[1] # pitch at (t-1)
        phi   = euler[2] # roll  at (t-1)

        c_theta = math.cos(theta)
        tan_theta = math.tan(theta)
        s_phi = math.sin(phi)
        c_phi = math.cos(phi)

        f = np.zeros((6, 1))
        f[0] = psi + (wz * c_phi + wy * s_phi) * dt / c_theta
        f[1] = theta + (wy * c_phi - wz * s_phi) * dt
        f[2] = phi + (wx + (wz * c_phi + wy * s_phi) * tan_theta) * dt
        f[3:6] = self.w_bias.reshape(3,1)

        return f

    def update_F(self, w, euler, dt):
        '''
        Update F, the Jacobian Matrix of state-transition matrix f.
        Args:
            euler: [yaw, pitch, roll], rad
            w: angular velocity, rad/s
            dt: delta time.
        Returns:
            F, the Jacobian Matrix of state-transition matrix f.
        '''
        [wx, wy, wz] = w - self.w_bias

        psi   = euler[0] # yaw   at (t-1)
        theta = euler[1] # pitch at (t-1)
        phi   = euler[2] # roll  at (t-1)

        s_theta = math.sin(theta)
        c_theta = math.cos(theta)
        tan_theta = math.tan(theta)
        s_phi = math.sin(phi)
        c_phi = math.cos(phi)

        F = np.zeros((6, 6))
        F[0][0] = 1
        F[0][1] = -(wz * c_phi + wy * s_phi) / c_theta / tan_theta * dt
        F[0][2] = (-wz * s_phi + wy * c_phi) / c_theta * dt
        F[0][3] = 0
        F[0][4] = -s_phi / c_theta * dt
        F[0][5] = -c_phi / c_theta * dt
        F[1][0] = 0
        F[1][1] = 1
        F[1][2] = (-wy * s_phi - wz * c_phi) * dt
        F[1][3] = 0
        F[1][4] = -c_phi * dt
        F[1][5] = s_phi * dt
        F[2][0] = 0
        F[2][1] = (wz * c_phi + wy * s_phi) / s_theta / s_theta * dt
        F[2][2] = 1 + (wy * c_phi - wz * s_phi) * tan_theta * dt
        F[2][3] = -dt
        F[2][4] = -s_phi * tan_theta * dt
        F[2][5] = -c_phi * tan_theta * dt
        F[3:6, 3:6] = np.identity(3)
        return F

    # def update_h(self, pred_states, acc, mag):
    #     '''
    #     update h, the Measurement Model matrix hk.
    #     Args:
    #         pred_states: [pred_euler, pred_wb] which shape is [6x1]
    #                      pred_euler [3x1], pred_wb [3x1]
    #         acc: acc measurement, numpy array of 3x1.
    #         mag: mag sensor data which in body frame and have been calibrated, numpy array of 3x1.
    #     Returns:
    #         h, the Measurement Model matrix.
    #     '''
    #     h = self.cal_attitude_by_accel_and_mag(acc, mag)
    #     return h

    # def update_H(self, pred_states, mag):
    #     '''
    #     Update H, the Jacobian Matrix of Measurement Model matrix hk.
    #     Args:
    #         pred_states: [pred_euler, pred_wb] which shape is [6x1]
    #                      pred_euler [3x1], pred_wb [3x1]
    #         mag: mag sensor data which in body frame and have been calibrated, numpy array of 3x1.
    #     Returns:
    #         H, the Jacobian Matrix of hk.
    #     '''
    #     euler = pred_states[0:3]
    #     psi   = euler[0]
    #     theta = euler[1]
    #     phi   = euler[2]
    #     s_theta = math.sin(theta)
    #     c_theta = math.cos(theta)
    #     s_phi = math.sin(phi)
    #     c_phi = math.cos(phi)

    #     [mx, my, mz] = mag / math.sqrt(np.dot(mag, mag))

    #     hx = mx * c_theta + my * s_theta * s_phi + mz * s_theta * c_phi
    #     hy = my * c_phi - mz * s_phi
    #     A = -hy/hx # yaw = math.atan2(-hy, hx)

    #     H = np.zeros((3, 6))
    #     H[0][0] = 0
    #     H[0][1] = (1/(1+A*A)) * (hy * (-mx * s_theta + my * c_theta * s_phi + mz * c_theta * c_phi)) / (hx * hx)
    #     H[0][2] = (1/(1+A*A)) * ((my * s_phi + mz * c_phi) * hx + hy * (my * s_theta * c_phi + mz * s_theta * c_phi)) / (hx * hx)

    #     return H

    def update_Q(self, euler, dt, bi, arw):
        '''
        update Q, the covariance of state-transition matrix.
        Args:
            q: quaternion, scalar first.
            wb: gyro bias.
            dt: delta time.
            bi: Bias Instability
            arw: Angle Random Walk, deg/sec
        Returns:
            Q.
        '''
        D2R = math.pi/180
        wn_x = arw * D2R
        wn_y = wn_x
        wn_z = wn_x

        psi   = euler[0] # yaw   at (t-1)
        theta = euler[1] # pitch at (t-1)
        phi   = euler[2] # roll  at (t-1)

        c_theta = math.cos(theta)
        tan_theta = math.tan(theta)
        s_phi = math.sin(phi)
        c_phi = math.cos(phi)

        n_psi   = -(wn_z * c_phi + wn_y * s_phi) / c_theta * dt
        n_theta = -(wn_y * c_phi - wn_z * s_phi) * dt
        n_phi   = -(wn_x + (wn_z * c_phi + wn_y * s_phi) * tan_theta)*dt

        Q = np.zeros((6, 6))
        Q[0][0] = n_psi * n_psi
        Q[1][1] = n_theta * n_theta
        Q[2][2] = n_phi * n_phi

        sigma_dd_w = (2*math.pi/math.log(2,math.e)) * (bi*bi/arw)
        sigma_wb = (sigma_dd_w*dt)*(sigma_dd_w*dt)*np.identity(3)
        Q[3:6, 3:6] = sigma_wb

        # Q = np.ones((7, 7)) * 0.0001
        return Q

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
