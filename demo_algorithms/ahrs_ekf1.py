# -*- coding: utf-8 -*-
# Filename: ahrs_ekf.py

"""
An algorithm for ahrs ekf test.
Created on 2020-01-11
@author: Ocean

选取状态量：[q0, q1, q2, q3, wb_x, wb_y, wb_z]
选取观测量：[accel]  # 只选accel作为观测量的话还是欠缺了点，因为没有用到mag的信息。但如果把mag也作为观测量，其推导又非常复杂。
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
        self.output = ['att_quat'] #  ['att_quat'];  ['att_euler'] units: ['rad', 'rad', 'rad']
        self.results = None

        self.w_bias = np.ones(3) * 0.0001
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) #上一时刻的attitude

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
        self.qs = np.zeros((n, 4))

        # init start
        init_n = 10 #use init_n accel datas to cal init attitude.
        if (n < init_n): 
            raise OSError('input data is too short!')

        acc_mean = np.mean(accel[0:init_n], axis=0) # cal mean value for acc.
        mag_mean = np.mean(mag[0:init_n], axis=0) # cal mean value for mag.
        euler = self.cal_attitude_by_accel_and_mag(acc_mean, mag_mean) #calculate init attitude by accels and mag
        self.q = attitude.euler2quat(euler)
        print("init yaw:{0:0.3f} ,pitch:{1:0.3f}, roll:{2:0.3f}".format(euler[0]/D2R, euler[1]/D2R, euler[2]/D2R))

        for i in range(init_n): #前init_n个姿态用q来填充。
            self.qs[i] = self.q
    
        P = np.identity(7) * 0.000001 #init P  [7x7]
        R = np.identity(3) * 0.1 #init R  [3x3]
        # init done

        for i in range(init_n, n): # i is from init_n+1 to n.
            # accel normalize.
            accel[i] /= np.linalg.norm(accel[i], ord=2)

            # # pred_states, f shape [6x1]
            # pred_states = self.update_f(gyro[i], self.euler, self.dt)
            # self.update_F(gyro[i], self.euler, self.dt) # shape [6x6]

            # construct f and F.   q是上一时刻的姿态
            omega = self.cal_big_omega_matrix(gyro[i] - self.w_bias) # shape [4x4]
            xi = self.cal_big_xi_matrix(self.q) # shape [4x3]
            f = self.update_f(omega, xi, self.dt) # shape [7x7]
            F = np.copy(f) #这里的状态转移模型是线性的，所以f和F相同。

            #predict states by f and last states.
            last_states = np.hstack((self.q, self.w_bias)) # shape [7x1]
            pred_states = np.dot(f, last_states) # shape [7x1]

            pred_q = attitude.quat_normalize(pred_states[0:4]) #shape [4x1]
            pred_wb = pred_states[4:7] #shape [3x1]

            # construct Q by bi, arw.
            bi = 6/3600  # Bias Instability of IMU381. 6 deg/hr
            arw = 0.3/60 # Angle Random Walk of IMU381. 0.3 deg/sqr(hr)
            Q = self.update_Q(self.q, self.w_bias, self.dt, bi, arw) #shape [7x7]

            #Covariance Estimate.
            P_ = np.dot(np.dot(F, P), F.T) + Q #shape [7x7]

            # update Measurement by predict states.
            h = self.update_h(pred_states) #shape [3x1]
            H = self.update_H(pred_states) #shape [3x7]   有效 [3x4],后边3列补0

            # cal Kalman gain
            S = np.dot(np.dot(H, P_), H.T) + R   #shape [3x3]
            K = np.dot(np.dot(P_, H.T), np.linalg.inv(S)) #shape [7x3]

            # update
            zk = accel[i] #.reshape(3, 1) # accel measurement.
            update_states = pred_states + np.dot(K, (zk - np.squeeze(h))) # update states 
            # print(i, (zk - np.squeeze(h)) )

            self.q = attitude.quat_normalize(update_states[0:4]) #shape [4x1]
            self.w_bias = update_states[4:7] #shape [3x1]
            # print(self.w_bias)
            self.qs[i] = self.q

        # results
        self.results = [self.qs]

    def cal_big_omega_matrix(self, w):
        '''
        Calculat big Omega matrix with angular velocity.
        Args:
            w: angular velocity, rad/s
        Returns:
            big Omega matrix
            __                       __     
            |  0    -w_x  -w_y  -w_z  |
            |  w_x   0     w_z  -w_y  |
            |  w_y  -w_z    0    w_x  |
            |  w_z   w_y  -w_x    0   |
            --                       --
        '''
        omega = np.zeros((4,4))
        omega[0, 1:] = -np.copy(w)
        omega[1:, 0] = np.copy(w.T)
        omega[1:, 1:] = -attitude.get_cross_mtx(w)
        return omega

    def cal_big_xi_matrix(self, q):
        '''
        calculate Ξ matrix.
        Args:
            q: quaternion, scalar first.
        Returns:
            Ξ matrix.

            __               __     
            |  -q1  -q2  -q3  |
            |   q0  -q3   q2  |
            |   q3   q0  -q1  |
            |  -q2   q1   q0  |
            --               --
        '''
        xi = np.zeros((4, 3))
        xi[0, :] = -q[1:]
        xi[1:, 0:] = q[0]*np.identity(3) + attitude.get_cross_mtx(q[1:])
        return xi

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

    def update_f(self, omega, xi, t):
        '''
        update f, the state-transition matrix f(x,u).
        Args:
            omega: omega matrix which is calculated by cal_big_omega_matrix
            xi: xi matrix which is calculated by cal_big_xi_matrix
            t: delta time.
        Returns:
            f, the state-transition matrix.
        '''        
        f = np.zeros((7, 7))
        f[0:4, 0:4] = np.identity(4) + 0.5 * omega * t
        f[0:4, 4:7] = -0.5 * xi * t
        f[4:7, 4:7] = np.identity(3)
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

    def update_h(self, pred_states):
        '''
        update h, the Measurement Model matrix hk.
        Args:
            pred_states: [pred_q, pred_wb] which shape is [7x1]
                         pred_q [4x1], pred_wb [3x1]
        Returns:
            h, the Measurement Model matrix.
        '''
        h = np.zeros((3, 1))
        q = pred_states[0:4]

        h[0] = 2*q[1]*q[3] - 2*q[0]*q[2]
        h[1] = 2*q[0]*q[1] + 2*q[2]*q[3]
        h[2] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3]
        return -h

    def update_H(self, pred_states):
        '''
        Update H, the Jacobian Matrix of Measurement Model matrix hk.
        Args:
            pred_states: [pred_q, pred_wb] which shape is [7x1]
                         pred_q [4x1], pred_wb [3x1]
        Returns:
            H, the Jacobian Matrix of hk.
        '''
        # H = np.zeros((3, 4))
        H = np.zeros((3, 7))
        q = pred_states[0:4]

        H[0][0] = -q[2]
        H[0][1] =  q[3]
        H[0][2] = -q[0]
        H[0][3] =  q[1]
        H[1][0] =  q[1]
        H[1][1] =  q[0]
        H[1][2] =  q[3]
        H[1][3] =  q[2]
        H[2][0] =  q[0]
        H[2][1] = -q[1]
        H[2][2] = -q[2]
        H[2][3] =  q[3]

        return 2*H

    def update_Q(self, q, wb, t, bi, arw):
        '''
        update Q, the covariance of state-transition matrix.
        Args:
            q: quaternion, scalar first.
            wb: gyro bias.
            t: delta time.
            bi: Bias Instability
            arw: Angle Random Walk
        Returns:
            Q.
        '''
        Q = np.zeros((7, 7))
        sigma_q = np.zeros((4, 4))
        sigma_q[0][0] = 1 - q[0]*q[0]
        sigma_q[0][1] = -q[0]*q[1]
        sigma_q[0][2] = -q[0]*q[2]
        sigma_q[0][3] = -q[0]*q[3]
        sigma_q[1][0] = -q[0]*q[1]
        sigma_q[1][1] = 1 - q[1]*q[1]
        sigma_q[1][2] = -q[1]*q[2]
        sigma_q[1][3] = -q[1]*q[3]
        sigma_q[2][0] = -q[0]*q[2]
        sigma_q[2][1] = -q[1]*q[2]
        sigma_q[2][2] = 1 - q[2]*q[2]
        sigma_q[2][3] = -q[2]*q[3]
        sigma_q[3][0] = -q[0]*q[3]
        sigma_q[3][1] = -q[1]*q[3]
        sigma_q[3][2] = -q[2]*q[3]
        sigma_q[3][3] = 1 - q[3]*q[3]
        sigma_q = (0.5*arw*t)*(0.5*arw*t)*sigma_q

        sigma_dd_w = (2*math.pi/math.log(2,math.e)) * (bi*bi/arw)
        sigma_wb = (sigma_dd_w*t)*(sigma_dd_w*t)*np.identity(3)

        Q[0:4, 0:4] = sigma_q
        Q[4:7, 4:7] = sigma_wb

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
