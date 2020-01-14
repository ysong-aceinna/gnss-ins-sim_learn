# -*- coding: utf-8 -*-
# Filename: vg_ekf1.py

"""
An algorithm for vg ekf test.
Created on 2019-12-17
@author: Ocean

和vg_ekf.py的不同点，仅仅是h和H的不同。从测试效果看，误差比vg_ekf.py小。
vg_ekf.py：使用q推导h和H。
vg_ekf1.py：使用pitch和roll推导h和H。
"""

import numpy as np
import math
from gnss_ins_sim.attitude import attitude
from gnss_ins_sim.geoparams import geoparams

class VGEKFTest(object):
    '''
    calculate pitch, roll by EKF.
    '''
    def __init__(self):
        '''
        Args:
        '''
        # algorithm description
        self.input = ['fs', 'gyro', 'accel'] #, 'mag'
        self.output = ['att_quat'] # ['att_euler'] ['att_quat']
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
        '''
        D2R = math.pi/180
        self.run_times += 1
        # get input
        self.dt = 1.0 / set_of_input[0]
        gyro = set_of_input[1] # rad/sec
        accel = set_of_input[2] # m/s^2

        n = accel.shape[0]
        self.qs = np.zeros((n, 4))
        # self.att = np.zeros((n, 3))

        # init start
        init_n = 10 #use init_n accel datas to cal init attitude.
        if (n < init_n): 
            raise OSError('input data is too short!')
        
        acc_mean = np.mean(accel[0:init_n], axis=0) # cal mean value for acc.
        self.q = self.cal_attitude_by_accel(acc_mean) #calculate init attitude by accels
        euler = attitude.quat2euler(self.q) / D2R
        print("init yaw:{0:0.3f} ,pitch:{1:0.3f}, roll:{2:0.3f}".format(euler[0], euler[1], euler[2]))

        for i in range(init_n): #前init_n个姿态用q来填充。
            self.qs[i] = self.q
    
        P = np.identity(7) * 0.000001 #init P 
        R = np.identity(3) * 0.1 #init R 
        # init done

        for i in range(init_n, n): # i is from init_n+1 to n.
            # accel normalize.
            acc_norm = np.linalg.norm(accel[i], ord=2)
            accel[i] /= acc_norm
            # accel[i] /= 9.8

            # construct f and F.   q是上一时刻的姿态
            omega = self.cal_big_omega_matrix(gyro[i] - self.w_bias) # shape [4x4]
            xi = self.cal_big_xi_matrix(self.q) # shape [4x3]
            f = self.update_f(omega, xi, self.dt) # shape [7x7]
            F = np.copy(f) #这里的状态转移模型是线性的，所以f和F相同。# shape [7x7]

            #predict states by f and last states.
            last_states = np.hstack((self.q, self.w_bias)) # shape [7x1]
            pred_states = np.dot(f, last_states) # shape [7x1]

            pred_q = attitude.quat_normalize(pred_states[0:4]) #shape [4x1]
            pred_wb = pred_states[4:7] #shape [3x1]
            # self.q = pred_q
            # Q 应该根据k-1时刻的q和wb估计呢，还是根据刚刚得到的q和wb的预测值来估计呢？
            # 应该区别不大，因为数值都非常的小。
            bi = 6/3600  # Bias Instability of IMU381. 6 deg/hr
            arw = 0.3/60 # Angle Random Walk of IMU381. 0.3 deg/sqr(hr)
            Q = self.update_Q(self.q, self.w_bias, self.dt, bi, arw) #shape [7x7]
            # Q = self.update_Q(pred_q, pred_wb, self.dt, bi, arw)

            #Covariance Estimate.
            # P_ = F * P * F.T + Q; #shape [7x7]
            P_ = np.dot(np.dot(F, P), F.T) + Q #shape [6x6]

            # update Measurement by predict states.
            h = self.update_h(pred_states) #shape [3x1]
            H = self.update_H(pred_states) #shape [3x7]   有效 [3x4],后边3列补0

            # cal Kalman gain
            S = np.dot(np.dot(H, P_), H.T) + R   #shape [3x3]
            K = np.dot(np.dot(P_, H.T), np.linalg.inv(S)) #shape [7x3]

            z = accel[i] #.reshape(3, 1) # accel measurement.
            update_states = pred_states + np.dot(K, (z - np.squeeze(h))) # update states 

            self.q = attitude.quat_normalize(update_states[0:4]) #shape [4x1]
            self.w_bias = update_states[4:7] #shape [3x1]
            # print(self.w_bias)
            # euler = attitude.quat2euler(self.q) / D2R
            # print("pred yaw:{0:0.3f} ,pitch:{1:0.3f}, roll:{2:0.3f}".format(euler[0], euler[1], euler[2]))

            self.qs[i] = self.q

        # results
        self.results = [self.qs]
        # self.results = [self.att]

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

    def cal_attitude_by_accel(self, acc):
        '''
        Calculate pitch and roll by given acc data.
        Args:
            acc: acc measurement, numpy array of 3x1.

        Returns:
            q: quaternion, [q0, q1, q2, q3], q0 is the scalar
        '''
        a_norm = np.linalg.norm(acc,ord=2) #Normalize 
        pitch = math.asin(acc[0]/a_norm)
        roll = math.atan2(-acc[1], -acc[2])
        q = attitude.euler2quat(np.array([0, pitch, roll]))
        return q

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

    # def update_F(self, omega, xi, t):
    #     '''
    #     Update F, the Jacobian Matrix of state-transition matrix f(x,u).
    #     Args:
    #         omega: omega matrix which is calculated by cal_big_omega_matrix
    #         xi: xi matrix which is calculated by cal_big_xi_matrix
    #     Returns:
    #         F, the Jacobian Matrix.
    #     '''
    #     F = np.zeros((7, 7))
    #     F[0:4, 0:4] = np.identity(4) + 0.5 * omega * t
    #     F[0:4, 4:7] = -0.5 * xi * t
    #     F[4:7, 4:7] = np.identity(3)

    #     return F

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

        [yaw, pitch, roll] = attitude.quat2euler(q)

        h[0] = -math.sin(pitch)
        h[1] = math.sin(roll)*math.cos(pitch)
        h[2] = math.cos(roll)*math.cos(pitch)
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
        [yaw, pitch, roll] = attitude.quat2euler(q)

        H[0][0] = math.cos(pitch)
        H[1][0] = math.sin(pitch)*math.sin(roll)
        H[1][1] = -math.cos(pitch)*math.cos(roll)
        H[2][0] = math.sin(pitch)*math.cos(roll)
        H[2][1] = math.sin(roll)*math.cos(pitch)

        return H

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


    def test(self):


        # ax = 1.7
        # ay = 0.001
        # az = -9.67
        ax = -0.9
        ay = 5.3
        az = 9.78
        nor = math.sqrt(ax*ax + ay*ay + az*az)
        ax /= nor
        ay /= nor
        az /= nor
        D2R = math.pi/180

        #常规方式，根据DCM求解。
        pitch = math.asin(ax) / D2R
        roll = math.atan2(-ay, -az) /D2R

        #根据几何关系求解。从国内的论文中看到的这个公式，其实并不好用。
        # pitch = math.atan2(ax, math.sqrt(ay*ay + az*az)) / D2R
        # roll = math.atan2(-ay, -math.sqrt(ax*ax + az*az)) /D2R

        print("pitch:{0:0.3f}, roll:{1:0.3f}".format(pitch, roll))
        pass