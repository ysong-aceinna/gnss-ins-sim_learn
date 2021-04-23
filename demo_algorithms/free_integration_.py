# -*- coding: utf-8 -*-
# Filename: free_integration_Test.py

"""
IMU free integration for test.
Created on 2020-11-19
"""
import sys
sys.path.append("./")
import math
import numpy as np
import scipy.io as io
from gnss_ins_sim.attitude import attitude


D2R = math.pi/180.0

class FreeIntegration(object):
    '''
    Integrate gyro to get attitude, double integrate linear acceleration to get position.

    1. 根据Gyro递推EulerAngles，根据Accel和姿态递推Vel和Pos
    2. 初始条件: ODR，初始姿态、速度和位置
    3. 输入csv格式: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, roll, pitch, yaw     unit:[g],[dps],[deg]

    !使用注意：
        1. 注意csv中数据的顺序和单位。如果不是[g,dps,dgree], 需要在 run 中转换单位。
        2. 根据csv中有无表头来调节 skiprows.
        3. 在__init__中配置相关参数.
        4. 输出的att次序是ypr.
    '''
    def __init__(self, csvFile):
        '''
        '''
        self.gravity = 9.806
        self.dt = 1/100;     # 需要根据测试数据配置ODR
        self.csvFile = csvFile

    def run(self):
        '''
        '''
        data = np.loadtxt(open(self.csvFile),delimiter=",",skiprows=1) 

        # get input
        gyro = data[:,3:6]  * D2R # 注意：需要将csv中gyro和rpy的单位转换为 [rad/s],[rad]
        accel = data[:,0:3]
        n = accel.shape[0]

        # Free IMU integration
        att = np.zeros((n, 3))
        pos = np.zeros((n, 3))
        vel = np.zeros((n, 3))     # NED vel
        vel_b = np.zeros((n, 3))   # body vel
        c_bn = np.eye((3))

        # initial posisiton, velocity and attitude.
        r0 = np.zeros((1,3))
        v0 = np.zeros((1,3))
        # att0 = data[0,15:18][::-1] # csv中的顺序是rpy, 反转成ypr.
        # att0 = att0.T * D2R

        # Earth gravity
        g_n = np.array([0, 0, self.gravity])
        start_idx = 0
        for i in range(n):
        # for i in range(start_idx, start_idx+250):
            #### initialize
            if i == start_idx:
                att0 = data[start_idx,6:9][::-1] # csv中的顺序是rpy, 反转成ypr.
                att0 = att0.T * D2R

                att[i, :] = att0
                pos[i, :] = r0
                vel_b[i, :] = v0
                c_bn = attitude.euler2dcm(att[i, :])
                vel[i, :] = c_bn.T.dot(vel_b[i, :])
                continue
            #### propagate Euler angles
            att[i, :] = attitude.euler_update_zyx(att[i-1, :], gyro[i-1, :], self.dt)
            #### propagate velocity in the navigation frame
            # accel_n = c_nb.dot(accel[i-1, :])
            # vel[i, :] = vel[i-1, :] + (accel_n + g_n) * self.dt
            #### propagate velocity in the body frame
            # c_bn here is from last loop (i-1), and used to project gravity
            vel_b[i, :] = vel_b[i-1, :] +\
                            (accel[i-1, :] + c_bn.dot(g_n)) * self.dt -\
                            attitude.cross3(gyro[i-1, :], vel_b[i-1, :]) * self.dt
            # c_bn (i)
            c_bn = attitude.euler2dcm(att[i, :])
            vel[i, :] = c_bn.T.dot(vel_b[i, :])   # velocity in navigation frame
            pos[i, :] = pos[i-1, :] + vel[i-1, :] * self.dt

        self.results = [att, pos, vel]

        # save to mat.
        tmp = {}
        tmp["gyro"] = gyro / D2R;    # [dps]
        tmp["accel"] = accel;        # [g] 
        tmp["att"] = att / D2R;      # [deg]
        tmp["pos"] = pos;            # [m]
        tmp["vel"] = vel;            # [m/s]
        mat_path = '/Users/songyang/project/analyze/drive_test/2020-11-30/analyze/a.mat'
        io.savemat(mat_path, tmp)

        pass


if __name__ == '__main__':
    csvFile = '/Users/songyang/project/analyze/drive_test/2020-11-30/analyze/335.csv'
    # csvFile = '/Users/songyang/project/analyze/drive_test/2020-11-30/analyze/span.csv'
    fi = FreeIntegration(csvFile)
    fi.run()
    pass
