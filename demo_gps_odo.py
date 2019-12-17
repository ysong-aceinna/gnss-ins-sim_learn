# -*- coding: utf-8 -*-
# Filename: demo_free_integration.py

"""
odometer simulation
Created on 2019-12-17
@author: Ocean
"""

"""
odometer simulation

仿真内容：
在GPS信号丢失，只有IMU和odometer（速度里程计）正常工作时的位置、速度和姿态推算。

输入：
1. 实际应用中，使用GPS信号丢失时的position,velocity,attitude作为初始状态。
本次仿真使用csv中的初始状态。
2. odometer的实时速度（本程序用基于GPS实时输出的sqrt(Vn^2+Ve^2)模拟odometer的速度输出）

输出：
实时的:
1. position: NED, LLA表示, [rad,rad,m]
2. velocity: NED, [m/s, m/s, m/s]
3. attitude

"""

import os
import math
import numpy as np
from gnss_ins_sim.sim import imu_model
from gnss_ins_sim.sim import ins_sim

# globals
D2R = math.pi/180

motion_def_path = os.path.abspath('.//demo_motion_def_files//')
fs = 100.0          # IMU sample frequency

def odometer_sim():
    #### IMU model, typical for IMU381
    imu_err = {'gyro_b': np.array([0.0, 0.0, 0.0]),
               'gyro_arw': np.array([0.25, 0.25, 0.25]) * 1.0,
               'gyro_b_stability': np.array([3.5, 3.5, 3.5]) * 1.0,
               'gyro_b_corr': np.array([100.0, 100.0, 100.0]),
               'accel_b': np.array([0.0e-3, 0.0e-3, 0.0e-3]),
               'accel_vrw': np.array([0.03119, 0.03009, 0.04779]) * 1.0,
               'accel_b_stability': np.array([4.29e-5, 5.72e-5, 8.02e-5]) * 1.0,
               'accel_b_corr': np.array([200.0, 200.0, 200.0]),
               'mag_std': np.array([0.2, 0.2, 0.2]) * 1.0
              }
    gps_err = {'stdp': np.array([5.0, 5.0, 7.0]) * 1.0,
               'stdv': np.array([0.05, 0.05, 0.05]) * 1.0}
    # generate GPS and magnetometer data
    imu = imu_model.IMU(accuracy=imu_err, axis=6, gps=True, gps_opt=gps_err)

    #### Algorithm
    # Free integration based on odometer in NED frame
    from demo_algorithms import gps_odo
    '''
    Initial states (position, velocity and attitude) are from motion_def csv file.
    '''
    ini_pos_vel_att = np.genfromtxt(motion_def_path+"//motion_def-ins_odo.csv",\
                                    delimiter=',', skip_header=1, max_rows=1)
    ini_pos_vel_att[0] = ini_pos_vel_att[0] * D2R   #将lat (deg),lon (deg), yaw (deg), pitch (deg), roll (deg)单位转为弧度 rad
    ini_pos_vel_att[1] = ini_pos_vel_att[1] * D2R
    ini_pos_vel_att[6:9] = ini_pos_vel_att[6:9] * D2R
    # add initial states error if needed
    ini_vel_err = np.array([0.0, 0.0, 0.0]) # initial velocity error in the body frame, m/s
    ini_att_err = np.array([0.0, 0.0, 0.0]) # initial Euler angles error, deg
    ini_pos_vel_att[3:6] += ini_vel_err
    ini_pos_vel_att[6:9] += ini_att_err * D2R
    # create the algorith object
    algo = gps_odo.OdometerSim(ini_pos_vel_att)

    #### start simulation
    sim = ins_sim.Sim([fs, fs, 0.0], # imu, gps, no mag.
                      motion_def_path+"//motion_def-ins_odo.csv", #  motion_def-90deg_turn_gps    motion_def-ins_odo
                      ref_frame=1,
                      imu=imu,
                      mode=None,
                      env=None,
                      algorithm=algo)
    # run the simulation for 1 times
    sim.run(1)
    # generate simulation results, summary
    # save results
    sim.results(err_stats_start=-1, gen_kml=True)
    sim.plot(['ref_pos', 'gyro', 'gps_visibility', 'accel'], opt={'ref_pos': '3d'})

if __name__ == '__main__':
    odometer_sim()
