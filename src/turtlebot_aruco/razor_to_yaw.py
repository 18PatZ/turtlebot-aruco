#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64
import tf
import math
import numpy as np


import serial
from threading import Thread


port = '/dev/ttyACM1'
baud = 115200

ser = None


def read_line():
    while ser.inWaiting() <= 0:
        pass

    line = ser.readline().decode()[:-2]

    # print("|",line,"|")
    # print("  ", len(line))
    return line

def read_a_bunch(n):
    for i in range(n):
        read_line()

def read_serial():
    print("Reading from " + ser.name)
    while True:
        line = read_line()
        # print("|",line,"|")

        line = line[line.index("=")+1:]
        yaw, pitch, roll, lx, ly, lz, ax, ay, az = tuple([float(w) for w in line.split(",")])
        print("| YAW:",yaw,"| PITCH:",pitch,"| ROLL:",roll,"|")


def send_command(command):
    ser.write(command.encode())


if __name__=="__main__":
    ser = serial.Serial(port = port, baudrate = baud)
    # read_serial()
    
    rospy.init_node('razor_imu_to_yaw')
    pub = rospy.Publisher('/razor/yaw', Float64, queue_size=1)

    while not rospy.is_shutdown():
        line = read_line()
        # print("|",line,"|")

        line = line[line.index("=")+1:]
        yaw, pitch, roll, lx, ly, lz, ax, ay, az = tuple([float(w) for w in line.split(",")])
        print("| YAW:",yaw,"| PITCH:",pitch,"| ROLL:",roll,"|")
    
        # time.sleep(0.01)

        pub.publish(yaw)

    ser.close()

