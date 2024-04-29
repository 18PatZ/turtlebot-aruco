#!/usr/bin/env python3

import rospy 
from std_msgs.msg import String, Float64, Bool, Empty, Int32
import signal
import sys

pub = None


def wait_for_sub(pub, rateHz):
    rate = rospy.Rate(rateHz) # Hz
    print(id()+": Current subscribers:",pub.get_num_connections())

    if pub.get_num_connections() == 0:
        print(id()+":   Waiting for subscriber...")

    while pub.get_num_connections() == 0:
        rate.sleep()


def signal_handler(sig, frame):
    print(id()+": Signal received.")

    wait_for_sub(pub, 1000) # Hz

    pub.publish(1)
    print(id()+": Sent start command.")
    sys.exit(0)

def id():
    return "TB"+str(agent_id)

if __name__=="__main__":
    rospy.init_node('turtlebot_mdp_start')

    agent_id = rospy.get_param('~agent_id')

    pub = rospy.Publisher('/control/start', Int32, queue_size=1)
    signal.signal(signal.SIGUSR1, signal_handler)
    print(id()+": Ready, waiting for signal.")
    signal.pause()