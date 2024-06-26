#!/usr/bin/env python3

import rospy 
from std_msgs.msg import String

import socket
import codecs

HOST = "0.0.0.0"

def receiveMessage(sock):
    data = sock.recv(8)
    # expect = int.from_bytes(data, 'little', signed=False)
    expect = int(codecs.encode(data, 'hex'), 16)
    print("Waiting for", expect, "bytes")

    received = ""
    while len(received) < expect:
        data = sock.recv(1024)
        received += data.decode()
    # print("Received: " + received)
    print("Received", len(received),"bytes")

    return received


def wait_for_sub(pub, rateHz):
    rate = rospy.Rate(rateHz) # Hz
    print("Current subscribers:",pub.get_num_connections())

    if pub.get_num_connections() == 0:
        print("  Waiting for subscriber...")

    while pub.get_num_connections() == 0:
        rate.sleep()


def run(pub, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    s.bind((HOST, port))
    print("Server bound to " + HOST + ":" + str(port))
    s.listen()
    print("Listening for inbound connections")
    
    conn, addr = s.accept()
    print("Connection established from " + str(addr))

    msg = receiveMessage(conn)
    # print("Received message", msg)

    wait_for_sub(pub, 100) # Hz

    pub.publish(msg)
    print("Published message of length", len(msg))

    s.close()


if __name__=="__main__":
    rospy.init_node('turtlebot_mdp_plan_bridge')

    pub = rospy.Publisher('/mdp/policy', String, queue_size=1)

    port = rospy.get_param('~port')

    run(pub, port)


    