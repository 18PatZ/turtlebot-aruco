export ROS_MASTER_URI=http://turtlebot1:11311
rosrun turtlebot_aruco plan_forwarder.py _port:=6660 &
PID1=$!

export ROS_MASTER_URI=http://turtlebot2:11311
rosrun turtlebot_aruco plan_forwarder.py _port:=6661 &
PID2=$!

sleep 3

rosrun turtlebot_aruco planner.py _port1:=6660 _port2:=6661 _mode:=load

sleep 3

./start.sh

kill -9 $PID1
kill -9 $PID2
