export ROS_MASTER_URI=http://turtlebot1:11311

rosrun turtlebot_aruco start.py _agent_id:=0 &
PID1=$!

export ROS_MASTER_URI=http://turtlebot2:11311

rosrun turtlebot_aruco start.py _agent_id:=1 &
PID2=$!

#echo "PIDS:"
#echo $PID1
#echo $PID2

echo "Sleeping for 3 seconds"
sleep 3

echo "Sending signals..."
kill -s SIGUSR1 $PID1
kill -s SIGUSR1 $PID2

