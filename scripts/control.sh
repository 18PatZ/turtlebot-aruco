export ROS_MASTER_URI=http://turtlebot1:11311

#rosrun turtlebot_aruco control.py _agent_id:=0 &
sshpass -pturtlebot ssh turtlebot1@turtlebot1 -t "./control.sh"
rosrun turtlebot_aruco planner.py &

export ROS_MASTER_URI=http://turtlebot2:11311

#rosrun turtlebot_aruco control.py _agent_id:=1 &
sshpass -pturtlebot ssh turtlebot2@turtlebot2 -t "./control.sh"
rosrun turtlebot_aruco planner.py &

sleep 25

./start.sh

#trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
