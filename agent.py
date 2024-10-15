__author__ = "Johvany Gustave, Jonatan Alvarez"
__copyright__ = "Copyright 2024, IN424, IPSA 2024"
__credits__ = ["Johvany Gustave", "Jonatan Alvarez"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__modified_by__ = "Adrien Cordonnier, Mustapha Koytcha A."

from tf_transformations import euler_from_quaternion
from nav_msgs.msg import Odometry, OccupancyGrid
from rclpy.qos import qos_profile_sensor_data
from math import atan2, cos, sin, pi, sqrt
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
from .frontiers_finder import *
from rclpy.node import Node
from .trajectory import *   # PID controler
from .my_common import *    #common variables are stored here
from .A_star import *
import numpy as np
import rclpy


class Agent(Node):
    """This class is used to define the behavior of ONE agent"""
    
    def __init__(self):
        Node.__init__(self, "Agent")
        self.load_params()

        #initialize attributes
        self.agents_pose = [None] * self.nb_agents    #[(x_1, y_1), (x_2, y_2), (x_3, y_3)] if there are 3 agents
        self.x = self.y = self.yaw = None   #the pose of this specific agent running the node
        self.front_dist = self.left_dist = self.right_dist = 0.0    #range values for each ultrasonic sensor

        self.map_agent_pub = self.create_publisher(OccupancyGrid, f"/{self.ns}/map", 1) #publisher for agent's own map
        self.init_map()

        #Subscribe to agents' pose topic
        odom_methods_cb = [self.odom1_cb, self.odom2_cb, self.odom3_cb]
        for i in range(1, self.nb_agents + 1):  
            self.create_subscription(Odometry, f"/bot_{i}/odom", odom_methods_cb[i-1], 1)
        
        if self.nb_agents != 1: #if other agents are involved subscribe to the merged map topic
            self.create_subscription(OccupancyGrid, "/merged_map", self.merged_map_cb, 1)
        
        #Subscribe to ultrasonic sensor topics for the corresponding agent
        self.create_subscription(Range, f"{self.ns}/us_front/range", self.us_front_cb, qos_profile=qos_profile_sensor_data) #subscribe to the agent's own us front topic to get distance measurements from ultrasonic sensor placed at front of the robot
        self.create_subscription(Range, f"{self.ns}/us_left/range", self.us_left_cb, qos_profile=qos_profile_sensor_data)   #subscribe to the agent's own us front topic to get distance measurements from ultrasonic sensor placed on the left side of the robot
        self.create_subscription(Range, f"{self.ns}/us_right/range", self.us_right_cb, qos_profile=qos_profile_sensor_data) #subscribe to the agent's own us front topic to get distance measurements from ultrasonic sensor placed on the right of the robot
        
        self.cmd_vel_pub = self.create_publisher(Twist, f"{self.ns}/cmd_vel", 1)    #publisher to send velocity commands to the robot

        #Create timers to autonomously call the following methods periodically
        self.create_timer(0.2, self.map_update) #0.2s of period <=> 5 Hz
        self.create_timer(0.5, self.strategy)   #0.5s of period <=> 2 Hz
        self.create_timer(1, self.publish_maps) #1Hz
    
    def load_params(self):
        """Load parameters from launch file"""
        self.declare_parameters(    # A node has to declare ROS parameters before getting their values from launch files
            namespace="",
            parameters=[
                ("ns", rclpy.Parameter.Type.STRING),    #robot's namespace: either 1, 2 or 3
                ("robot_size", rclpy.Parameter.Type.DOUBLE),    #robot's diameter in meter
                ("env_size", rclpy.Parameter.Type.INTEGER_ARRAY),   #environment dimensions (width height)
                ("nb_agents", rclpy.Parameter.Type.INTEGER),    #total number of agents (this agent included) to map the environment
            ])

        #Get launch file parameters related to this node
        self.ns = self.get_parameter("ns").value
        self.robot_size = self.get_parameter("robot_size").value
        self.env_size = self.get_parameter("env_size").value
        self.nb_agents = self.get_parameter("nb_agents").value

        self.pd_controller_theta = Controller(P=0.5, D=0.5)# PID controler
        self.map_data = {"Obstacle":[], "Free":[]}         # For the map
        self.pos_obstacles = []
        self.waypoints = []                                # Store the way points 
        self.need_front = True                             # Flag to check if we need new frontiers        
        self.start = False                                 # To start only
        self.msg = Twist()
        self.compteur = 0                                  # To count when we need to start the strategy
        self.is_blocked = False
        self.var = 0
        
    def init_map(self):
        """Initialize the map to share with others if it is bot_1"""
        self.map_msg = OccupancyGrid()
        self.map_msg.header.frame_id = "map"    #set in which reference frame the map will be expressed (DO NOT TOUCH)
        self.map_msg.header.stamp = self.get_clock().now().to_msg() #get the current ROS time to send the msg
        self.map_msg.info.resolution = self.robot_size  #Map cell size corresponds to robot size
        self.map_msg.info.height = int(self.env_size[0]/self.map_msg.info.resolution)   #nb of rows
        self.map_msg.info.width = int(self.env_size[1]/self.map_msg.info.resolution)    #nb of columns
        self.map_msg.info.origin.position.x = -self.env_size[1]/2   #x and y coordinates of the origin in map reference frame
        self.map_msg.info.origin.position.y = -self.env_size[0]/2
        self.map_msg.info.origin.orientation.w = 1.0    #to have a consistent orientation in quaternion: x=0, y=0, z=0, w=1 for no rotation
        self.map = np.ones(shape=(self.map_msg.info.height, self.map_msg.info.width), dtype=np.int8)*UNEXPLORED_SPACE_VALUE #all the cells are unexplored initially
        self.w, self.h = self.map_msg.info.width, self.map_msg.info.height  

    def publish_maps(self):
        """Publish updated map to topic /bot_x/map, where x is either 1, 2 or 3.
            This method is called periodically (1Hz) by a ROS2 timer, as defined in the constructor of the class."""
        self.map_msg.data = np.flipud(self.map).flatten().tolist()  # transform the 2D array into a list to publish it
        self.map_agent_pub.publish(self.map_msg)                    # publish map to other agents

    def merged_map_cb(self, msg):
        """ Get the current common map and update ours accordingly.
            This method is automatically called whenever a new message is published on the topic /merged_map.
            'msg' is a nav_msgs/msg/OccupancyGrid message."""
        received_map = np.flipud(np.array(msg.data).reshape(self.h, self.w))    #convert the received list into a 2D array and reverse rows
        for i in range(self.h):
            for j in range(self.w):
                if (self.map[i, j] == UNEXPLORED_SPACE_VALUE) and (received_map[i, j] != UNEXPLORED_SPACE_VALUE):
                    self.map[i, j] = received_map[i, j]

    def us_front_cb(self, msg):
        """ Get measurement from the front ultrasonic sensor.
            This method is automatically called whenever a new message is published on topic /bot_x/us_front/range, where 'x' is either 1, 2 or 3.
            'msg' is a sensor_msgs/msg/Range message."""
        self.front_dist = msg.range

    def us_left_cb(self, msg):
        """ Get measurement from the ultrasonic sensor placed on the left.
            This method is automatically called whenever a new message is published on topic /bot_x/us_left/range, where 'x' is either 1, 2 or 3.
            'msg' is a sensor_msgs/msg/Range message."""
        self.left_dist = msg.range

    def us_right_cb(self, msg):
        """ Get measurement from the ultrasonic sensor placed on the right.
            This method is automatically called whenever a new message is published on topic /bot_x/us_right/range, where 'x' is either 1, 2 or 3.
            'msg' is a sensor_msgs/msg/Range message."""
        self.right_dist = msg.range
    
    def odom1_cb(self, msg):
        """Get agent 1 position.
            This method is automatically called whenever a new message is published on topic /bot_1/odom.
            'msg' is a nav_msgs/msg/Odometry message."""
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if int(self.ns[-1]) == 1:
            self.x, self.y = x, y
            self.yaw = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        self.agents_pose[0] = (x, y)
    
    def odom2_cb(self, msg):
        """ Get agent 2 position.
            This method is automatically called whenever a new message is published on topic /bot_2/odom.
            'msg' is a nav_msgs/msg/Odometry message."""
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if int(self.ns[-1]) == 2:
            self.x, self.y = x, y
            self.yaw = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        self.agents_pose[1] = (x, y)

    def odom3_cb(self, msg):
        """ Get agent 3 position.
            This method is automatically called whenever a new message is published on topic /bot_3/odom.
            'msg' is a nav_msgs/msg/Odometry message."""
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if int(self.ns[-1]) == 3:
            self.x, self.y = x, y
            self.yaw = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        self.agents_pose[2] = (x, y)

    def check_border(self):
        if int(self.calcul[0]) == self.map_msg.info.width: # if the value is equal to the borders, we take minus 1
                self.calcul[0] = self.map_msg.info.width-1
        if int(self.calcul[1]) == self.map_msg.info.height:  # if the value is equal to the borders, we take minus 1
            self.calcul[1]= self.map_msg.info.height-1

    def obstc_front(self):
        "This function returns a list of all the points in front of the robot"
        if 0 <= self.front_dist <= 3:                                # if we detect an obstacle
            self.front_dist_inf = np.arange(0, self.front_dist, 0.5) # we do a list from 0 to the distance of the detected obstacle
            self.obstacle_front = True
        else:
            self.front_dist_inf = np.arange(0, 3, 0.5)             # if no obstacle
            self.obstacle_front = False

        # Compute
        self.trans_front = np.array([[1, 0, 0, self.robot_size/2],
                                     [0, 1, 0,                 0],
                                     [0, 0, 1,                 0],
                                     [0, 0, 0,                 1]])

        self.Tr0_front = np.array([[np.cos(self.yaw), -np.sin(self.yaw), 0, self.x],
                                   [np.sin(self.yaw),  np.cos(self.yaw), 0, self.y],
                                   [0,                                0, 1,      0],
                                   [0,                                0, 0,      1]])

        self.front=[]
        self.calcul = []

        for i in range(len(self.front_dist_inf)):
            self.p_front = np.array([self.front_dist_inf[i], 0, 0, 1]).transpose() # searching the matrix for all the points p in front of the robots
            self.front.append(self.Tr0_front @ self.trans_front @ self.p_front)
            self.calcul = ((np.array([[1,  0,  0, -self.map_msg.info.origin.position.x],
                                      [0, -1,  0, -self.map_msg.info.origin.position.y],
                                      [0,  0, -1,                                    0],
                                      [0,  0,  0,                                    1]]) @ self.front[i])/self.map_msg.info.resolution)

            self.check_border()

            if [int(self.calcul[1]), int(self.calcul[0])] not in self.map_data["Obstacle"] or [int(self.calcul[1]), int(self.calcul[0])] not in self.map_data["Free"]:
                if self.obstacle_front:
                    if i==len(self.front_dist_inf)-1:
                        self.map_data["Obstacle"].append([int(self.calcul[1]), int(self.calcul[0])]) # obstacle
                    else:
                        self.map_data["Free"].append([int(self.calcul[1]), int(self.calcul[0])]) # all free element until the obstacle
                else:
                    self.map_data["Free"].append([int(self.calcul[1]), int(self.calcul[0])]) # free element

    def obstc_right(self):
        "This function returns a list of all the points at eh right of the robot"
        if 0 <= self.right_dist <= 3:                                # if we detect an obstacle
            self.right_dist_inf = np.arange(0, self.right_dist, 0.5) # we do a list from 0 to the distance of the detected obstacle
            self.obstacle_right = True
        else:
            self.right_dist_inf = np.arange(0, 3, 0.5)             # if no obstacle
            self.obstacle_right = False
            
        self.trans_right = np.array([[ 0, 1, 0,                  0],
                                     [-1, 0, 0, -self.robot_size/2],
                                     [ 0, 0, 1,                  0],
                                     [ 0, 0, 0,                  1]])
        self.p_right = np.array([self.right_dist, 0, 0, 1]).transpose()
        self.Tr0_right = np.array([[np.cos(self.yaw), -np.sin(self.yaw), 0, self.x],
                                   [np.sin(self.yaw),  np.cos(self.yaw), 0, self.y],
                                   [0,                                0, 1,      0],
                                   [0,                                0, 0,      1]])
        
        self.right=[]
        self.calcul = []

        for i in range(len(self.right_dist_inf)):
            self.p_right = np.array([self.right_dist_inf[i], 0, 0, 1]).transpose()
            self.right.append(self.Tr0_right @ self.trans_right @ self.p_right)
            self.calcul = ((np.array([[1,  0,  0, -self.map_msg.info.origin.position.x],
                                      [0, -1,  0, -self.map_msg.info.origin.position.y],
                                      [0,  0, -1,                                    0],
                                      [0,  0,  0,                                    1]]) @ self.right[i])/self.map_msg.info.resolution)
            
            self.check_border()

            if [int(self.calcul[1]), int(self.calcul[0])] not in self.map_data["Obstacle"] or [int(self.calcul[1]), int(self.calcul[0])] not in self.map_data["Free"]:
                if self.obstacle_right:
                    if i==len(self.right_dist_inf)-1:
                        self.map_data["Obstacle"].append([int(self.calcul[1]), int(self.calcul[0])]) # obstacle
                    else:
                        self.map_data["Free"].append([int(self.calcul[1]), int(self.calcul[0])]) # all free element until the obstacle
                else:
                    self.map_data["Free"].append([int(self.calcul[1]), int(self.calcul[0])]) # free element
    
    def obstc_left(self):
        if 0 <= self.left_dist <= 3:                             # if we detect an obstacle
            self.left_dist_inf = np.arange(0, self.left_dist, 0.5) # we do a list from 0 to the distance of the detected obstacle
            self.obstacle_left = True
        else:
            self.left_dist_inf = np.arange(0, 3, 0.5)            # if no obstacle
            self.obstacle_left = False
            
        self.trans_left = np.array([[0, -1, 0,                 0],
                                    [1,  0, 0, self.robot_size/2],
                                    [0,  0, 1,                 0],
                                    [0,  0, 0,                 1]])
        self.p_left = np.array([self.left_dist, 0, 0, 1]).transpose()
        self.Tr0_left = np.array([[np.cos(self.yaw), -np.sin(self.yaw), 0, self.x],
                                  [np.sin(self.yaw),  np.cos(self.yaw), 0, self.y],
                                  [0,                                0, 1,      0],
                                  [0,                                0, 0,      1]])
        self.left=[]
        self.calcul = []
        self.coordonnée = []

        for i in range(len(self.left_dist_inf)):
            self.p_left = np.array([self.left_dist_inf[i], 0, 0, 1]).transpose()
            self.left.append(self.Tr0_left @ self.trans_left @ self.p_left)
            self.calcul = ((np.array([[1,  0, 0, -self.map_msg.info.origin.position.x],
                                      [0, -1, 0, -self.map_msg.info.origin.position.y],
                                      [0, 0, -1,                                    0],
                                      [0, 0,  0,                                    1]]) @ self.left[i])/self.map_msg.info.resolution)
            
            self.check_border()

            if [int(self.calcul[1]), int(self.calcul[0])] not in self.map_data["Obstacle"] or [int(self.calcul[1]), int(self.calcul[0])] not in self.map_data["Free"]:
                if self.obstacle_left:
                    if i==len(self.left_dist_inf)-1:
                        self.map_data["Obstacle"].append([int(self.calcul[1]), int(self.calcul[0])]) # obstacle
                    else:
                        self.map_data["Free"].append([int(self.calcul[1]), int(self.calcul[0])]) # all free element until the obstacle
                else:
                    self.map_data["Free"].append([int(self.calcul[1]), int(self.calcul[0])]) # free element
            
    def map_update(self):
        """ Consider sensor readings to update the agent's map """

        self.obstc_front()
        self.obstc_right()
        self.obstc_left()

        for coordinate in self.map_data['Obstacle']:
            self.map[coordinate[0], coordinate[1]] = OBSTACLE_VALUE

        for coordinate in self.map_data['Free']:
            self.map[coordinate[0], coordinate[1]] = FREE_SPACE_VALUE

    def robot_position_origin(self):
        "Funtion to get the origin position of a robot"
        self.p_robot = np.array([self.x, self.y, 0, 1]).transpose()
        
        self.coordonnee = np.array(([[1,  0,  0, -self.map_msg.info.origin.position.x],
                                     [0, -1,  0, -self.map_msg.info.origin.position.y],
                                     [0,  0, -1,                                    0],
                                     [0,  0,  0,                                    1]] @ self.p_robot)/self.map_msg.info.resolution)
        self.x_ref_map = self.coordonnee[1]
        self.y_ref_map = self.coordonnee[0]

    def obstacles(self):
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] == 100 and (i, j) not in self.pos_obstacles:
                    self.pos_obstacles.append((i, j))
    
    def strategy(self):
        if not self.start:
            self.msg.angular.z = 0.4
            self.cmd_vel_pub.publish(self.msg)
            self.compteur += 1
            if self.compteur == 20: # should be =20 for completing the analysis of initial position
                self.start = True
                self.msg.angular.z = 0.0
                self.cmd_vel_pub.publish(self.msg)
                self.compteur = 0
        else:
            #self.move_to_right_only() #RANDOM
            self.run()                #Strategy

    def run(self):
        self.robot_position_origin()

        #if self.need_horizon:
        front = detect_frontiers(self.map, (int(self.x_ref_map), int(self.y_ref_map))) # Search all the frontiers
        self.frontiers = median_point_Frontiers(front)                                 # Search median point of new frontiers
        self.robot_assignment = assign_best_frontier((int(self.x_ref_map), int(self.y_ref_map)), self.frontiers) # Assign the closer median point

        for point in self.robot_assignment:
            self.waypoints = astar(self.map, self.pos_obstacles, (int(self.x_ref_map), int(self.y_ref_map)), point)

            if type(self.waypoints) is list:
                pass
            else:
                #self.need_horizon = False
                break

        #else:
        if self.distance(self.waypoints[0]) < 0.1:
            self.waypoints.pop(0)
                
        if self.waypoints != [] :
            #self.go_to_point(self.waypoints[0])
            self.move_to_point(self.waypoints[0])
            #self.need_horizon = True

        else:
            self.msg.linear.x = 0.0
            self.msg.angular.z = 0.0
    
        self.cmd_vel_pub.publish(self.msg)
        
    
    def distance(self, dest):
        return sqrt((dest[0]-self.x_ref_map)**2 + (dest[1]-self.y_ref_map)**2)

    def go_to_point(self, dest):
        """WITHOUT PID"""
        angle_to_dest = atan2(dest[1] - int(self.y_ref_map), dest[0] - int(self.x_ref_map))
        angle_diff = angle_to_dest - self.yaw

        # Adjust angle to be between -pi and pi
        if angle_diff > pi:
            angle_diff -= 2 * pi
        elif angle_diff < -pi:
            angle_diff += 2 * pi

        if abs(angle_diff) <= 0.05:
            if self.distance(dest) < 0.2:
                self.msg.linear.x = 0.0
                self.msg.angular.z = 0.0
            else:
                self.msg.linear.x = 0.2
                self.msg.angular.z = 0.0
        else:
            self.msg.linear.x = 0.0
            self.msg.angular.z = angle_diff / pi

    def move_to_point(self, dest):
        """WITH PID"""
        try:
            # Position error:
            dif_x = dest[0] - self.x_ref_map
            dif_y = dest[1] - self.y_ref_map

            # Orientation error
            phi_d = atan2(dif_y, dif_x)
            phi_err = phi_d - self.yaw

            # Limit the error to (-pi, pi):
            phi_err = atan2(sin(phi_err), cos(phi_err))

            if abs(phi_err) < 0.04:
                self.msg.linear.x = 0.6
                self.msg.angular.z = 0.0
            
            else:
                self.pd_controller_theta.setPoint(phi_d)
                self.msg.linear.x = 0.05
                self.msg.angular.z = self.pd_controller_theta.update(phi_err)

        except Exception as e:
            self.msg.linear.x = 0.0
            self.msg.angular.z = 0.0

    def move_to_right_only(self):
        direction = np.random.randint(0, 40)

        if int(self.ns[-1]) == 1 or int(self.ns[-1]) == 2:
            if self.front_dist < 1.1 and self.compteur < direction or self.var==1:
                self.msg.linear.x = 0.0
                self.msg.angular.z = -0.4
                self.compteur += 1
                self.var=1
                if self.compteur < direction:
                    self.var=0
            else:
                self.msg.linear.x = 0.6
                self.msg.angular.z = 0.0
                self.compteur = 0
                #self.get_logger().info(f"Forward")
            self.cmd_vel_pub.publish(self.msg)
        else:
            if self.front_dist < 1.1 and self.compteur < direction or self.var==1:
                self.msg.linear.x = 0.0
                self.msg.angular.z = 0.4
                self.compteur += 1
                self.var=1
                if self.compteur < direction:
                    self.var=0
            else:
                self.msg.linear.x = 0.6
                self.msg.angular.z = 0.0
                self.compteur = 0
                #self.get_logger().info(f"Forward")
            self.cmd_vel_pub.publish(self.msg)

def main():
    rclpy.init()

    node = Agent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()