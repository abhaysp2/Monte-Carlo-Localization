import numpy as np
import turtle
import bisect
import argparse
from scipy.integrate import ode
import rospy
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32MultiArray

class Maze(object):
    def __init__(self, grid_height, grid_width, maze = None, x_start = None, y_start = None, num_rows = None, num_cols = None, wall_prob = None, random_seed = None):
        '''
        maze: 2D numpy array.
        passages are coded as a 4-bit number, with a bit value taking
        0 if there is a wall and 1 if there is no wall.
        The 1s register corresponds with a square's top edge,
        2s register the right edge,
        4s register the bottom edge,
        and 8s register the left edge.
        (numpy array)
        '''
        self.grid_height = grid_height
        self.grid_width = grid_width

        if maze is not None:
            self.maze = maze
            self.num_rows = maze.shape[0]
            self.num_cols = maze.shape[1]
            self.fix_maze_boundary()
            self.fix_wall_inconsistency()
        else:
            assert num_rows is not None and num_cols is not None and wall_prob is not None, 'Parameters for random maze should not be None.'
            self.random_maze(num_rows = num_rows, num_cols = num_cols, wall_prob = wall_prob, random_seed = random_seed)

        self.height = self.num_rows * self.grid_height          # The height of the map
        self.width = self.num_cols * self.grid_width            # The width of the map
        self.x_start = x_start                                  # The starting point of the map with respect to the x position in gazebo
        self.y_start = y_start                                  # The starting point of the map with respect to the y position in gazebo

        self.turtle_registration()

    def turtle_registration(self):
        turtle.register_shape('tri', ((-3, -2), (0, 3), (3, -2), (0, 0)))

    def check_wall_inconsistency(self):
        wall_errors = list()

        # Check vertical walls
        for i in range(self.num_rows):
            for j in range(self.num_cols-1):
                if (self.maze[i,j] & 2 != 0) != (self.maze[i,j+1] & 8 != 0):
                    wall_errors.append(((i,j), 'v'))
        # Check horizonal walls
        for i in range(self.num_rows-1):
            for j in range(self.num_cols):
                if (self.maze[i,j] & 4 != 0) != (self.maze[i+1,j] & 1 != 0):
                    wall_errors.append(((i,j), 'h'))

        return wall_errors

    def fix_wall_inconsistency(self, verbose = True):
        '''
        Whenever there is a wall inconsistency, put a wall there.
        '''
        wall_errors = self.check_wall_inconsistency()

        if wall_errors and verbose:
            print('Warning: maze contains wall inconsistency.')

        for (i,j), error in wall_errors:
            if error == 'v':
                self.maze[i,j] |= 2
                self.maze[i,j+1] |= 8
            elif error == 'h':
                self.maze[i,j] |= 4
                self.maze[i+1,j] |= 1
            else:
                raise Exception('Unknown type of wall inconsistency.')
        return

    def fix_maze_boundary(self):
        '''
        Make sure that the maze is bounded.
        '''
        for i in range(self.num_rows):
            self.maze[i,0] |= 8
            self.maze[i,-1] |= 2
        for j in range(self.num_cols):
            self.maze[0,j] |= 1
            self.maze[-1,j] |= 4

    def random_maze(self, num_rows, num_cols, wall_prob, random_seed = None):
        '''
        Generate a random maze.
        '''
        if random_seed is not None:
            np.random.seed(random_seed)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.maze = np.zeros((num_rows, num_cols), dtype = np.int8)
        for i in range(self.num_rows):
            for j in range(self.num_cols-1):
                if np.random.rand() < wall_prob:
                    self.maze[i,j] |= 2
        for i in range(self.num_rows-1):
            for j in range(self.num_cols):
                if np.random.rand() < wall_prob:
                    self.maze[i,j] |= 4

        self.fix_maze_boundary()
        self.fix_wall_inconsistency(verbose = False)

    def permissibilities(self, cell):
        '''
        Descrption:
            Check if the directions of a given cell are permissible.
        Input:
            cell: a tuple that contain the x, y position to be checked
        Return:
            A tuple of boolean contain if (up, right, down, left) is permissible
        '''
        cell_value = self.maze[cell[0], cell[1]]
        return (cell_value & 1 == 0, cell_value & 2 == 0, cell_value & 4 == 0, cell_value & 8 == 0)

    def distance_to_walls(self, coordinates, sensor_limit):
        '''
        Description:
            Measure the distance of coordinates to nearest walls at four directions.
        Input:
            coordinates: a tuple that contain the x, y position to be checked
            sensor_limit: the sensor limit to be checked
        Return:
            A list of number contain the (up, right, down, left) direction distance to wall
        '''

        x, y = coordinates

        i = int(y // self.grid_height)
        j = int(x // self.grid_width)
        d1 = y - y // self.grid_height * self.grid_height
        while self.permissibilities(cell = (i,j))[0] and d1 < sensor_limit:
            i -= 1
            d1 += self.grid_height

        i = int(y // self.grid_height)
        j = int(x // self.grid_width)
        d2 = self.grid_width - (x - x // self.grid_width * self.grid_width)
        while self.permissibilities(cell = (i,j))[1] and d1 < sensor_limit:
            j += 1
            d2 += self.grid_width

        i = int(y // self.grid_height)
        j = int(x // self.grid_width)
        d3 = self.grid_height - (y - y // self.grid_height * self.grid_height)
        while self.permissibilities(cell = (i,j))[2] and d1 < sensor_limit:
            i += 1
            d3 += self.grid_height

        i = int(y // self.grid_height)
        j = int(x // self.grid_width)
        d4 = x - x // self.grid_width * self.grid_width
        while self.permissibilities(cell = (i,j))[3] and d1 < sensor_limit:
            j -= 1
            d4 += self.grid_width

        return [d1, d2, d3, d4]

    def show_maze(self):
        '''
        Discription:
            Show maze in graphic interface.
        '''

        turtle.setworldcoordinates(0, 0, self.width * 1.005, self.height * 1.005)

        wally = turtle.Turtle()
        wally.speed(0)
        wally.width(1.5)
        wally.hideturtle()
        turtle.tracer(0, 0)

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                permissibilities = self.permissibilities(cell = (i,j))
                turtle.up()
                wally.setposition((j * self.grid_width, i * self.grid_height))
                # Set turtle heading orientation
                # 0 - east, 90 - north, 180 - west, 270 - south
                wally.setheading(0)
                if not permissibilities[0]:
                    wally.down()
                else:
                    wally.up()
                wally.forward(self.grid_width)
                wally.setheading(90)
                wally.up()
                if not permissibilities[1]:
                    wally.down()
                else:
                    wally.up()
                wally.forward(self.grid_height)
                wally.setheading(180)
                wally.up()
                if not permissibilities[2]:
                    wally.down()
                else:
                    wally.up()
                wally.forward(self.grid_width)
                wally.setheading(270)
                wally.up()
                if not permissibilities[3]:
                    wally.down()
                else:
                    wally.up()
                wally.forward(self.grid_height)
                wally.up()

        turtle.update()


    def weight_to_color(self, weight):
        '''
        Discription:
            Assign particles color with respect to their weight
            High weight particle: Red
            Low  weight particle: Blue
        '''
        return '#%02x00%02x' % (int(weight * 255), int((1 - weight) * 255))


    def show_particles(self, particles, show_frequency = 10):

        turtle.shape('tri')

        for i, particle in enumerate(particles):
            if i % show_frequency == 0:
                turtle.setposition((particle.x, particle.y))
                turtle.setheading((particle.heading*180/np.pi)%360)
                turtle.color(self.weight_to_color(particle.weight))
                turtle.stamp()

        turtle.update()

    def show_estimated_location(self, particles):
        '''
        Discription:
            Show average weighted mean location of the particles.
        Input:
            particles: Particle list
        Output:
            a list contain the estimated x, u position of the vehicle
        '''

        x_accum = 0
        y_accum = 0
        heading_accum = 0
        weight_accum = 0

        num_particles = len(particles)

        for particle in particles:

            weight_accum += particle.weight
            x_accum += particle.x * particle.weight
            y_accum += particle.y * particle.weight
            heading_accum += ((particle.heading*180/np.pi)%360) * particle.weight
        if weight_accum == 0:
            return False

        x_estimate = x_accum / weight_accum
        y_estimate = y_accum / weight_accum
        heading_estimate = heading_accum

        turtle.color('orange')
        turtle.setposition(x_estimate, y_estimate)
        turtle.setheading(heading_estimate)
        turtle.shape('turtle')
        turtle.stamp()
        turtle.update()
        return [x_estimate,y_estimate, heading_estimate]

    def show_robot(self, robot):
        '''
        Discription:
            Show actual robot location in graphic interface.
        Input:
            robot: a special particle contains acutal robot location
        '''
        turtle.color('green')
        turtle.shape('turtle')
        turtle.shapesize(0.7, 0.7)
        turtle.setposition((robot.x, robot.y))
        turtle.setheading((robot.heading*180/np.pi)%360)
        turtle.stamp()
        turtle.update()

    def clear_objects(self):
        turtle.clearstamps()

class Particle(object):
    '''
        Particle:
        x, y coordinates, heading and weight are all
        stored in a partcle. Sensor limit is also posed on particle to qualify its readings
    '''
    def __init__(self, x, y, maze, heading = None, weight = 1.0, sensor_limit = None, noisy = False):

        if heading is None:
            heading = np.random.uniform(0,2*np.pi)

        self.x = x                          # The x position of the particle
        self.y = y                          # The y position of the particle
        self.heading = heading              # The orientation of the particle
        self.weight = weight                # The weight of the particle
        self.maze = maze                    # The map the particle have
        self.sensor_limit = sensor_limit    # How far the sensor can see

        # Add noise to control
        if noisy:
            std = max(self.maze.grid_height, self.maze.grid_width) * 0.2
            self.x = self.x+np.random.normal(0,std)
            self.y = self.y+np.random.normal(0,std)
            self.heading = self.heading+np.random.normal(0,np.pi*2*0.05)

        self.fix_invalid_particles()


    def fix_invalid_particles(self):
        # Fix invalid particles
        if self.x < 0:
            self.x = 0
        if self.x > self.maze.width:
            self.x = self.maze.width * 0.9999
        if self.y < 0:
            self.y = 0
        if self.y > self.maze.height:
            self.y = self.maze.height * 0.9999
        self.heading = self.heading % (np.pi*2)

    def read_sensor(self):
        '''
        Description:
            Read the sensor model of a particle
        Return:
            A list of number contain the (up, right, down, left) direction distance to wall
        '''
        readings = self.maze.distance_to_walls(coordinates = (self.x, self.y), sensor_limit = self.sensor_limit)
        # readings[up, right, down, left]
        # add noise to sensor model
        for i in range(len(readings)):
            std = readings[i] * 0.05 / 2+1e-8
            readings[i] = readings[i] + np.random.normal(0, std)

        heading = self.heading*180/np.pi

        # Remove the compass from particle
        if heading >= 45 and heading < 135:
            readings = readings

        elif heading >= 135 and heading < 225:
            readings = readings[-1:] + readings[:-1]
            # readings = [readings[3], readings[0], readings[1], readings[2]]
        elif heading >= 225 and heading < 315:
            readings = readings[-2:] + readings[:-2]
            # readings = [readings[2], readings[3], readings[0], readings[1]]
        else:
            readings = readings[-3:] + readings[:-3]
            # readings = [readings[1], readings[2], readings[3], readings[0]]

        if self.sensor_limit is not None:
            for i in range(len(readings)):
                if readings[i] > self.sensor_limit:
                    readings[i] = self.sensor_limit

        return readings

class Robot(Particle):
    '''
        Robot:
        Robot is a special particle; Unlike regular particle, robot sensor
        retrun readings with noises
    '''
    def __init__(self, x, y, maze, heading = None, sensor_limit = None, noisy = True):
        super(Robot, self).__init__(x = x, y = y, maze = maze, heading = heading, sensor_limit = sensor_limit, noisy = noisy)
        self.noisy = noisy # Controls add/not add noise to sensor reading
        self.modelStateSub = rospy.Subscriber("/gem/model_state", Float32MultiArray, self.__modelStateHandler, queue_size = 1)
        self.modelState = [20,20,0]

    def __modelStateHandler(self,modelState):
        self.modelState = list(modelState.data)

    # def getModelState(self):
    #     rospy.wait_for_service('/gazebo/get_model_state')
    #     try:
    #         serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    #         modelState = serviceResponse(model_name='polaris')
    #     except rospy.ServiceException as exc:
    #         rospy.loginfo("Service did not process request: "+str(exc))
    #     return modelState

    def add_sensor_noise(self, x, z = 0.05):
        '''
        Description:
            Add noise to sensor
        Input:
            x: The reading from the sensor
            z: The amount of noise to be added
        Output:
            The sensor reading with noise
        '''
        readings = list(x)
        for i in range(len(readings)):
            std = readings[i] * z / 2+1e-8
            readings[i] = readings[i] + np.random.normal(0, std)
        return readings

    def read_sensor(self):
        '''
        Description:
            Read the sensor model of the vehicle. It will also update the state of the vehicle
        Return:
            A list of number contain the (up, right, down, left) direction distance to wall
        '''
        # curr_state = self.modelState
        # x = curr_state.pose.position.x
        # y = curr_state.pose.position.y
        # euler = self.quaternion_to_euler(curr_state.pose.orientation.x,
        #                             curr_state.pose.orientation.y,
        #                             curr_state.pose.orientation.z,
        #                             curr_state.pose.orientation.w)
        self.heading = self.modelState[2] % (2*np.pi)
        self.x = (self.modelState[0]+100-self.maze.x_start)*self.maze.grid_width
        self.y = (self.modelState[1]+100-self.maze.y_start)*self.maze.grid_height
        # Robot has error in reading the sensor while particles do not.
        readings = super(Robot, self).read_sensor()
        if self.noisy == True:
            readings = self.add_sensor_noise(x = readings)
        return readings

    def quaternion_to_euler(self, x, y, z, w):
        """
        Description:
            converts quaternion angles to euler angles. Note: Gazebo reports angles in quaternion format
        Inputs:
            x,y,z,w:
                Quaternion orientation values
        Returns:
            List containing the conversion from quaternion to euler [roll, pitch, yaw]
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = np.arcsin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        return [roll, pitch, yaw]
