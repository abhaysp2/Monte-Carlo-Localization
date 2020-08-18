import numpy as np
import random
from maze import Maze, Particle, Robot
import bisect
from bisect import bisect_left
import rospy
from gazebo_msgs.msg import  ModelState
from gazebo_msgs.srv import GetModelState
import shutil
from std_msgs.msg import Float32MultiArray
from scipy.integrate import ode
import copy
import math
import matplotlib.pyplot as plt
import pickle

def func1(t, vars, vr, delta):
    curr_x = vars[0]
    curr_y = vars[1]
    curr_theta = vars[2]

    dx = vr * np.cos(curr_theta)
    dy = vr * np.sin(curr_theta)
    dtheta = delta
    return [dx,dy,dtheta]

class particleFilter:
    def __init__(self, gem, world, grid_width, grid_height, num_particles, sensor_limit, kernel_sigma, x_start, y_start):
        self.num_particles = num_particles  # The number of particles for the particle filter
        self.sensor_limit = sensor_limit    # The sensor limit of the sensor
        self.rand_particle_ratio = 0.1
        particles = list()
        for i in range(num_particles):
            x = np.random.uniform(0, world.width)
            y = np.random.uniform(0, world.height)
            particles.append(Particle(x = x, y = y, maze = world, sensor_limit = sensor_limit))
        self.particles = particles          # Randomly assign particles at the begining
        self.bob = gem                      # The actual state of the vehicle
        self.world = world                  # The map of the maze
        self.grid_width = grid_width        # Each grid in gazebo simulator is divided into grid_width sections horizontally
        self.grid_height = grid_height      # Each grid in gazebo simulator is divided into grid_height sections vertically
        self.x_start = x_start              # The starting position of the map in the gazebo simulator
        self.y_start = y_start              # The starting position of the map in the gazebo simulator
        self.kernel_sigma = kernel_sigma
        self.modelStatePub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.controlSub = rospy.Subscriber("/gem/control", Float32MultiArray, self.__controlHandler, queue_size = 1)
        self.control = []                   # A list of control signal from the vehicle
        return

    def __controlHandler(self,data):
        tmp = list(data.data)
        self.control.append(tmp)

    def getModelState(self):
        """
        Description:
            Requests the current state of the polaris model when called
        Returns:
            modelState: contains the current model state of the polaris vehicle in gazebo
        """
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            modelState = serviceResponse(model_name='polaris')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
        return modelState

    def weight_gaussian_kernel(self,x1, x2, std = 10):
        """
        Description:
            Given the sensor reading from vehicle and the sensor reading from particle, compute the weight of the particle based on gaussian kernel
        Input:
            x1: The sensor reading from vehicle
            x2: The sensor reading from particle
        Returns:
            Returns weight of the particle
        """
        distance = np.linalg.norm(np.asarray(x1) - np.asarray(x2))
        return np.exp(-distance ** 2 / (2 * std))

    def showMarker(self, x, y):
        """
        Description:
            Update the position of the marker in gazebo environment
        Input:
            x: x position of the marker
            y: y position of the marker
        Returns:
        """
        markerState = ModelState()
        markerState.model_name = 'marker'
        markerState.pose.position.x = x/self.grid_width + self.x_start - 100
        markerState.pose.position.y = y/self.grid_height + self.y_start - 100
        self.modelStatePub.publish(markerState)

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

    def updateWeight(self, readings_robot):

        ## TODO #####
        # Update the weight of each particle based on sensor reading from the robot
        # Normalize the weight

        ###############
        total = 0.0
        for particle in self.particles:
            readings_particle = particle.read_sensor()
            particle.weight = self.weight_gaussian_kernel(readings_robot, readings_particle, 5000)
            total += particle.weight

        if total == 0:
            total = 1e-8

        for particle in self.particles:
            particle.weight /= total

        return

    def resampleParticle_randomized(self, k):

        particles_new = list()

        # Resample particles by randomly sampling some and then sampling the rest by weights
        sum = 0.0
        tot = 0.0
        sum_array = []
        for particle in self.particles:
            sum += particle.weight
            sum_array.append(sum)

        samples_20 = int(len(self.particles)/k)
        samples_80 = len(self.particles) - samples_20

        for i in range(samples_80):
            rand_num = np.random.random()
            index = bisect_left(sum_array, rand_num)
            particle = self.particles[index]
            new_particle = Particle(particle.x, particle.y, particle.maze, particle.heading, particle.weight, particle.sensor_limit, noisy = True)
            particles_new.append(new_particle)

        for i in range(samples_20):
            x = np.random.uniform(0, self.world.width)
            y = np.random.uniform(0, self.world.height)
            new_particle = Particle(x, y, self.world,self.sensor_limit, noisy = True)
            particles_new.append(new_particle)

        self.particles = particles_new

    def resampleParticle(self):

        particles_new = list()

        ## TODO #####
        # Resample particles base on their calculated weight
        # Note: At least TWO different resampling method need to be implemented; You may add new function for other resampling techniques

        ###############
        sum = 0.0
        tot = 0.0
        sum_array = []
        for particle in self.particles:
            sum += particle.weight
            sum_array.append(sum)

        for i in range(self.num_particles):
            index = bisect_left(sum_array, np.random.uniform(0, 1))
            particle = self.particles[index]
            new_particle = Particle(particle.x, particle.y, particle.maze, particle.heading, particle.weight, particle.sensor_limit, noisy = True)
            particles_new.append(new_particle)

        self.particles = particles_new

    def particleMotionModel(self):

        ## TODO #####
        # Update the position of each particle based on the control signal from the vehicle

        ###############
        if self.control == []:
            return

        controls = copy.copy(self.control)
        self.control = []

        for particle in self.particles:

            x_gazebo = (particle.x / 100) - 85
            y_gazebo = (particle.y / 100) + 45
            values = [x_gazebo, y_gazebo, particle.heading]
            r = ode(func1)
            r.set_initial_value(values)

            for u in controls:
                r.set_f_params(u[0],u[1])
                values = r.integrate(r.t+0.01)

            x_python = (values[0] + 85) * 100
            y_python = (values[1] - 45) * 100
            heading_python = values[2] % (np.pi *2)

            if x_python < 0 or x_python > (self.world.width -1) or y_python < 0 or y_python > (self.world.height -1):
                x_python = np.random.uniform(0, self.world.width)
                y_python = np.random.uniform(0, self.world.height)

            particle.x = x_python
            particle.y = y_python
            particle.heading = heading_python 

        return

    def runFilter(self):
        # Run PF localization
        itr = 0
        dist_error_list = []
        heading_error_list = []
        while True:
            ## TODO #####
            # Finish this function to have the particle filter running

            # Read sensor msg
            robot_readings = self.bob.read_sensor()

            #use particleMotionModel
            self.particleMotionModel()
            self.updateWeight(robot_readings)

            # Display actual and estimated state of the vehicle and particles on map
            self.world.show_particles(particles = self.particles, show_frequency = 10)
            self.world.show_robot(robot = self.bob)
            [est_x,est_y,est_heading] = self.world.show_estimated_location(particles = self.particles)
            dist_error = math.sqrt((est_x - self.bob.x)**2 + (est_y - self.bob.y)**2)
            robot_heading = (self.bob.heading * 60) % 360

            heading_error = abs(robot_heading - est_heading)

            dist_error_list.append(dist_error)
            heading_error_list.append(heading_error)

            # with open("T3_dist_300.txt", "w") as file:
            #     file.write(str(dist_error_list))
            #
            # with open("T3_heading_300.txt", "w") as file:
            #     file.write(str(heading_error_list))

            if itr == 100:

                plt.figure()
                #plt.subplot(211)
                plt.plot(dist_error_list)
                plt.title('Random Resample')
                plt.xlabel('Iterations')
                plt.ylabel('Distance Error')

                # plt.subplot(212)
                # plt.plot(heading_error_list)
                # plt.xlabel('Iterations')
                # plt.ylabel('Heading Error')
                plt.show()

            self.world.clear_objects()
            self.showMarker(est_x, est_y)
            itr = itr + 1

            # Resample particles
            if itr <= 6:
                self.resampleParticle_randomized(10)
            # elif itr <= 10 and itr >3:
            #     self.resampleParticle_randomized(6)
            # elif itr > 10 and itr < 15:
            #     self.resampleParticle_randomized(12)
            else:
                self.resampleParticle()
