import gym
from gym import spaces
import numpy as np
import cv2
from utils import *


class SailingEnv(gym.Env):
    def __init__(self):
        super(SailingEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # 0: left, 1: right, 2: no action
        # these low and high parameters are for sanity check, they could be np.inf and -np.inf
        # the shape is the number of inputs in the observation space
        self.observation_space = spaces.Box(
            low=-1000, high=1000, shape=(5,), dtype=np.float32
        )
        self.tck = initialize_speed_config()

    def render(self):
        cv2.imshow("SailingEnv", self.img)
        cv2.waitKey(1)

        # initialize map and dashboard
        self.img = draw_sea_and_dashboard(
            self.wind_direction, self.boat_heading, self.boat_speed, self.total_reward
        )

        # Display Target
        self.img = draw_target(self.img, self.target_position)

        # optional:
        self.img = draw_boat_rectangle(self.img, self.boat_position)

        # Display Boat
        self.img = draw_boat(self.img, self.boat_position, self.boat_heading)

        # Display Boat Path
        self.img = draw_boat_path(self.img, self.position_memory)

        step()

    def step(self, action):

        button_direction = action

        if button_direction == 1:
            self.boat_heading += 10
        elif button_direction == 0:
            self.boat_heading -= 10

        self.boat_heading = self.boat_heading % 360

        # get boat speed
        self.boat_speed = get_boat_speed(
            self.boat_heading, self.wind_direction, self.tck
        )

        # update boat position

        x = self.boat_speed * np.sin(degrees_to_radians(self.boat_heading))
        y = self.boat_speed * np.cos(degrees_to_radians(self.boat_heading))

        self.boat_position[0] += x
        self.boat_position[1] += y

        self.position_memory.append(self.boat_position.copy())

        target_reward = 0

        if euclidean_distance(self.boat_position, self.target_position) < 30:
            self.target_position = reposition_target(self.target_position)
            target_reward = 10000
            self.done = True

        # On collision with boundaries kill the game
        if collision_with_boundaries(self.boat_position) == 1:
            target_reward = -10000
            self.done = True

        euclidean_dist_to_target = np.linalg.norm(
            np.array(self.boat_position) - np.array(self.target_position)
        )

        self.total_reward = (
            (250 - euclidean_dist_to_target) + target_reward - 2 * self.time
        ) / 100

        # Update wind direction
        self.wind_direction = 180 + np.random.normal(0, 3)
        self.wind_direction = self.wind_direction % 360

        boat_position_x = self.boat_position[0]
        boat_position_y = self.boat_position[1]

        target_delta_x = self.target_position[0] - boat_position_x
        target_delta_y = self.target_position[1] - boat_position_y

        # create observation:
        observation = [
            boat_position_x,
            boat_position_y,
            target_delta_x,
            target_delta_y,
            self.boat_heading,
        ]

        observation = np.array(observation).astype("float32")

        info = {}

        self.time += 1

        return observation, self.total_reward, self.done, info

    def reset(self):

        self.done = False

        self.time = 0

        self.position_memory = []

        self.img = np.zeros((600, 900, 3), dtype="uint8")

        # environment setup (wind):
        self.wind_direction = 180

        # Initial Boat and Target position
        self.target_position = [350, 75]
        self.button_direction = 2
        self.boat_position = [350, 450]
        self.boat_heading = 180 - 45

        boat_position_x = self.boat_position[0]
        boat_position_y = self.boat_position[1]

        target_delta_x = self.target_position[0] - boat_position_x
        target_delta_y = self.target_position[1] - boat_position_y

        # create observation:
        observation = [
            boat_position_x,
            boat_position_y,
            target_delta_x,
            target_delta_y,
            self.boat_heading,
        ]
        observation = np.array(observation).astype("float32")

        return observation
