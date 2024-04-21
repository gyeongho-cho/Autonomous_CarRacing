__credits__ = ["Gyeongho Cho"]
'''
This code is created by modifying 'CarRacing-V1' from OpenAI gym library.
This code is made for term project of Pusan National University AI system lecture.
'''

import sys
import math
from typing import Optional

import numpy as np
import pygame
from pygame.locals import *
from pygame import gfxdraw

import Box2D
from Box2D.b2 import fixtureDef
from Box2D.b2 import polygonShape, edgeShape
from Box2D.b2 import contactListener, rayCastCallback
from Box2D import b2RayCastCallback, b2Vec2

import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.utils import seeding, EzPickle

import matplotlib.pyplot as plt

STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]
LIDAR_RANGE = 50
num_lidars = 20
LIDAR_ANGLES = [-math.pi*2*i/num_lidars for i in range(num_lidars)]

class RayCastClosestCallback(b2RayCastCallback):
    """This callback finds the closest object"""

    def __repr__(self):
        return 'Closest object'

    def __init__(self, **kwargs):
        b2RayCastCallback.__init__(self, **kwargs)
        self.fixture = None
        self.hit = False
        self.point = None
        self.normal = None
        self.fraction = 10
        self.range = LIDAR_RANGE
    def ReportFixture(self, fixture, point, normal, fraction):
        '''
        Search all objects and memory only closest object.
        '''
        if fixture.friction>=10:
            if self.fraction>fraction:
                self.hit = True
                self.fixture = fixture
                self.point = b2Vec2(point)
                self.normal = b2Vec2(normal)    
                self.fraction = fraction 
            return True
        else:
            return True
        
    def reset(self):
        self.fixture = None
        self.hit = False
        self.point = None
        self.normal = None
        self.fraction = 10
        self.range = LIDAR_RANGE


class LidarSensor():
    def __init__(self, raycast):
        self.hit = raycast.hit
        self.fixture = raycast.fixture
        self.point = raycast.point
        self.normal = raycast.normal
        self.fraction = raycast.fraction


class FrictionDetector(contactListener):
    def __init__(self, env, lap_complete_percent):
        contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            self.env.done = True
            return

        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1

                # Lap is considered completed if enough % of the track was covered
                if (
                    tile.idx == 0
                    and self.env.tile_visited_count / len(self.env.track)
                    > self.lap_complete_percent
                ):
                    self.env_new_lap = True
        else:
            obj.tiles.remove(tile)


class Autonomous_CarRacing(gym.Env, EzPickle):
    """
    ### Description
    The easiest continuous control task to learn from pixels - a top-down
    racing environment. Discrete control is reasonable in this environment as
    well; on/off discretization is fine.

    The game is solved when the agent consistently gets 900+ points.
    The generated track is random every episode.

    Some indicators are shown at the bottom of the window along with the
    state RGB buffer. From left to right: true speed, four ABS sensors,
    steering wheel position, and gyroscope.
    To play yourself (it's rather fast for humans), type:
    ```
    python gym/envs/box2d/car_racing.py
    ```
    Remember: it's a powerful rear-wheel drive car - don't press the accelerator
    and turn at the same time.

    ### Action Space
    There is 1 action: steering (-1 is full left, +1 is full right)

    ### Observation Space
    There are 22 observations: 20ch depth from lidar sensors (0 ~ 50), current steering angle (-1 ~ +1),
    driving status (-1:crushed, 0:driving, 1:goal)

    ### Rewards
    Zero

    ### Starting State
    The car starts at rest in the center of the road.

    ### Episode Termination
    The episode finishes when all of the tiles are visited. The car can also go
    outside of the playfield - that is, far off the track, in which case it will
    receive -100 reward and die.

    ### Arguments
    There are no arguments supported in constructing the environment.

    ### Version History
    - v0: Current version

    ### References
    - Chris Campbell (2014), http://www.iforce2d.net/b2dtut/top-down-car.
    - OpenAI gym, https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
    
    ### Credits
    Created by Gyeongho Cho
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "state_pixels"],
        "render_fps": FPS,
    }

    def __init__(self, verbose=1, lap_complete_percent=0.9, init_vel = 0):
        EzPickle.__init__(self)
        pygame.init()
        self.contactListener_keepref = FrictionDetector(self, lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.world2 = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen = None
        self.clock = None
        self.isopen = True
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.new_lap = False
        self.init_vel = init_vel
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )
        self.fd_object = fixtureDef(
            # shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
            shape=edgeShape(vertices=[(0, 0), (1, 0)])
        )
        self.raycast = RayCastClosestCallback()
        self.lidar_sensors = []

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised however this is not possible here so ignore
        self.action_space = spaces.Box(
            np.array([-1, 0, 0]).astype(np.float32),
            np.array([+1, +1, +1]).astype(np.float32),
        )  # steer, gas, brake

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )
        
    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        for t2 in self.obj:
            self.world.DestroyBody(t2)
        self.road = []
        self.obj = []
        self.car.destroy()

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []
        self.obj = []
        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            while True:  # Find destination from checkpoints
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2 * math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = (
                track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose == 1:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1 : i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            dx = x2-x1
            dy = y2-y1
            
            #-------object start------
            # right
            b1_l = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            b1_r = (
                x1 + (TRACK_WIDTH + BORDER) * math.cos(beta1),
                y1 + (TRACK_WIDTH + BORDER) * math.sin(beta1),
            )
            b1_ll = (
                x1 + (TRACK_WIDTH + 1) * math.cos(beta1) - dx*1,
                y1 + (TRACK_WIDTH + 1) * math.sin(beta1) - dy*1,
            )
            b1_rr = (
                x1 + (TRACK_WIDTH - 0.5) * math.cos(beta1) - dx*0.5,
                y1 + (TRACK_WIDTH - 0.5) * math.sin(beta1) - dy*0.5,
            )
            b2_l = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            b2_r = (
                x2 + (TRACK_WIDTH + BORDER) * math.cos(beta2),
                y2 + (TRACK_WIDTH + BORDER) * math.sin(beta2),
            )
            b2_ll = (
                x2 + (TRACK_WIDTH + 1) * math.cos(beta2) + dx*1,
                y2 + (TRACK_WIDTH + 1) * math.sin(beta2) + dy*1,
            )
            b2_rr = (
                x2 + (TRACK_WIDTH - 0.5) * math.cos(beta2) + dx*0.5,
                y2 + (TRACK_WIDTH - 0.5) * math.sin(beta2) + dy*0.5,
            )
            self.obj_poly.append(
                ([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0))
            )
            
            # self.fd_object.shape.vertices = [b1_l, b1_r, b2_r, b2_l]
            # self.fd_object.shape.vertices = [b1_l, b1_rr, b2_rr, b2_l]
            self.fd_object.shape.vertices = [b1_l, b2_l]
            self.fd_object.friction = 10
            t2 = self.world.CreateStaticBody(fixtures=self.fd_object)
            t2.color = [1, 1, 1] if i % 2 == 0 else [1, 0, 0]
            t2.road_visited = False
            t2.road_friction = 1.0
            t2.idx = i
            t2.fixtures[0].sensor = True
            t2.userData = t2
            self.obj.append(t2)
            
            # left
            b1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            b1_r = (
                x1 - (TRACK_WIDTH + BORDER) * math.cos(beta1),
                y1 - (TRACK_WIDTH + BORDER) * math.sin(beta1),
            )
            b1_ll = (
                x1 - (TRACK_WIDTH + 0) * math.cos(beta1) - dx*1,
                y1 - (TRACK_WIDTH + 0) * math.sin(beta1) - dy*1,
            )
            b1_rr = (
                x1 - (TRACK_WIDTH - 0.5) * math.cos(beta1) - dx*0.5,
                y1 - (TRACK_WIDTH - 0.5) * math.sin(beta1) - dy*0.5,
            )
            b2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            b2_r = (
                x2 - (TRACK_WIDTH + BORDER) * math.cos(beta2),
                y2 - (TRACK_WIDTH + BORDER) * math.sin(beta2),
            )
            b2_ll = (
                x2 - (TRACK_WIDTH + 0) * math.cos(beta2) + dx*1,
                y2 - (TRACK_WIDTH + 0) * math.sin(beta2) + dy*1,
            )
            b2_rr = (
                x2 - (TRACK_WIDTH - 0.5) * math.cos(beta2) + dx*0.5,
                y2 - (TRACK_WIDTH - 0.5) * math.sin(beta2) + dy*0.5,
            )
            self.obj_poly.append(
                ([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0))
            )
            # print('-'*5)
            # print([b1_ll, b1_rr, b2_rr, b2_ll])
            # self.fd_object.shape.vertices = [b1_l, b1_r, b2_r, b2_l]
            # self.fd_object.shape.vertices = [b1_l, b1_rr, b2_rr, b2_l]
            self.fd_object.shape.vertices = [b1_l, b2_l]
            
            # print(self.fd_object.shape.vertices)
            self.fd_object.friction = 10
            t2 = self.world.CreateStaticBody(fixtures=self.fd_object)
            t2.color = [1, 1, 1] if i % 2 == 0 else [1, 0, 0]
            t2.road_visited = False
            t2.road_friction = 1.0
            t2.idx = i
            t2.fixtures[0].sensor = True
            t2.userData = t2
            self.obj.append(t2)
            #-------object end------
            
            #road
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            c = 0.01 * (i % 3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.idx = i
            t.fixtures[0].sensor = True
            t.userData = t
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            # if border[i]:
            #     side = np.sign(beta2 - beta1)
            #     b1_l = (
            #         x1 + side * TRACK_WIDTH * math.cos(beta1),
            #         y1 + side * TRACK_WIDTH * math.sin(beta1),
            #     )
            #     b1_r = (
            #         x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
            #         y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
            #     )
            #     b2_l = (
            #         x2 + side * TRACK_WIDTH * math.cos(beta2),
            #         y2 + side * TRACK_WIDTH * math.sin(beta2),
            #     )
            #     b2_r = (
            #         x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
            #         y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
            #     )
            #     self.road_poly.append(
            #         ([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0))
            #     )
            
            
        self.track = track
        return True

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        np.random.seed(seed)
        
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.new_lap = False
        self.step_time = pygame.time.get_ticks()
        self.plot_time = pygame.time.get_ticks()
        while True:
            self._destroy()
            self.road_poly = []
            self.obj_poly = []
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        
        if options:
            # if options['new_road']:
                        
            if options['init_vel']:
                self.init_vel = options['init_vel']
                
        
        self.car = Car(self.world, *self.track[0][1:4])
        
        # # place random position and directions
        # idx = np.random.randint(0, len(self.track))
        # if idx%2 == 0:
        #     self.car = Car(self.world, self.track[idx][1], *self.track[idx][2:4])
        # else:
        #     self.car = Car(self.world, self.track[idx][1]+np.pi, *self.track[idx][2:4])
        
        if self.init_vel:
            for i, w in enumerate(self.car.wheels):
                self.car.wheels[i].omega = self.init_vel*1.35
            
        if not return_info:
            return self.step(None)[0]
        else:
            return self.step(None)[0], {}

    def step(self, action):
        self.step_time = pygame.time.get_ticks()
        if action is not None:
            self.car.steer(-action)
            self.car.gas(0)
            self.car.brake(0)
        
        if self.init_vel:
            total_omega = 0.
            for i, w in enumerate(self.car.wheels):                
                total_omega += self.car.wheels[i].omega
            total_omega /= 4
            gain = self.init_vel/total_omega
            for i, w in enumerate(self.car.wheels):                
                self.car.wheels[i].omega = self.car.wheels[i].omega*gain
                
        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS 
        # self.state = self.render("rgb_array")
        # self.state = self.getDistance(self.car, world)
        
        step_reward = 0
        done = False
        
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track) or self.new_lap:
                done = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -100
                
        lidar_sensors = []
        for lidar_angle in LIDAR_ANGLES:
            point1 = self.car.hull.position
            angle = self.car.hull.angle+lidar_angle
            d = (-LIDAR_RANGE * math.sin(angle), LIDAR_RANGE * math.cos(angle))
            point2 = point1 + d
            self.point1 = point1
            self.point2 = point2
            self.raycast.reset()
            self.world.RayCast(self.raycast, point1, point2)  
            lidar = LidarSensor(self.raycast)
            if (self.car.hull.position-lidar.point).length < LIDAR_RANGE:
                lidar.range = (self.car.hull.position-lidar.point).length
            else:
                lidar.range = LIDAR_RANGE
            lidar_sensors.append(lidar)
        self.lidar_sensors = lidar_sensors
        
        
            
        for w in self.car.wheels:
            for tile in w.tiles:
                if tile.fixtures[0].friction >=10:
                    done = True
        
        if done:
            if self.tile_visited_count == len(self.track):
                is_done = 1
            else:
                is_done = -1
        else:
            is_done = 0
        
        self.state = np.array([l.range for l in self.lidar_sensors]+[self.car.wheels[0].joint.angle, is_done])
        
        return self.state, 0, done, {}

    def render(self, mode="human"):
        assert mode in ["human", "state_pixels", "rgb_array"]
        if self.screen is None and mode == "human":
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        # computing transformations
        angle = -self.car.hull.angle
        # Animating first second zoom.
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -(self.car.hull.position[0] + PLAYFIELD) * zoom
        scroll_y = -(self.car.hull.position[1] + PLAYFIELD) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self.render_road(zoom, trans, angle)
        self.car.draw(self.surf, zoom, trans, angle, mode != "state_pixels")

        self.surf = pygame.transform.flip(self.surf, False, True)
        
        # draw raycast
        x=500
        y=600
        for i, lidar in enumerate(self.lidar_sensors):
                 
            omega = LIDAR_ANGLES[i]
            dx = np.sin(-omega)*lidar.range*zoom
            dy = -np.cos(-omega)*lidar.range*zoom
            # raycolor = (int(100/len(LIDAR_ANGLES)*(i+1)),int(100/len(LIDAR_ANGLES)*(i+1)),int(255/len(LIDAR_ANGLES)*(i+1)))
            raycolor = (100,100,255)
            pygame.draw.lines(self.surf, raycolor, True, [(x,y), (x+dx,y+dy)], 2)
        
        # showing stats
        self.render_indicators(WINDOW_W, WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        self.surf.blit(text, text_rect)
        
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()

        if mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            state_img = self._create_image_array(self.surf, (STATE_W, STATE_H))
            plt.imshow(state_img)
            plt.show()
            return state_img
        else:
            return self.isopen

    def render_road(self, zoom, translation, angle):
        bounds = PLAYFIELD
        field = [
            (2 * bounds, 2 * bounds),
            (2 * bounds, 0),
            (0, 0),
            (0, 2 * bounds),
        ]
        trans_field = []
        self.draw_colored_polygon(
            self.surf, field, (102, 204, 102), zoom, translation, angle
        )

        k = bounds / (20.0)
        grass = []
        for x in range(0, 40, 2):
            for y in range(0, 40, 2):
                grass.append(
                    [
                        (k * x + k, k * y + 0),
                        (k * x + 0, k * y + 0),
                        (k * x + 0, k * y + k),
                        (k * x + k, k * y + k),
                    ]
                )
        for poly in grass:
            self.draw_colored_polygon(
                self.surf, poly, (102, 230, 102), zoom, translation, angle
            )

        for poly, color in self.road_poly:
            # converting to pixel coordinates
            poly = [(p[0] + PLAYFIELD, p[1] + PLAYFIELD) for p in poly]
            color = [int(c * 255) for c in color]
            self.draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)
        
        # for poly, color in self.road_poly:
        #     # converting to pixel coordinates
        #     poly = [(p[0] + PLAYFIELD, p[1] + PLAYFIELD) for p in poly]
        #     color = [int(c * 255) for c in color]
        #     self.draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)
        
        for poly, color in self.obj_poly:
            # converting to pixel coordinates
            poly = [(p[0] + PLAYFIELD, p[1] + PLAYFIELD) for p in poly]
            color = [int(c * 255) for c in color]
            self.draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)
    
    # def render_object(self, zoom, translation, angle):
        
    #     for poly, color in self.object_poly:
    #         # converting to pixel coordinates
    #         poly = [(p[0] + PLAYFIELD, p[1] + PLAYFIELD) for p in poly]
    #         color = [int(c * 255) for c in color]
    #         self.draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)

    def render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)

        def vertical_ind(place, val):
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]

        def horiz_ind(place, val):
            return [
                ((place + 0) * s, H - 4 * h),
                ((place + val) * s, H - 4 * h),
                ((place + val) * s, H - 2 * h),
                ((place + 0) * s, H - 2 * h),
            ]

        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )

        # simple wrapper to render if the indicator value is above a threshold
        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(self.surf, points=points, color=color)

        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
        # ABS sensors
        render_if_min(
            self.car.wheels[0].omega,
            vertical_ind(7, 0.01 * self.car.wheels[0].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[1].omega,
            vertical_ind(8, 0.01 * self.car.wheels[1].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[2].omega,
            vertical_ind(9, 0.01 * self.car.wheels[2].omega),
            (51, 0, 255),
        )
        render_if_min(
            self.car.wheels[3].omega,
            vertical_ind(10, 0.01 * self.car.wheels[3].omega),
            (51, 0, 255),
        )

        render_if_min(
            self.car.wheels[0].joint.angle,
            horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle),
            (0, 255, 0),
        )
        render_if_min(
            self.car.hull.angularVelocity,
            horiz_ind(30, -0.8 * self.car.hull.angularVelocity),
            (255, 0, 0),
        )

    def draw_colored_polygon(self, surface, poly, color, zoom, translation, angle):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        gfxdraw.aapolygon(self.surf, poly, color)
        gfxdraw.filled_polygon(self.surf, poly, color)

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )
    def plot(self, crop=True):
        
        if self.step_time>self.plot_time:
            track = np.array(self.track)
            plt.plot(track[:,2], track[:,3])
            # for i in range(len(track)):
            #     plt.plot([track[i,2],track[i,2]-10*math.sin(track[i,1])],[track[i,3],track[i,3]+10*math.cos(track[i,1])])
            
            # for i, r in enumerate(self.road):
            #     fix = np.array(r.fixtures[0].shape.vertices)    
            #     plt.plot(fix[:,0], fix[:,1])

            for i, o in enumerate(self.obj):
                fix = np.array(o.fixtures[0].shape.vertices)    
                plt.plot(fix[:,0], fix[:,1])
                
            plt.scatter(self.car.hull.position[0],self.car.hull.position[1])
            for i, lidar in enumerate(self.lidar_sensors):
                if lidar.hit:
                    plt.scatter(lidar.point[0],lidar.point[1])
                    fix = np.array(lidar.fixture.shape.vertices)
                    plt.plot(fix[:,0]+lidar.fixture.body.position[0], fix[:,1]+lidar.fixture.body.position[1], color='red')
            if crop:
                plt.xlim([self.car.hull.position[0]-30,self.car.hull.position[0]+30])
                plt.ylim([self.car.hull.position[1]-30,self.car.hull.position[1]+30])
            # plt.scatter(x_list,y_list)
            # for i, a in enumerate(a_list):
            #     x = x_list[i]
            #     y = y_list[i]
            #     plt.plot([x,x+10*math.sin(-a)],[y,y-10*math.cos(-a)])
            # road_block = np.array(env.road_poly[0][0])
            # plt.plot(road_block[:,0], road_block[:,1])
            plt.show()
            self.plot_time+=1000


    def close(self):
        pygame.quit()
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False


if __name__ == "__main__":
        
    def plot_trajectory():
        plt.figure(figsize=(12,5))
        plt.subplot(121)
        plt.title('trajectory')
        track = np.array(env.track)
        plt.plot(track[:,2], track[:,3])
        
        for i, r in enumerate(env.obj):
            fix = np.array(r.fixtures[0].shape.vertices)    
            plt.plot(fix[:,0], fix[:,1])
        
        plt.plot(x_list,y_list, 'r', linewidth=5)
        
        plt.subplot(122)
        plt.title('action')
        plt.plot(act_list,'.-')
        plt.show()
        
        
    env = Autonomous_CarRacing(verbose=0)
    init_vel = 80
    
    is_pressed = [False]*5 # [up, down, left, right, esc]
    
    while True:
        env.reset(options = {'init_vel': init_vel})
        
        steer = 0.
        t=-1
        x_list = []
        y_list = []
        a_list = []
        act_list = []
        while True: 
            t+=1
            
            env.render() # show with pygame
            
            # logging car position and angle for plot
            x, y = env.car.hull.position
            a = env.car.hull.angle
            x_list.append(x)
            y_list.append(y)
            a_list.append(a)
            act_list.append(steer)
            
            events = pygame.event.get()
            
            for event in events:
                if event.type == QUIT:
                     print('quitting')
                     pygame.quit()
                     sys.exit()
                else:
                    if event.type == KEYDOWN:
                        if event.key == K_UP:
                            is_pressed[0] = True
                        if event.key == K_DOWN:
                            is_pressed[1] = True
                        if event.key == K_RIGHT:
                            is_pressed[3] = True
                        elif event.key == K_LEFT:
                            is_pressed[2] = True
                        if event.key == K_ESCAPE:
                            is_pressed[4] = True
                    if event.type == KEYUP:
                        if event.key == K_UP:
                            is_pressed[0] = False
                        if event.key == K_DOWN:
                            is_pressed[1] = False
                        if event.key == K_RIGHT:
                            is_pressed[3] = False
                        elif event.key == K_LEFT:
                            is_pressed[2] = False
                        if event.key == K_ESCAPE:
                            is_pressed[4] = False
            
            if is_pressed[2] & is_pressed[3]:
                pass
            elif is_pressed[2]:
                steer-=0.02
                steer = max(-1,steer)
            elif is_pressed[3]:
                steer+=0.02
                steer = min(1,steer)
            else:
                if steer>0.001:
                    steer-=0.01
                elif steer<-0.001:  
                    steer+=0.01
                else:
                    steer=0
                                
            # simulate step
            observation, _, done, _ = env.step(steer)
            
            if done or is_pressed[4]:
                print("Episode finished after {} timesteps".format(t+1))
                break
        
        # plot logged trajectory
        plot_trajectory()
        
        if is_pressed[4]:
            break
    env.close()