"""
Environment for one joint Robot Arm.
You can customize this script in a way you want.

Requirement:
pyglet >= 1.2.4
numpy >= 1.12.1
"""

import numpy as np
import pyglet
from pyglet import shapes
import time


class ArmEnv(object):
    action_type = [-1, 1]
    action_dim = len(action_type)
    space_length = 50
    observation_space = [space_length, space_length]  # discret 2*pi into 50 pieces
    d_s = 2*np.pi/space_length
    arm1l = 110
    viewer = None
    viewer_xy = (400, 400)
    mouse_in = np.array([False])
    point_r = 10
    point_track = 130

    def __init__(self, ep_len=50, scale=False):
        # node1 (l, d_rad, x, y),
        self.ep_len = ep_len
        self.ep_step = 0

        self.arm_info = np.zeros(4)
        self.arm_info[0] = self.arm1l
        # the radius of point track, poit angle, end point x,y
        self.point_info = np.zeros(4)
        self.point_info[0] = self.point_track
        self.center_coord = np.array(self.viewer_xy)/2
        # if normalize observation
        self.scale = scale

    def step(self, action_num):
        self.ep_step += 1

        # updata goal value
        self.point_info[2:4] = self.center_coord + np.array([self.point_track * np.cos(self.point_info[1]),
                                                             self.point_track * np.sin(self.point_info[1])])

        action = self.action_type[action_num]
        self.arm_info[1] += action * self.d_s
        self.arm_info[1] %= np.pi * 2

        arm1rad = self.arm_info[1]
        arm1dx_dy = np.array([self.arm_info[0] * np.cos(arm1rad), self.arm_info[0] * np.sin(arm1rad)])
        self.arm_info[2:4] = self.center_coord + arm1dx_dy  # (x1, y1)

        s, distance = self._get_state()
        r = self._r_func(distance)

        if (self.ep_step >= self.ep_len) or self.touched:
            self.done = True
        else:
            self.done = False

        # if touched the game stop early
        return s, r, self.done, self.touched

    def reset(self, state=None):
        self.done = False
        self.ep_step = 0
        if state is None:
            # random the point position
            arm1rad, self.point_info[1] = np.random.randint(50, size=2) * self.d_s
        else:
            arm1rad, self.point_info[1] = state * self.d_s
        self.point_info[2:4] = self.center_coord + np.array([self.point_track * np.cos(self.point_info[1]),
                                                             self.point_track * np.sin(self.point_info[1])])

        self.arm_info[1] = arm1rad
        arm1dx_dy = np.array([self.arm_info[0] * np.cos(arm1rad), self.arm_info[0] * np.sin(arm1rad)])
        self.arm_info[2:4] = self.center_coord + arm1dx_dy  # (x1, y1)

        return self._get_state()[0]

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(*self.viewer_xy, self.arm_info, self.point_info, self.point_r, self.point_track)
        self.viewer.render()
        time.sleep(0.1)

    def sample_action(self):
        return np.random.randint(len(self.action_type))

    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)

    def _get_state(self):
        # return the distance (dx, dy) between arm finger point with blue point
        arm_angle = int(self.arm_info[1]/self.d_s)
        point_angle = int(self.point_info[1]//self.d_s)
        distance = np.linalg.norm(self.point_info[2:4] - self.arm_info[2:4])
        if self.scale:
            return np.hstack([arm_angle/self.space_length, point_angle/self.space_length]), distance
        else:
            return np.hstack([arm_angle, point_angle]), distance

    def _r_func(self, distance):
        self.touched = False
        r = - distance/240
        # with early stop
        if (1-distance/240) > 0.9: # touched
            r += 1
            self.touched = True
        return r

    def get_instant_reward(self, state: list, action: int):
        """
        state: angle manipulator, angle of point
        :return:
        """
        self.arm_info[1] = state[0] * self.d_s
        self.point_info[1] = state[1] * self.d_s
        _, r, _ = self.step(action)
        return r


class Viewer(pyglet.window.Window):
    color = {
        'background': [1]*3 + [1]
    }

    bar_thc = 5

    def __init__(self, width, height, arm_info, point_info, point_r, point_track):
        super(Viewer, self).__init__(width, height, resizable=False, caption='Arm', vsync=False)  # vsync=False to not use the monitor FPS
        self.set_location(x=80, y=10)
        pyglet.gl.glClearColor(*self.color['background'])

        self.arm_info = arm_info
        self.point_info = point_info
        self.point_r = point_r
        self.point_track = point_track

        self.center_coord = np.array((min(width, height)/2, ) * 2)
        self.batch = pyglet.graphics.Batch()

        c1, c2 = (249, 86, 86), (86, 109, 249)

        # draw the point track
        self.circle1 = shapes.Circle(self.center_coord[0], self.center_coord[1],
                                     self.point_track+5, color=(25, 225, 25), batch=self.batch)
        self.circle1.opacity = 250
        self.circle2 = shapes.Circle(self.center_coord[0], self.center_coord[1],
                                     self.point_track-5, color=(225, 225, 225), batch=self.batch)
        self.circle2.opacity = 150
        self.point = shapes.Circle(self.point_info[2], self.point_info[3], self.point_r,
                                   color=(250, 25, 30), batch=self.batch)

        self.arm1 = shapes.Rectangle(self.center_coord[0], self.center_coord[1], self.arm_info[0], self.bar_thc,
                                     c2, batch=self.batch)
        self.arm1.anchor_position = [0, self.bar_thc/2]

    def render(self):
        pyglet.clock.tick()
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_arm(self):
        self.point.position = self.point_info[2:4]
        self.arm1.rotation = - self.arm_info[1] * 180/np.pi

