import math
from random import randint
from typing import Tuple, List

from jaxprvis.math_util import vector_norm, angle_of_vector

DRAG = 0.5

ATTRACT = 0.00001
REPEL = 200000


class PhysicsObject:
    def __init__(self, node_id: str, anchored: bool = False):
        self.node_id = node_id
        self.x: int = 0
        self.y: int = 0
        self.w: int = 60
        self.h: int = 60

        self.vx: float = 0.0
        self.vy: float = 0.0

        self.ax: float = 0.0
        self.ay: float = 0.0

        self.anchored = anchored

    def get_center(self) -> Tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    def get_center_left_edge(self) -> Tuple[int, int]:
        return self.x, self.y + self.h // 2

    def get_center_right_edge(self) -> Tuple[int, int]:
        return self.x + self.w, self.y + self.h // 2

    def get_center_top_edge(self) -> Tuple[int, int]:
        return self.x + self.w//2, self.y

    def get_center_bottom_edge(self) -> Tuple[int, int]:
        return self.x + self.w//2, self.y + self.h

    def apply_force(self, vector: Tuple[float, float]):
        if self.anchored:
            return
        self.ax += vector[0]
        self.ay += vector[1]

    def is_inside(self, box_x, box_y, box_w, box_h):
        return self.x >= box_x and self.x + self.w <= box_x + box_w \
               and self.y >= box_y and self.y + self.h <= box_y + box_h

    def set_random_pos(self, box_x1, box_x2, box_y1, box_y2):
        self.x = randint(box_x1, box_x2)
        self.y = randint(box_y1, box_y2)

    def anchor_to(self, x: int, y: int):
        self.anchored = True
        self.x = x
        self.y = y

    def update(self, nodes: List['PhysicsObject']):
        for other in nodes:
            if other != self:
                self.gravitate(other, False)

        if self.anchored:
            return

        self.vx *= DRAG
        self.vy *= DRAG

        self.vx += self.ax
        self.vy += self.ay

        self.ax = 0
        self.ay = 0

        self.x += int(self.vx)
        self.y += int(self.vy)

    def gravitate(self, other: 'PhysicsObject', attract: bool):
        self_c = self.get_center()
        other_c = other.get_center()

        dist_x = (other_c[0] - self_c[0])
        dist_y = (other_c[1] - self_c[1])

        distance = vector_norm((dist_x, dist_y))

        if distance == 0:
            return

        angle = angle_of_vector((dist_x, dist_y))

        if attract:
            grav_force = -ATTRACT * distance ** 2
            grav_force_x_component = grav_force * math.cos(angle)
            grav_force_y_component = grav_force * math.sin(angle)
            self.apply_force((grav_force_x_component, grav_force_y_component))
            other.apply_force((-grav_force_x_component, -grav_force_y_component))
        else:
            grav_force = REPEL / (distance ** 2)
            grav_force_x_component = -grav_force * (1 - math.cos(angle))
            grav_force_y_component = -grav_force * (1 - math.sin(angle))
            self.apply_force((grav_force_x_component, grav_force_y_component))
            other.apply_force((-grav_force_x_component, -grav_force_y_component))

    def is_point_inside_self(self, pos):
        (x, y) = pos
        return self.x <= x < self.x + self.w and self.y <= y < self.y + self.h
