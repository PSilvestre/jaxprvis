import math
from random import randint
from typing import Tuple, List

from jaxprvis.physics.math_util import vector_norm, angle_of_vector
from dataclasses import dataclass, field

# TODO https://github.com/MyreMylar/pygame_gui/
DRAG = 0.5

ATTRACT = 0.00001
REPEL = 200000


@dataclass(frozen=False, slots=True)
class PhysicsObject:
    node_id: str  # = node_id

    w: int  # = 60
    h: int  # = 60

    x: int = field(default=0)  # = 0
    y: int = field(default=0)  # = 0

    vx: float = field(default=0.0)  # = 0.0
    vy: float = field(default=0.0)  # = 0.0

    ax: float = field(default=0.0)  # = 0.0
    ay: float = field(default=0.0)  # = 0.0

    mass: float = field(default=1.0)

    # Controls whether the position of this object can be updated by physics
    anchored: bool = field(default=False)  # = anchored

    def get_center(self) -> Tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    def get_center_left_edge(self) -> Tuple[int, int]:
        return self.x, self.y + self.h // 2

    def get_center_right_edge(self) -> Tuple[int, int]:
        return self.x + self.w, self.y + self.h // 2

    def get_center_top_edge(self) -> Tuple[int, int]:
        return self.x + self.w // 2, self.y

    def get_center_bottom_edge(self) -> Tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h

    def set_random_pos(self, box_x1, box_x2, box_y1, box_y2):
        self.x = randint(box_x1, box_x2)
        self.y = randint(box_y1, box_y2)

    def anchor_to(self, x: int, y: int):
        self.anchored = True
        self.x = x
        self.y = y

    def is_point_inside_self(self, pos):
        (x, y) = pos
        return self.x <= x < self.x + self.w and self.y <= y < self.y + self.h

    def is_inside(self, box_x, box_y, box_w, box_h):
        return self.x >= box_x and self.x + self.w <= box_x + box_w \
            and self.y >= box_y and self.y + self.h <= box_y + box_h


def apply_force(node: PhysicsObject, vector: Tuple[float, float]):
    # f = ma
    # f/m = a
    if node.anchored:
        return
    node.ax += vector[0] / node.mass
    node.ay += vector[1] / node.mass


def update(node: PhysicsObject, nodes: List[PhysicsObject]):
    for other in nodes:
        if other != node:
            gravitate(node, other, False)

    if node.anchored:
        return

    node.vx *= DRAG
    node.vy *= DRAG

    node.vx += node.ax
    node.vy += node.ay

    node.ax = 0
    node.ay = 0

    node.x += int(node.vx)
    node.y += int(node.vy)


def gravitate(this: PhysicsObject, other: PhysicsObject, attract: bool):
    this_c = this.get_center()
    other_c = other.get_center()

    dist_x = (other_c[0] - this_c[0])
    dist_y = (other_c[1] - this_c[1])

    distance = vector_norm((dist_x, dist_y))

    if distance == 0:
        return

    angle = angle_of_vector((dist_x, dist_y))

    if attract:
        grav_force = -ATTRACT * distance ** 2
        grav_force_x_component = grav_force * math.cos(angle)
        grav_force_y_component = grav_force * math.sin(angle)
        apply_force(this, (grav_force_x_component, grav_force_y_component))
        apply_force(other, (-grav_force_x_component, -grav_force_y_component))
    else:
        grav_force = REPEL / (distance ** 2)
        grav_force_x_component = -grav_force * (1 - math.cos(angle))
        grav_force_y_component = -grav_force * (1 - math.sin(angle))
        apply_force(this, (grav_force_x_component, grav_force_y_component))
        apply_force(other, (-grav_force_x_component, -grav_force_y_component))
