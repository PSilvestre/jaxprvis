import math
import sys

import pygame
from numpy import argmin

from jaxprvis.graph.graph import Graph, CodeNode
from jaxprvis.physics.math_util import vector_norm, angle_of_vector
from jaxprvis.render.display_assets import DisplayAssets
from jaxprvis.physics.physics import PhysicsObject, gravitate, update, apply_force
from jaxprvis.visualization.colorcheme import Colorscheme

SCREEN_WIDTH = 1800
SCREEN_HEIGHT = 1000

SCALE = 2

SIMULATION_WIDTH = SCALE * SCREEN_WIDTH
SIMULATION_HEIGHT = SCALE * SCREEN_HEIGHT

FRAME_RATE = 60

ANCHOR_HEIGHT_SECTION_RATIO = 7 / 8


def _anchor_inputs_and_outputs(graph: Graph):
    usable_px = (SIMULATION_HEIGHT * ANCHOR_HEIGHT_SECTION_RATIO)
    non_usable_px = (SIMULATION_HEIGHT * (1 - ANCHOR_HEIGHT_SECTION_RATIO))
    if len(graph.inputs) > 0:
        input_spacing = usable_px / (len(graph.inputs) + 1)
        for (i, node) in enumerate(graph.inputs):
            y_pos = non_usable_px / 2 + (i + 1) * input_spacing
            node.phys.anchor_to(SIMULATION_WIDTH // 32, y_pos - node.phys.h // 2)

    if len(graph.outputs) > 0:
        output_spacing = usable_px / (len(graph.outputs) + 1)
        for (i, node) in enumerate(graph.outputs):
            y_pos = non_usable_px / 2 + (i + 1) * output_spacing
            node.phys.anchor_to(SIMULATION_WIDTH // 32 * 31 - node.phys.w, y_pos - node.phys.h // 2)


class Simulation:
    def __init__(self, g: Graph, colorscheme: Colorscheme):
        self.colorscheme: Colorscheme = colorscheme

        pygame.init()
        pygame.display.set_caption("Dataflow Visualizer")

        self.screen: pygame.Surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.stage: pygame.Surface = pygame.Surface((SIMULATION_WIDTH, SIMULATION_HEIGHT))

        # TODO: Might make a "Controller" abstraction later
        self.paused = False
        self.selected_node = None
        self.select_offset_x = 0
        self.select_offset_y = 0

        self.computation_stack = []
        self.computation_stack.append(g)

        self.display_assets = []
        self.physics_objs = []

        self.initialize_graph_display()

    def initialize_graph_display(self):
        self.display_assets = []
        self.physics_objs = []

        for node in self.computation_stack[-1].nodes.values():
            phys: PhysicsObject = PhysicsObject(node.node_id, 60, 60)
            assets: DisplayAssets = DisplayAssets(node, phys, self.colorscheme)

            # Set node internal references
            node.phys = phys
            node.assets = assets

            # Append to list of objects
            self.physics_objs.append(phys)
            self.display_assets.append(phys)

            phys.set_random_pos(SIMULATION_WIDTH / 8, SIMULATION_WIDTH / 8 * 7, SIMULATION_HEIGHT / 8,
                                SIMULATION_HEIGHT / 8 * 7)
        _anchor_inputs_and_outputs(self.computation_stack[-1])

    def start(self):
        clock = pygame.time.Clock()
        while True:
            self.handle_inputs()
            if not self.paused:
                self.update()
            self.render()
            clock.tick(FRAME_RATE)

    def handle_inputs(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == pygame.BUTTON_LEFT:
                    scaled_pos = (event.pos[0] * SCALE, event.pos[1] * SCALE)
                    for node in self.physics_objs:
                        if node.is_point_inside_self(scaled_pos):
                            self.select_offset_x = node.x - scaled_pos[0]
                            self.select_offset_y = node.y - scaled_pos[1]
                            self.selected_node = node
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == pygame.BUTTON_LEFT:
                    scaled_pos = (event.pos[0] * SCALE, event.pos[1] * SCALE)
                    self.selected_node = None
                    for node in self.physics_objs:
                        if node.is_point_inside_self(scaled_pos):
                            node_id = node.node_id
                            graph_node = self.computation_stack[-1].nodes[node_id]
                            if isinstance(graph_node, CodeNode):
                                self.computation_stack.append(graph_node.jaxpr)
                                self.initialize_graph_display()
                if event.button == pygame.BUTTON_RIGHT:
                    if len(self.computation_stack) > 1:
                        self.computation_stack.pop()
                        self.initialize_graph_display()

            elif event.type == pygame.MOUSEMOTION:
                if self.selected_node:
                    scaled_pos = (event.pos[0] * SCALE, event.pos[1] * SCALE)
                    self.selected_node.x = scaled_pos[0] + self.select_offset_x
                    self.selected_node.y = scaled_pos[1] + self.select_offset_y
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.key == pygame.K_x:
                    mouse_pos = pygame.mouse.get_pos()
                    scaled_pos = (mouse_pos[0] * SCALE, mouse_pos[1] * SCALE)
                    for node in self.physics_objs:
                        if node.is_point_inside_self(scaled_pos):
                            removed = self.computation_stack[-1].remove_recursive(node.node_id)
                            for r in removed:
                                self.physics_objs.remove(r.phys)
                                # self.display_assets.remove(r.assets)

    def render(self):
        self.stage.fill(self.colorscheme.background)

        # self.screen.blit(arrow_sfc, (50, 50))
        for node in self.computation_stack[-1].nodes.values():
            self.stage.blit(node.assets.sfc, (node.phys.x, node.phys.y))

        for node in self.computation_stack[-1].nodes.values():
            for out_node in node.out_adj_list:
                self.draw_arrow(node.phys, out_node.phys)
        # screen.blit(arrow_sfc, (50, 50))

        # Draw the stage to the screen, scaled.
        scaled_stage = pygame.transform.scale(self.stage, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.screen.blit(scaled_stage, (0, 0))

        pygame.display.update()

    def draw_arrow(self, phys1: PhysicsObject, phys2: PhysicsObject):
        # 1. Find angle of line to horizontal

        phys1_c = phys1.get_center()
        phys2_c = phys2.get_center()

        phys1_candidates = [phys1.get_center_top_edge(), phys1.get_center_left_edge(), phys1.get_center_bottom_edge(),
                            phys1.get_center_right_edge()]
        phys1_norms = [vector_norm((phys2_c[0] - p[0], phys2_c[1] - p[1])) for p in phys1_candidates]
        phys1_point = phys1_candidates[argmin(phys1_norms)]

        phys2_candidates = [phys2.get_center_top_edge(), phys2.get_center_left_edge(), phys2.get_center_bottom_edge(),
                            phys2.get_center_right_edge()]
        phys2_norms = [vector_norm((phys1_c[0] - p[0], phys1_c[1] - p[1])) for p in phys2_candidates]
        phys2_point = phys2_candidates[argmin(phys2_norms)]

        dist_x = phys2_point[0] - phys1_point[0]
        dist_y = phys2_point[1] - phys1_point[1]

        if vector_norm((dist_x, dist_y)) == 0:
            return

        angle = angle_of_vector((dist_x, dist_y))

        # 2. Add or subtract 45 degrees
        angle_p45 = angle + math.pi / 4
        angle_m45 = angle - math.pi / 4
        # 3. Compute the x,y offsets using the trig circle
        x_p45_off = math.cos(angle_p45) * 10
        y_p45_off = math.sin(angle_p45) * 10
        x_m45_off = math.cos(angle_m45) * 10
        y_m45_off = math.sin(angle_m45) * 10

        arrow_lines = [phys1_point, phys2_point, (phys2_point[0] + x_p45_off, phys2_point[1] + y_p45_off),
                       (phys2_point[0] + x_m45_off, phys2_point[1] + y_m45_off), phys2_point, phys1_point]

        pygame.draw.lines(self.stage, self.colorscheme.arrow, True,
                          arrow_lines, width=2)

    def update(self):
        for node in self.computation_stack[-1].nodes.values():
            node_phys: PhysicsObject = node.phys

            # TODO: Needs to be first as acceleration is zeroed each update.
            # Should be part of update tbh, but needs high-level graph info (outnodes)
            for other in node.out_adj_list:
                if other != self:
                    gravitate(node_phys, other.phys, True)

                    # NOTE: Prevent nodes from being left of their parents
                    if other.phys.get_center()[0] < node.phys.get_center()[0]:
                        apply_force(node.phys, (-10, 0))

            update(node_phys, self.physics_objs)

            # Ensure inside window
            if not node_phys.is_inside(0, 0, SIMULATION_WIDTH, SIMULATION_HEIGHT):
                if node_phys.x < 0:
                    node_phys.x = 0
                    node_phys.vx *= -1
                if node_phys.x + node_phys.w > SIMULATION_WIDTH:
                    node_phys.x = SIMULATION_WIDTH - node_phys.w
                    node_phys.vx *= -1
                if node_phys.y < 0:
                    node_phys.y = 0
                    node_phys.vy *= -1
                if node_phys.y + node_phys.h > SIMULATION_HEIGHT:
                    node_phys.y = SIMULATION_HEIGHT - node_phys.h
                    node_phys.vy *= -1
