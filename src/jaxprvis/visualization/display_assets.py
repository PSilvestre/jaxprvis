import pygame

from jaxprvis.graph.graph import Node, TensorNode, OpNode, CodeNode
from jaxprvis.visualization.physics import PhysicsObject

# OP_COLOR = (143, 188, 187) # Nord 7
#TODO customisable colorchemes
OP_COLOR = (76, 86, 106)  # Nord 3
CODE_COLOR = (180, 142, 173)  # Nord 3

TENSOR_NORMAL_COLOR = (129, 161, 193)  # Nord 9
TENSOR_INPUT_COLOR = (163, 190, 140)  # Nord 14
TENSOR_OUTPUT_COLOR = (208, 135, 112)  # Nord 12

TEXT_SIZE = 26


class DisplayAssets:
    def __init__(self, node: Node, physics_object: PhysicsObject):

        self.sfc = pygame.Surface((120, 120))
        self.sfc.set_colorkey((0, 0, 0))

        if isinstance(node, TensorNode):
            self.prerender_tensor(node, physics_object)
        elif isinstance(node, OpNode):
            self.prerender_op(node, physics_object)
        elif isinstance(node, CodeNode):
            self.prerender_code(node, physics_object)

    def prerender_tensor(self, node: TensorNode, physics_object: PhysicsObject):
        if not node.is_input and not node.is_output:
            pygame.draw.rect(self.sfc, TENSOR_NORMAL_COLOR, (0, 0, physics_object.w, physics_object.h))
        if node.is_input:
            pygame.draw.rect(self.sfc, TENSOR_INPUT_COLOR, (0, 0, physics_object.w, physics_object.h))
        if node.is_output:
            pygame.draw.rect(self.sfc, TENSOR_OUTPUT_COLOR, (0, 0, physics_object.w, physics_object.h))
        font = pygame.font.SysFont(None, TEXT_SIZE)
        text_node_sfc = font.render("{}".format(node.node_id), True, "White")
        text_node_rect = text_node_sfc.get_rect(
            center=(physics_object.w // 2, physics_object.h // 3 * 0 + physics_object.h // 3 // 2))
        text_shape_sfc = font.render("{}".format(node.shape), True, "White")
        text_shape_rect = text_shape_sfc.get_rect(
            center=(physics_object.w // 2, physics_object.h // 3 * 1 + physics_object.h // 3 // 2))
        if node.dtype is not None:
            text_dtype_sfc = font.render("{}".format(node.dtype.name), True, "White")
            text_dtype_rect = text_dtype_sfc.get_rect(
                center=(physics_object.w // 2, physics_object.h // 3 * 2 + physics_object.h // 3 // 2))
            self.sfc.blit(text_dtype_sfc, text_dtype_rect)
        self.sfc.blit(text_node_sfc, text_node_rect)
        self.sfc.blit(text_shape_sfc, text_shape_rect)

    def prerender_op(self, node: OpNode, physics_object: PhysicsObject):
        pygame.draw.circle(self.sfc, OP_COLOR, (physics_object.w // 2, physics_object.h // 2), physics_object.w // 2)
        font = pygame.font.SysFont(None, TEXT_SIZE)
        text_prim_sfc = font.render("{}".format(node.primitive), True, "White")
        text_prim_rect = text_prim_sfc.get_rect(
            center=(physics_object.w // 2, physics_object.h // 2,))
        self.sfc.blit(text_prim_sfc, text_prim_rect)

    def prerender_code(self, node: CodeNode, physics_object: PhysicsObject):
        pygame.draw.circle(self.sfc, CODE_COLOR, (physics_object.w // 2, physics_object.h // 2), physics_object.w // 2)
        font = pygame.font.SysFont(None, TEXT_SIZE)
        text_prim_sfc = font.render("{}".format(node.display_name), True, "White")
        text_prim_rect = text_prim_sfc.get_rect(
            center=(physics_object.w // 2, physics_object.h // 2))
        self.sfc.blit(text_prim_sfc, text_prim_rect)
