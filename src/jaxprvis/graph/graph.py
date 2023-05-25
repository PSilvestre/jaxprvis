from typing import Tuple, Dict, List, Any

from jax.core import AbstractValue, ShapedArray


#TODO move this to networkx probably

class Node:
    def __init__(self, node_id: str):
        super().__init__()
        self.node_id: str = node_id
        self.out_adj_list: List[Node] = []

        self.phys: 'PhysicsObject' = None
        self.assets: 'DisplayAssets' = None

    def __hash__(self):
        hash(self.node_id)

    def prerender(self):
        raise Exception("Unimplemented")


class CodeNode(Node):
    def __init__(self, node_id: str, display_name: str, jaxpr: 'Graph'):
        super().__init__(node_id)
        self.display_name: str = display_name
        self.jaxpr: 'Graph' = jaxpr


class TensorNode(Node):
    def __init__(self, node_id: str, aval: AbstractValue, is_input: bool = False, is_output: bool = False, literal: Any = None):
        super().__init__(node_id)


        # TODO: need better handling here
        if isinstance(aval, ShapedArray):
            self.shape = aval.shape
            self.dtype = aval.dtype
        #elif aval == abstract_unit:
        #    self.shape = ()
        #    self.dtype = None

        self.is_input = is_input
        self.is_output = is_output
        self.literal = literal


class OpNode(Node):
    def __init__(self, node_id: str, primitive: str, other: str = None):
        super().__init__(node_id)
        self.primitive = primitive
        self.other = other


class Graph:
    def __init__(self, nodes: Dict[str, Node] = None):
        if nodes is None:
            nodes = {}
        self.nodes = nodes
        self.inputs = []
        self.outputs = []

    def edges(self):
        return self.findedges()

    def add_edge(self, edge: Tuple[Node, Node]):
        (node1, node2) = edge
        self.add_node(node1)
        self.add_node(node2)
        node1.out_adj_list.append(node2)

    def add_edge_by_id(self, edge: Tuple[str, str]):
        node1 = self.nodes[edge[0]]
        node2 = self.nodes[edge[1]]
        if node1 is None or node2 is None:
            raise Exception("tried to add edge with non existant node")
        node1.out_adj_list.append(node2)

    def add_node(self, node: Node):
        if node.node_id not in self.nodes:
            self.nodes[node.node_id] = node

    def get_node_by_id(self, idx: str):
        return self.nodes[idx]

    def findedges(self):
        edgename = []
        for vrtx in self.nodes:
            for nxtvrtx in self.nodes[vrtx].out_adj_list:
                if {nxtvrtx, vrtx} not in edgename:
                    edgename.append({vrtx, nxtvrtx})

        return edgename

    def remove_recursive(self, node_id: str):
        node = self.nodes.pop(node_id)
        removed = [node]

        for other in node.out_adj_list:
            removed.extend(self.remove_recursive(other.node_id))

        for remaining in self.nodes.values():
            if node in remaining.out_adj_list:
               remaining.out_adj_list.remove(node)

        return removed


