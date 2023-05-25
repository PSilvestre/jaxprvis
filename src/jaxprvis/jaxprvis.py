from typing import Callable, Dict, Union
from jax.core import Jaxpr, ClosedJaxpr

from jaxprvis.graph.graph import Graph
from jaxprvis.graph.jaxpr_to_graph import jaxpr_to_graph
from jaxprvis.visualization.simulation import Simulation


def visualize(jaxpr: Union[Jaxpr, ClosedJaxpr], colorscheme: Union[str, Dict[str, str]] = "nord"):
    if isinstance(jaxpr, ClosedJaxpr):
        jaxpr = jaxpr.jaxpr
    g: Graph = jaxpr_to_graph(jaxpr)
    sim: Simulation = Simulation(g)
    sim.start()