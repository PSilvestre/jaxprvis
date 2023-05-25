import itertools as it
import string
from collections import defaultdict
from typing import Dict

from jax.core import Var, JaxprEqn, Jaxpr, ClosedJaxpr, Literal

from jaxprvis.graph.graph import Graph, TensorNode, OpNode, CodeNode


def add_var(names: Dict[Var, str], v: Var, g: Graph):
    g.add_node(TensorNode(binder_name(v, names), v.aval))


def add_lit(lit: Literal, g: Graph):
    g.add_node(TensorNode(binder_name(lit), aval=lit.aval, literal=lit.val))


def add_binder(g, names, ob):
    if binder_name(ob, names) not in g.nodes.keys():
        if isinstance(ob, Var):
            add_var(names, ob, g)
        elif isinstance(ob, Literal):
            add_lit(ob, g)


def binder_name(binder, names=None):
    if isinstance(binder, Var):
        return names[binder]
    elif isinstance(binder, Literal):
        return "L" + str(binder.val)


def add_eqn(names: Dict[Var, str], eqn: JaxprEqn, g: Graph):
    for ob in eqn.outvars:
        add_binder(g, names, ob)

    op_id = ",".join([binder_name(inp, names) for inp in
                      eqn.invars]) + "-" + eqn.primitive.name + "-" + ",".join(
        [binder_name(out, names) for out in eqn.outvars])

    op = OpNode(op_id, eqn.primitive.name, other=str(eqn.params))
    g.add_node(op)

    process_inner_jaxprs(eqn, g, op_id)

    for ib in eqn.invars:
        add_binder(g, names, ib)
        g.add_edge_by_id((binder_name(ib, names), op_id))

    for ob in eqn.outvars:
        add_binder(g, names, ob)
        g.add_edge_by_id((op_id, binder_name(ob, names)))


def process_inner_jaxprs(eqn, g, op_id):
    for inner_jaxpr_key in ["cond_jaxpr", "body_jaxpr", "inner", "inner_jaxpr", "jaxpr", "call_jaxpr", "fun_jaxpr"]:
        if inner_jaxpr_key in eqn.params.keys():
            inner: Jaxpr = eqn.params[inner_jaxpr_key]
            if isinstance(inner, ClosedJaxpr):
                inner = inner.jaxpr
            display_name = "Inner"
            if inner_jaxpr_key == "cond_jaxpr":
                display_name = "Cond"
            elif inner_jaxpr_key == "body_jaxpr":
                display_name = "Body"
            code_node_id = "{}-{}".format(op_id, inner_jaxpr_key)
            print("Entering a " + inner_jaxpr_key)
            code_node: CodeNode = CodeNode(code_node_id, display_name, jaxpr_to_graph(inner))
            g.add_node(code_node)
            g.add_edge_by_id((code_node_id, op_id))


def jaxpr_to_graph(jaxpr: Jaxpr) -> Graph:
    g = Graph()
    idgen = (''.join(s) for r in it.count(1) for s in it.permutations(string.ascii_lowercase, r))
    names = defaultdict(lambda: next(idgen))

    for inb in jaxpr.invars:
        print("Adding binder: {}".format(str(inb)))
        add_binder(g, names, inb)

    for eqn in jaxpr.eqns:
        add_eqn(names, eqn, g)

    for inb in jaxpr.invars:
        id = binder_name(inb, names)
        g.nodes[id].is_input = True
        g.inputs.append(g.nodes[id])

    for ob in jaxpr.outvars:
        id = binder_name(ob, names)
        if id in g.nodes:
            g.nodes[id].is_output = True
            g.outputs.append(g.nodes[id])

    return g
