import jax
import jax.numpy as jnp
from jax import make_jaxpr


import jaxprvis as jvis

def visualize_while_jaxpr(iter: int):
    example_args = jnp.zeros((2, 2))

    def cond(state):
        return state[0] < iter

    def body(state):
        return state[0] + 1, state[1] ** 2

    def f(x, y):
        z = jax.lax.while_loop(cond, body, (0, x + y))
        return z[1] / y

    jaxpr = make_jaxpr(f)(example_args, example_args)
    jvis.visualize(jaxpr)


if __name__ == '__main__':
    visualize_while_jaxpr(3)
