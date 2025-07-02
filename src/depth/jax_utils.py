import jax


def print_shapes(**arrays: jax.Array):
    for name, a in arrays.items():
        print(f"{name}: {a.shape}")
