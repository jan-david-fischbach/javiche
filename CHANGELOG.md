# Release notes

<!-- do not remove -->

## 0.0.6

### New Features

- Consider using `autograd.make_vjp()` instead of `ceviche.jacobian()` ([#1](https://github.com/Jan-David-Black/javiche/issues/1))
  - Hey there, cool package! I wanted to suggest an alternative implementation that may be more efficient.

It looks like `javiche` is currently using `ceviche`'s `jacobian()` method, which (I think) may be less efficient if `@javiche.jaxit` is ever applied to a function whose output is not scalar-valued. The reason is that `ceviche`'s `jacobian()` method loops over the output basis vectors to construct the Jacobian, and you typically don't need the explicit Jacobian when calculating VJPs. An example of where this would matter is if you just applied `@jaxit` to a function that returns the field distribution, but performed the loss function calculation in terms of the field distribution in JAX, calling `jax.grad` or `jax.value_and_grad` on the combination.

Below is a sketch of a more direct approach that maps autograd's `make_vjp()` function to JAX's VJP mechanism.

Given an autograd function, `f_ag(*args) -> np.ndarray` we can wrap it into a function, `f(*args) -> jnp.ndarray`. This is not ceviche-specific; it can be used for any autograd function with multiple inputs (`*args`) and a single array output, though it could be generalized to support multiple array outputs as well.

```python
import jax
import jax.numpy as jnp
import numpy as np
import autograd


def as_numpy(x):
  def as_numpy_map(a):
    if isinstance(a, jnp.ndarray):
      return np.asarray(a)
    else:
      return a
  return jax.tree_util.tree_map(as_numpy_map, x)


def as_jax(x):
  def as_jax_map(a):
    if isinstance(a, np.ndarray):
      return jnp.asarray(a)
    else:
      return a
  return jax.tree_util.tree_map(as_jax_map, x)


@jax.custom_vjp
def f(*args):
  return as_jax(f_ag(*as_numpy(args)))


def f_fwd(*args):
  args = as_numpy(args)
  argnums = tuple(i for i, _ in enumerate(args))

  def f_ag_tupled(*args):
    ans = f_ag(*args)
    if isinstance(ans, tuple):
      return autograd.builtins.tuple(ans)
    else:
      return ans

  vjp_f, ans = autograd.make_vjp(f_ag_tupled, argnums)(*args)
  return as_jax(ans), jax.tree_util.Partial(vjp_f)


def f_rev(vjp_f, g):
  g = as_numpy(g)
  return as_jax(vjp_f(g))


f.defvjp(f_fwd, f_rev)
```

I was thinking about adding something like this to [ceviche_challenges](https://github.com/google/ceviche-challenges), but had not gotten around to it.

## 0.0.5

Added caching suport for functions that take only hashable inputs (or numpy/JAX arrays)

## 0.0.4

Avoid downgrading numpy by manually patching the pypi version of ceviche.

## 0.0.3

As github hosted version of packages cannot be requirements for pypi packages we instead downgrade numpy. Additionally a patched version of ceviche.viz is used instead to avoid errors.


## 0.0.2

The requirements were updated to point to the patched version of ceviche, that is compatible to python 3.10


## 0.0.1

This is the initial release of `javiche`, thus there is no Changelog yet

