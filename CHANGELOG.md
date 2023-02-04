# Release notes

<!-- do not remove -->

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

