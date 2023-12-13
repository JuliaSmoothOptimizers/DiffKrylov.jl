# DiffKrylov
[![][build-latest-img]][build-url] [![][codecov-latest-img]][codecov-latest-url]

DiffKrylov provides a differentiable API for
[Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl) using
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) and
[Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl). This is a work in progress and
eventually should enable numerical comparisons between discrete and continuous
tangent and adjoint methods (see this
[report](http://137.226.34.227/Publications/AIB/2012/2012-10.pdf)).

## Current Technical Limitations

* Only supports `gmres`, `cg`, and `bicgstab` methods
* No support for inplace methods `gmres!`, `cg!`, and `bicgstab!`
* No support for options when using Enzyme
* No support for sparse matrices using Enzyme
* No support for linear operators

## Current Open Questions
* How to handle preconditioners?
* How to set the options for the tangent/adjoint solve based on the options for the forward solve? For example `bicgtab` may return `NaN` for the tangents or adjoints.

## Installation

```julia
] add DiffKrylov
```

## Usage

Using ForwardDiff.jl, we can compute the Jacobian of `x` with respect to `b` using the ForwardDiff.jl API:

```julia
using ForwardDiff, DiffKrylov, Krylov, Random
A = rand(64,64)
b = rand(64)
J = ForwardDiff.jacobian(x -> gmres(A, x)[1], b)
```

[codecov-latest-img]: https://codecov.io/gh/JuliaSmoothOptimizers/DiffKrylov.jl/branch/main/graphs/badge.svg?branch=main
[codecov-latest-url]: https://codecov.io/github/JuliaSmoothOptimizers/DiffKrylov.jl?branch=main

[build-url]: https://github.com/JuliaSmoothOptimizers/DiffKrylov.jl/actions?query=workflow
[build-latest-img]: https://github.com/JuliaSmoothOptimizers/DiffKrylov.jl/workflows/CI/badge.svg?branch=main
