using Krylov
using DiffKrylov
using Test
using LinearAlgebra
using SparseArrays
using ForwardDiff
import ForwardDiff: Dual, Partials, partials, value
using FiniteDifferences

# Sparse Laplacian.
include("get_div_grad.jl")
function sparse_laplacian(n :: Int=16; FC=Float64)
  A = get_div_grad(n, n, n)
  b = ones(n^3)
  return A, b
end
A, b = sparse_laplacian(4, FC=Float64)

x, stats = cg(A,b)

adJ = ForwardDiff.jacobian(x -> cg(A, x)[1], b)
fdm = central_fdm(8, 1);
fdJ = FiniteDifferences.jacobian(fdm, x -> cg(A, x)[1], b)
all(isapprox.(adJ, fdJ[1], atol=1e-4))


denseA = Matrix(A)
adJ = ForwardDiff.jacobian(x -> cg(denseA, x)[1], b)
fdm = central_fdm(8, 1);
fdJ = FiniteDifferences.jacobian(fdm, x -> cg(denseA, x)[1], b)

all(isapprox.(adJ, fdJ[1], atol=1e-4))
x, stats = cg(A,b)

ForwardDiff.jacobian(x -> cg(A, x)[1], b)



function Krylov.cg(_A::Matrix{Dual{T, V, N}}, _b::Vector{Dual{T, V, N}}; options...) where {T, V, N}
  A = Matrix{Float64}(value.(_A))
  dAs = Vector{Matrix{Float64}}(undef, N)
  for i in 1:N
    dAs[i] = Matrix(partials.(_A, i))
  end
  m = length(b)
  b = value.(_b)
  dbs = Vector{Vector{Float64}}(undef, N)
  for i in 1:N
    dbs[i] = partials.(_b, i)
  end
  # db = partials.(_b,1)
  x, stats = cg(A,b)
  n = length(x)
  dxs = Matrix{Float64}(undef, n, N)
  px = Vector{Partials{N,V}}(undef, n)
  for i in 1:N
    nb = dbs[i] - dAs[i]*x
    dx, dstats = cg(dAs[i],nb)
    dxs[:, i] = dx
  end
  for i in 1:n
    px[i] = Partials{N,V}(Tuple(dxs[i,j] for j in 1:N))
  end
  duals = Dual{T,V,N}.(x, px)
  return (duals, stats)
end

testfwd()

x, stats = cg(A,b)


dA = ForwardDiff.Dual.(A, A)
db = ForwardDiff.Dual.(b, b)

dx, stats = cg(dA,db)

using Krylov
using ChainRulesCore
using DiffResults
using ForwardDiff
using Krylov
using Test
using LinearAlgebra
using SparseArrays
using ForwardDiff
import ForwardDiff: Dual, Partials, partials, value

function Krylov.cg(_A::Matrix{Dual{T, V, N}}, _b::Vector{Dual{T, V, N}}; options...) where {T, V, N}
    A = Matrix{Float64}(value.(_A))
    dAs = Vector{Matrix{Float64}}(undef, N)
    for i in 1:N
        dAs[i] = Matrix(partials.(_A, i))
    end
    b = value.(_b)
    m = length(b)
    dbs = Matrix{V}(undef, m, N)
    for i in 1:m
        dbs[i,:] = partials(_b[i])
    end
    x, stats = cg(A,b)
    m = length(x)
    dxs = Matrix{V}(undef, m, N)
    px = Vector{Partials{N,V}}(undef, m)
    for i in 1:N
        nb = dbs[:,i] - dAs[i]*x
        @show nb
        dx, dstats = cg(dAs[i],nb)
        dxs[:, i] = dx
    end
    for i in 1:m
        px[i] = Partials{N,V}(Tuple(dxs[i,j] for j in 1:N))
    end
    @show px
    duals = Dual{T,V,N}.(x, px)
    return (duals, stats)
end
