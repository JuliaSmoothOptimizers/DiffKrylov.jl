using Krylov
using DiffKrylov
using Test
using LinearAlgebra
using SparseArrays
using ForwardDiff
import ForwardDiff: Dual, Partials, partials, value
using FiniteDifferences

include("get_div_grad.jl")
include("utils.jl")

# Sparse Laplacian.
function sparse_laplacian(n :: Int=16; FC=Float64)
  A = get_div_grad(n, n, n)
  b = ones(n^3)
  return A, b
end

atol = 1e-12
rtol = 0.0
@testset "DiffKrylov" begin
    @testset "ForwardDiff" begin
        include("forwarddiff.jl")
    end
    @testset "Enzyme" begin
        include("enzymediff.jl")
    end
end
