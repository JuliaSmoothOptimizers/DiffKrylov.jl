using Krylov
using DiffKrylov
using Test
using LinearAlgebra
using SparseArrays
using ForwardDiff
import ForwardDiff: Dual, Partials, partials, value
using FiniteDifferences

atol = 1e-12
rtol = 0.0
@testset "DiffKrylov" begin
    @testset "ForwardDiff" begin
        include("forwarddiff.jl")
    end
end
