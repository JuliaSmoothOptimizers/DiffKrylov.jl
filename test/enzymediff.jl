using Enzyme
import .EnzymeRules: forward, reverse, augmented_primal
using .EnzymeRules
using DiffKrylov
using LinearAlgebra
using FiniteDifferences
using Krylov
using Random
using SparseArrays
using Test

Random.seed!(1)
include("create_matrix.jl")
@testset "Enzyme Rules" begin
    @testset "$MT" for MT = (Matrix, SparseMatrixCSC)
        @testset "($M, $N)" for (M,N) = ((I,I),)
            # Square unsymmetric solvers
            @testset "$solver" for solver = (Krylov.gmres, Krylov.bicgstab)
                A = []
                if MT == Matrix
                    A = create_unsymmetric_matrix(10)
                    b = rand(10)
                else
                    A, b = sparse_laplacian(4, FC=Float64)
                end
                test_enzyme_with(solver, A, b, M, N)
            end
            # Square symmetric solvers
            @testset "$solver" for solver = (Krylov.cg,)
                A, b = sparse_laplacian(4, FC=Float64)
                A = MT(A)
                test_enzyme_with(solver, A, b, M, N)
            end
        end
    end
end
