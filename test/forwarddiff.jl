A, b = sparse_laplacian(4, FC=Float64)
@testset "$solver" for solver = (Krylov.cg, Krylov.gmres, Krylov.bicgstab)
    x, stats = solver(A,b; atol=atol, rtol=rtol)

    # A passive, b active
    # Sparse
    @testset "A sparse passive, b active" begin
        check_jacobian(solver, A, b)
        check_values(solver, A, b)
    end
    # Dense
    @testset "A dense passive, b active" begin
        denseA = Matrix(A)
        check_jacobian(solver, denseA, b)
        check_values(solver, denseA, b)
    end

    # A active, b active
    # Sparse
    @testset "A sparse active, b active" begin
        check_derivatives_and_values_active_active(solver, A, b, x)
    end
    # Dense
    @testset "A dense active, b active" begin
        check_derivatives_and_values_active_active(solver, Matrix(A), b, x)
    end

    # A active, b passive
    # Sparse
    @testset "A sparse active, b passive" begin
        check_derivatives_and_values_active_passive(solver, A, b, x)
    end
    # Dense
    @testset "A dense active, b passive" begin
        check_derivatives_and_values_active_passive(solver, Matrix(A), b, x)
    end
end
