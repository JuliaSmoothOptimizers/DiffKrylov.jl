using Enzyme
import .EnzymeRules: forward, reverse, augmented_primal
using .EnzymeRules

include("get_div_grad.jl")
include("utils.jl")

@testset "$solver" for solver = (Krylov.cg, Krylov.gmres, Krylov.bicgstab)
    function sparse_laplacian(n :: Int=16; FC=Float64)
    A = get_div_grad(n, n, n)
    b = ones(n^3)
    return A, b
    end

    A, b = sparse_laplacian(4, FC=Float64)
    denseA = Matrix(A)
    fdm = central_fdm(8, 1);
    function A_one_one(x)
        _A = copy(denseA)
        _A[1,1] = x
        solver(_A,b)
    end

    function b_one(x)
        _b = copy(b)
        _b[1] = x
        solver(denseA,_b)
    end

    fda = FiniteDifferences.jacobian(fdm, a -> A_one_one(a)[1], copy(denseA[1,1]))
    fdb = FiniteDifferences.jacobian(fdm, a -> b_one(a)[1], copy(b[1]))
    fd =fda[1] + fdb[1]
    # Test forward
    ddA = Duplicated(denseA, zeros(size(denseA)))
    ddb = Duplicated(b, zeros(length(b)))
    ddA.dval[1,1] = 1.0
    ddb.dval[1] = 1.0
    ddx = Enzyme.autodiff(
        Forward,
        solver,
        ddA,
        ddb
    )
    @test isapprox(ddx[1][1], fd, atol=1e-4, rtol=1e-4)
    # Test reverse
    function driver!(x, A, b)
        x .= gmres(A,b)[1]
        nothing
    end
    ddA = Duplicated(denseA, zeros(size(denseA)))
    ddb = Duplicated(b, zeros(length(b)))
    ddx = Duplicated(zeros(length(b)), zeros(length(b)))
    ddx.dval[1] = 1.0
    Enzyme.autodiff(
        Reverse,
        driver!,
        ddx,
        ddA,
        ddb
    )

    @test isapprox(ddb.dval[1], fdb[1][1], atol=1e-4, rtol=1e-4)
    @test isapprox(ddA.dval[1,1], fda[1][1], atol=1e-4, rtol=1e-4)
end
