using Enzyme
import .EnzymeRules: forward, reverse, augmented_primal
using .EnzymeRules

@testset "$solver" for solver = (Krylov.cg, Krylov.gmres, Krylov.bicgstab)
    @testset "$MT" for MT = (Matrix, SparseMatrixCSC)
        A, b = sparse_laplacian(4, FC=Float64)
        A = MT(A)
        fdm = central_fdm(8, 1);
        function A_one_one(x)
            _A = copy(A)
            _A[1,1] = x
            solver(_A,b)
        end

        function b_one(x)
            _b = copy(b)
            _b[1] = x
            solver(A,_b)
        end

        fda = FiniteDifferences.jacobian(fdm, a -> A_one_one(a)[1], copy(A[1,1]))
        fdb = FiniteDifferences.jacobian(fdm, a -> b_one(a)[1], copy(b[1]))
        fd =fda[1] + fdb[1]
        # Test forward
        function duplicate(A::SparseMatrixCSC)
            dA = copy(A)
            fill!(dA.nzval, zero(eltype(A)))
            return dA
        end
        duplicate(A::Matrix) = zeros(size(A))

        dA = Duplicated(A, duplicate(A))
        db = Duplicated(b, zeros(length(b)))
        dA.dval[1,1] = 1.0
        db.dval[1] = 1.0
        dx = Enzyme.autodiff(
            Forward,
            solver,
            dA,
            db
        )
        @test isapprox(dx[1][1], fd, atol=1e-4, rtol=1e-4)
        # Test reverse
        function driver!(x, A, b)
            x .= gmres(A,b)[1]
            nothing
        end
        dA = Duplicated(A, duplicate(A))
        db = Duplicated(b, zeros(length(b)))
        dx = Duplicated(zeros(length(b)), zeros(length(b)))
        dx.dval[1] = 1.0
        Enzyme.autodiff(
            Reverse,
            driver!,
            dx,
            dA,
            db
        )

        @test isapprox(db.dval[1], fdb[1][1], atol=1e-4, rtol=1e-4)
        @test isapprox(dA.dval[1,1], fda[1][1], atol=1e-4, rtol=1e-4)
    end
end
