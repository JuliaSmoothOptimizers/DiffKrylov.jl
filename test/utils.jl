# Sparse Laplacian.
include("get_div_grad.jl")
function sparse_laplacian(n :: Int=16; FC=Float64)
  A = get_div_grad(n, n, n)
  b = ones(n^3)
  return A, b
end

function check(A,b)
  tA, tb = sparse_laplacian(4, FC=Float64)
  @test all(value.(tb) .== b)
  @test all(value.(tA) .== A)
end

function check_values(solver, A, b)
  x = solver(A,b; atol=atol, rtol=rtol)[1]
  db = Dual.(b)
  dx = solver(A,db; atol=atol, rtol=rtol)[1]
  @test all(dx .== x)
end

function check_jacobian(solver, A, b)
  adJ = ForwardDiff.jacobian(x -> solver(A, x; atol=atol, rtol=rtol)[1], b)
  fdm = central_fdm(8, 1);
  fdJ = FiniteDifferences.jacobian(fdm, x -> solver(A, x; atol=atol, rtol=rtol)[1], copy(b))
  @test all(isapprox.(adJ, fdJ[1]))
end

function check_derivatives_and_values_active_active(solver, A, b, x)
    fdm = central_fdm(8, 1);
    dualsA = copy(A)
    fill!(dualsA, 0.0)
    dualsA[1,1] = 1.0
    dA = ForwardDiff.Dual.(A, dualsA)
    check(A,b)

    dualsb = copy(b)
    fill!(dualsb, 0.0)
    dualsb[1] = 1.0
    db = ForwardDiff.Dual.(b, dualsb)
    dx, stats = solver(dA,db; atol=atol, rtol=rtol)

    all(isapprox(value.(dx), x))

    function A_one_one(x)
        _A = copy(A)
        _A[1,1] = x
        solver(_A,b; atol=atol, rtol=rtol)
    end

    function b_one(x)
        _b = copy(b)
        _b[1] = x
        solver(A,_b; atol=atol, rtol=rtol)
    end

    fda = FiniteDifferences.jacobian(fdm, a -> A_one_one(a)[1], copy(A[1,1]))
    fdb = FiniteDifferences.jacobian(fdm, a -> b_one(a)[1], copy(b[1]))
    isapprox(value.(dx), x)
    fd =fda[1] + fdb[1]
    @test isapprox(partials.(dx,1), fd)
end

function check_derivatives_and_values_active_passive(solver, A, b, x)
    fdm = central_fdm(8, 1);
    dualsA = copy(A)
    fill!(dualsA, 0.0)
    dualsA[1,1] = 1.0
    dA = ForwardDiff.Dual.(A, dualsA)
    check(A,b)

    dx, stats = solver(dA,b; atol=atol, rtol=rtol)

    all(isapprox(value.(dx), x))

    function A_one_one(x)
        _A = copy(A)
        _A[1,1] = x
        solver(_A,b; atol=atol, rtol=rtol)
    end

    fda = FiniteDifferences.jacobian(fdm, a -> A_one_one(a)[1], copy(A[1,1]))
    isapprox(value.(dx), x)
    @test isapprox(partials.(dx,1), fda[1])
end

struct GMRES end
struct BICGSTAB end
struct CG end

function driver!(::GMRES, x, A, b, M, N, ldiv=false)
    x .= gmres(A,b, atol=1e-16, rtol=1e-16, M=M, N=N, verbose=0, ldiv=ldiv)[1]
    nothing
end

function driver!(::BICGSTAB, x, A, b, M, N, ldiv=false)
    x .= bicgstab(A,b, atol=1e-16, rtol=1e-16, M=M, N=N, verbose=0, ldiv=ldiv)[1]
    nothing
end

function driver!(::CG, x, A, b, M, N, ldiv=false)
    x .= cg(A,b, atol=1e-16, rtol=1e-16, M=M, verbose=0, ldiv=ldiv)[1]
    nothing
end

function test_enzyme_with(solver, A, b, M, N, ldiv=false)
    tsolver = if solver == Krylov.cg
        CG()
    elseif solver == Krylov.gmres
        GMRES()
    elseif solver == Krylov.bicgstab
        BICGSTAB()
    else
        error("Unsupported solver $solver is tested in DiffKrylov.jl")
    end
    fdm = central_fdm(8, 1);
    function A_one_one(hx)
        _A = copy(A)
        _A[1,1] = hx
        x = zeros(length(b))
        driver!(tsolver, x, _A, b, M, N, ldiv)
        return x
    end

    function b_one(hx)
        _b = copy(b)
        _b[1] = hx
        x = zeros(length(b))
        driver!(tsolver, x, A, _b, M, N, ldiv)
        return x
    end

    fda = FiniteDifferences.jacobian(fdm, a -> A_one_one(a), copy(A[1,1]))
    fdb = FiniteDifferences.jacobian(fdm, a -> b_one(a), copy(b[1]))
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
    dx = Duplicated(zeros(length(b)), zeros(length(b)))
    dA.dval[1,1] = 1.0
    db.dval[1] = 1.0
    Enzyme.autodiff(
        Forward,
        driver!,
        Const(tsolver),
        dx,
        dA,
        db,
        Const(M),
        Const(N),
        Const(ldiv)
    )
    @test isapprox(dx.dval, fd, atol=1e-4, rtol=1e-4)
    # Test reverse
    dA = Duplicated(A, duplicate(A))
    db = Duplicated(b, zeros(length(b)))
    dx = Duplicated(zeros(length(b)), zeros(length(b)))
    dx.dval[1] = 1.0
    Enzyme.autodiff(
        Reverse,
        driver!,
        Const(tsolver),
        dx,
        dA,
        db,
        Const(M),
        Const(N),
        Const(ldiv)
    )
    @test isapprox(db.dval[1], fdb[1][1], atol=1e-4, rtol=1e-4)
    @test isapprox(dA.dval[1,1], fda[1][1], atol=1e-4, rtol=1e-4)
end
