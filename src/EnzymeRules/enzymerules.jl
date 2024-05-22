using Enzyme
import .EnzymeRules: forward, reverse, augmented_primal
using .EnzymeRules

export augmented_primal, reverse, forward

function custom_stopping_condition(solver::BicgstabSolver, A, b, r, tol)
    mul!(r, A, solver.x)
    r .-= b               # r := b - Ax
    normr = norm(r)
    bool = normr ≤ tol # tolerance based on the 2-norm of the residual
    if isnan(normr) || isinf(normr)
        bool = true
    end
    return bool
end


function custom_stopping_condition(solver::GmresSolver, A, b, r, tol)
    z = solver.z
    k = solver.inner_iter
    nr = sum(1:k)
    V = solver.V
    R = solver.R
    y = copy(z)

    # Solve Rk * yk = zk
    for i = k : -1 : 1
        pos = nr + i - k
        for j = k : -1 : i+1
        y[i] = y[i] - R[pos] * y[j]
        pos = pos - j + 1
        end
        y[i] = y[i] / R[pos]
    end

    # xk = Vk * yk
    xk = sum(V[i] * y[i] for i = 1:k)
    mul!(r, A, xk)
    r .-= b               # r := b - Ax
    normr = norm(r)
    if k == 30
        println("normr=$normr, k=$k, iter=$(solver.inner_iter)")
    end
    bool = normr ≤ tol # tolerance based on the 2-norm of the residual
    if isnan(normr) || isinf(normr)
        bool = true
    end
    return bool
end

for AMT in (:Matrix, :SparseMatrixCSC)
    for solver in (:bicgstab!, :gmres!)
        @eval begin
            function forward(
                func::Const{typeof(Krylov.$solver)},
                ret::Type{RT},
                solver::Annotation{ST},
                _A::Annotation{MT},
                _b::Annotation{VT};
                M = I,
                N = I,
                verbose = 0,
                options...
            ) where {RT <: Annotation, ST <: Krylov.KrylovSolver, MT <: $AMT, VT <: Vector}
                psolver = $solver
                pamt = $AMT
                if verbose > 0
                    @info "($psolver, $pamt) forward rule"
                end
                A = _A.val
                b = _b.val
                Krylov.$solver(solver.val, A,b; M=M, N=N, verbose=verbose, options...)
                if isa(_A, Duplicated) && isa(_b, Duplicated)
                    dA = _A.dval
                    db = _b.dval
                    db -= dA*solver.val.x
                    Krylov.$solver(solver.dval,A,db; M=M, N=N, verbose=verbose, options...)
                elseif isa(_A, Duplicated) && isa(_b, Const)
                    dA = _A.dval
                    db = -dA*x
                    Krylov.$solver(solver.dval,A,db; M=M, N=N, verbose=verbose, options...)
                elseif isa(_A, Const) && isa(_b, Duplicated)
                    db = _b.dval
                    Krylov.$solver(solver.dval,A,db; M=M, N=N, verbose=verbose, options...)
                elseif isa(_A, Const) && isa(_b, Const)
                    nothing
                else
                    error("Error in Krylov forward rule: $(typeof(_A)), $(typeof(_b))")
                end
                if RT <: Const
                    return solver.val
                else
                    return solver
                end
            end
        end
    end
    for solver in (:cg!,)
        @eval begin
            function forward(
                func::Const{typeof(Krylov.$solver)},
                ret::Type{RT},
                solver::Annotation{ST},
                _A::Annotation{MT},
                _b::Annotation{VT};
                verbose = 0,
                M = I,
                options...
            ) where {RT <: Annotation, ST <: Krylov.KrylovSolver, MT <: $AMT, VT <: Vector}
                psolver = $solver
                pamt = $AMT
                if verbose > 0
                    @info "($psolver, $pamt) forward rule"
                end
                A = _A.val
                b = _b.val
                Krylov.$solver(solver.val,A,b; M=M, verbose=verbose, options...)
                if isa(_A, Duplicated) && isa(_b, Duplicated)
                    dA = _A.dval
                    db = _b.dval
                    db -= dA*solver.val.x
                    Krylov.$solver(solver.dval,A,db; M=M, verbose=verbose, options...)
                elseif isa(_A, Duplicated) && isa(_b, Const)
                    dA = _A.dval
                    db = -dA*solver.val.x
                    Krylov.$solver(solver.dval,A,db; M=M, verbose=verbose, options...)
                elseif isa(_A, Const) && isa(_b, Duplicated)
                    db = _b.dval
                    Krylov.$solver(solver.dval,A,db; M=M, verbose=verbose, options...)
                elseif isa(_A, Const) && isa(_b, Const)
                    nothing
                else
                    error("Error in Krylov forward rule: $(typeof(_A)), $(typeof(_b))")
                end
                if RT <: Const
                    return solver.val
                else
                    return solver
                end
            end
        end
    end
end


for AMT in (:Matrix, :SparseMatrixCSC)
    for solver in (:bicgstab!, :gmres!, :bilq!, :qmr!)
        @eval begin
            function augmented_primal(
                config,
                func::Const{typeof(Krylov.$solver)},
                ret::Type{<:Annotation},
                solver::Annotation{ST},
                A::Annotation{MT},
                b::Annotation{VT};
                M=I,
                N=I,
                options...
            ) where {ST <: Krylov.KrylovSolver, MT <: $AMT, VT <: Vector}
                psolver = $solver
                pamt = $AMT
                verbose = options[:verbose]
                if verbose > 0
                    @info "($psolver, $pamt) augmented forward"
                end
                r = similar(b.val)
                # krylov_callback(_solver) = custom_stopping_condition(_solver, A.val, b.val, r, options[:atol])
                Krylov.$solver(
                    solver.val, A.val, b.val;
                    options...,
                    ldiv=true,
                    M=M, N=N,
                    # callback=krylov_callback,
                    verbose=verbose,
                    # atol = eps(Float64), rtol = eps(Float64),
                )

                cache = (solver.val.x, A.val, verbose,M,N)
                return AugmentedReturn(nothing, nothing, cache)
            end

            function reverse(
                config,
                ::Const{typeof(Krylov.$solver)},
                dret::Type{RT},
                cache,
                solver::Annotation{ST},
                _A::Annotation{MT},
                _b::Annotation{VT};
                options...
            ) where {ST <: Krylov.KrylovSolver, MT <: $AMT, VT <: Vector, RT}
                (x, A, verbose,M,N) = cache
                psolver = $solver
                pamt = $AMT
                if verbose > 0
                    @info "($psolver, $pamt) reverse"
                end
                adjM = adjoint(M)
                adjN = adjoint(N)
                r = similar(solver.dval.x)
                b = deepcopy(solver.dval.x)
                # krylov_callback(_solver) = custom_stopping_condition(_solver, adjoint(A), b, r, options[:atol])
                Krylov.$solver(
                    solver.dval,
                    adjoint(A), b;
                    options...,
                    ldiv=true,
                    verbose=options[:verbose],
                    M=adjM, N=adjN,
                    # atol = eps(Float64), rtol = eps(Float64),
                    # callback=krylov_callback,
                )
                copyto!(_b.dval, solver.dval.x)
                if isa(_A, Duplicated)
                    _A.dval .= -x .* _b.dval'
                end
                return (nothing, nothing, nothing)
            end
        end
    end
    for solver in (:cg!,)
        @eval begin
            function augmented_primal(
                config,
                func::Const{typeof(Krylov.$solver)},
                ret::Type{<:Annotation},
                solver::Annotation{ST},
                A::Annotation{MT},
                b::Annotation{VT};
                M=I,
                verbose=0,
                options...
            ) where {ST <: Krylov.KrylovSolver, MT <: $AMT, VT <: Vector}
                psolver = $solver
                pamt = $AMT
                if verbose > 0
                    @info "($psolver, $pamt) augmented forward"
                end
                Krylov.$solver(
                    solver.val, A.val,b.val;
                    M=M, verbose=verbose, options...
                )
                cache = (solver.val.x, A.val,verbose,M)
                return AugmentedReturn(nothing, nothing, cache)
            end

            function reverse(
                config,
                ::Const{typeof(Krylov.$solver)},
                dret::Type{RT},
                cache,
                solver::Annotation{ST},
                _A::Annotation{MT},
                _b::Annotation{VT};
                options...
            ) where {ST <: Krylov.KrylovSolver, MT <: $AMT, VT <: Vector, RT}
                (x, A, verbose,M) = cache
                psolver = $solver
                pamt = $AMT
                if verbose > 0
                    @info "($psolver, $pamt) reverse"
                end
                Krylov.$solver(
                    solver.dval,
                    A, copy(solver.dval.x); M=M,
                    verbose=verbose, options...
                )
                copyto!(_b.dval, solver.dval.x)
                if isa(_A, Duplicated)
                    _A.dval .= -x .* _b.dval'
                end
                return (nothing, nothing, nothing)
            end
        end
    end
end
