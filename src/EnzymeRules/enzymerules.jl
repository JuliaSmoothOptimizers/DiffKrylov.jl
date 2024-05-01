using Enzyme
import .EnzymeRules: forward, reverse, augmented_primal
using .EnzymeRules

export augmented_primal, reverse, forward

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
    for solver in (:bicgstab!, :gmres!)
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
                adjM = adjoint(N)
                adjN = adjoint(M)
                Krylov.$solver(
                    solver.dval,
                    adjoint(A), copy(solver.dval.x); M=adjM, N=adjN,
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
