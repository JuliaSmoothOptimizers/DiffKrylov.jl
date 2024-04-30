using Enzyme
import .EnzymeRules: forward, reverse, augmented_primal
using .EnzymeRules

export augmented_primal, reverse, forward

for AMT in (:Matrix, :SparseMatrixCSC)
    for solver in (:bicgstab, :gmres)
        @eval begin
            function forward(
                func::Const{typeof(Krylov.$solver)},
                ret::Type{RT},
                _A::Annotation{MT},
                _b::Annotation{VT};
                M = I,
                N = I,
                verbose = 0,
                options...
            ) where {RT, MT <: $AMT, VT <: Vector}
                psolver = $solver
                pamt = $AMT
                if verbose > 0
                    @info "($psolver, $pamt) forward rule"
                end
                A = _A.val
                b = _b.val
                dx = []
                x, stats = Krylov.$solver(A,b; M=M, N=N, verbose=verbose, options...)
                if isa(_A, Duplicated) && isa(_b, Duplicated)
                    dA = _A.dval
                    db = _b.dval
                    db -= dA*x
                    dx, dstats = Krylov.$solver(A,db; M=M, N=N, verbose=verbose, options...)
                elseif isa(_A, Duplicated) && isa(_b, Const)
                    dA = _A.dval
                    db = -dA*x
                    dx, dstats = Krylov.$solver(A,db; M=M, N=N, verbose=verbose, options...)
                elseif isa(_A, Const) && isa(_b, Duplicated)
                    db = _b.dval
                    dx, dstats = Krylov.$solver(A,db; M=M, N=N, verbose=verbose, options...)
                elseif isa(_A, Const) && isa(_b, Const)
                    nothing
                else
                    error("Error in Krylov forward rule: $(typeof(_A)), $(typeof(_b))")
                end

                if RT <: Const
                    return (x, stats)
                elseif RT <: DuplicatedNoNeed
                    return (dx, stats)
                else
                    return Duplicated((x, stats), (dx, dstats))
                end
            end
        end
    end
    for solver in (:cg,)
        @eval begin
            function forward(
                func::Const{typeof(Krylov.$solver)},
                ret::Type{RT},
                _A::Annotation{MT},
                _b::Annotation{VT};
                verbose = 0,
                M = I,
                options...
            ) where {RT, MT <: $AMT, VT <: Vector}
                psolver = $solver
                pamt = $AMT
                if verbose > 0
                    @info "($psolver, $pamt) forward rule"
                end
                A = _A.val
                b = _b.val
                dx = []
                x, stats = Krylov.$solver(A,b; M=M, verbose=verbose, options...)
                if isa(_A, Duplicated) && isa(_b, Duplicated)
                    dA = _A.dval
                    db = _b.dval
                    db -= dA*x
                    dx, dstats = Krylov.$solver(A,db; M=M, verbose=verbose, options...)
                elseif isa(_A, Duplicated) && isa(_b, Const)
                    dA = _A.dval
                    db = -dA*x
                    dx, dstats = Krylov.$solver(A,db; M=M, verbose=verbose, options...)
                elseif isa(_A, Const) && isa(_b, Duplicated)
                    db = _b.dval
                    dx, dstats = Krylov.$solver(A,db; M=M, verbose=verbose, options...)
                elseif isa(_A, Const) && isa(_b, Const)
                    nothing
                else
                    error("Error in Krylov forward rule: $(typeof(_A)), $(typeof(_b))")
                end

                if RT <: Const
                    return (x, stats)
                elseif RT <: DuplicatedNoNeed
                    return (dx, stats)
                else
                    return Duplicated((x, stats), (dx, dstats))
                end
            end
        end
    end
end


for AMT in (:Matrix, :SparseMatrixCSC)
    for solver in (:bicgstab, :gmres)
        @eval begin
            function augmented_primal(
                config,
                func::Const{typeof(Krylov.$solver)},
                ret::Type{RT},
                _A::Annotation{MT},
                _b::Annotation{VT};
                M=I,
                N=I,
                verbose=0,
                options...
            ) where {RT, MT <: $AMT, VT <: Vector}
                psolver = $solver
                pamt = $AMT
                if verbose > 0
                    @info "($psolver, $pamt) augmented forward"
                end
                A = _A.val
                b = _b.val
                x, stats = Krylov.$solver(A,b; M=M, N=N, verbose=verbose, options...)
                bx = zeros(length(x))
                bstats = deepcopy(stats)
                if needs_primal(config)
                    return AugmentedReturn(
                        (x, stats),
                        (bx, bstats),
                        (A,x, Ref(bx), verbose, M, N)
                    )
                else
                    return AugmentedReturn(nothing, (bx, bstats), (A,x))
                end
            end

            function reverse(
                config,
                ::Const{typeof(Krylov.$solver)},
                dret::Type{RT},
                cache,
                _A::Annotation{MT},
                _b::Annotation{<:Vector};
                options...
            ) where {RT, MT <: $AMT}
                (A,x,bx,verbose,M,N) = cache
                psolver = $solver
                pamt = $AMT
                if verbose > 0
                    @info "($psolver, $pamt) reverse"
                end
                adjM = adjoint(N)
                adjN = adjoint(M)
                _b.dval .= Krylov.$solver(adjoint(A), bx[]; M=adjM, N=adjN, verbose=verbose, options...)[1]
                if isa(_A, Duplicated)
                    _A.dval .= -x .* _b.dval'
                end
                return (nothing, nothing)
            end
        end
    end
    for solver in (:cg,)
        @eval begin
            function augmented_primal(
                config,
                func::Const{typeof(Krylov.$solver)},
                ret::Type{RT},
                _A::Annotation{MT},
                _b::Annotation{VT};
                M=I,
                verbose=0,
                options...
            ) where {RT, MT <: $AMT, VT <: Vector}
                psolver = $solver
                pamt = $AMT
                if verbose > 0
                    @info "($psolver, $pamt) augmented forward"
                end
                A = _A.val
                b = _b.val
                x, stats = Krylov.$solver(A,b; M=M, verbose=verbose, options...)
                bx = zeros(length(x))
                bstats = deepcopy(stats)
                if needs_primal(config)
                    return AugmentedReturn(
                        (x, stats),
                        (bx, bstats),
                        (A,x, Ref(bx), verbose, M)
                    )
                else
                    return AugmentedReturn(nothing, (bx, bstats), (A,x))
                end
            end

            function reverse(
                config,
                ::Const{typeof(Krylov.$solver)},
                dret::Type{RT},
                cache,
                _A::Annotation{MT},
                _b::Annotation{<:Vector};
                options...
            ) where {RT, MT <: $AMT}
                (A,x,bx,verbose,M) = cache
                psolver = $solver
                pamt = $AMT
                if verbose > 0
                    @info "($psolver, $pamt) reverse"
                end
                _b.dval .= Krylov.$solver(transpose(A), bx[]; M=M, verbose=verbose, options...)[1]
                _A.dval .= -x .* _b.dval'
                return (nothing, nothing)
            end
        end
    end
end
