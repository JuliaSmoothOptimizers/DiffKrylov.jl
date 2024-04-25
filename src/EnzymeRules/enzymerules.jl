using Enzyme
import .EnzymeRules: forward, reverse, augmented_primal
using .EnzymeRules

export augmented_primal, reverse, forward

for AMT in (:Matrix, :SparseMatrixCSC)
    for solver in (:cg, :bicgstab, :gmres)
        @eval begin
            function forward(
                func::Const{typeof(Krylov.$solver)},
                ret::Type{RT},
                _A::Annotation{MT},
                _b::Annotation{VT},
                options...
            ) where {RT, MT <: $AMT, VT <: Vector}
                psolver = $solver
                pamt = $AMT
                # println("($psolver, $pamt) forward rule")
                A = _A.val
                b = _b.val
                dx = []
                x, stats = Krylov.$solver(A,b; options...)
                if isa(_A, Duplicated) && isa(_b, Duplicated)
                    dA = _A.dval
                    db = _b.dval
                    db -= dA*x
                    dx, dstats = Krylov.$solver(A,db; options...)
                elseif isa(_A, Duplicated) && isa(_b, Const)
                    dA = _A.dval
                    db = -dA*x
                    dx, dstats = Krylov.$solver(A,db; options...)
                elseif isa(_A, Const) && isa(_b, Duplicated)
                    db = _b.dval
                    dx, dstats = Krylov.$solver(A,db; options...)
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
    for solver in (:cg, :bicgstab, :gmres)
        @eval begin
            function augmented_primal(
                config,
                func::Const{typeof(Krylov.$solver)},
                ret::Type{RT},
                _A::Annotation{MT},
                _b::Annotation{VT}
            ) where {RT, MT <: $AMT, VT <: Vector}
                psolver = $solver
                pamt = $AMT
                # println("($psolver, $pamt) augmented forward")
                A = _A.val
                b = _b.val
                x, stats = Krylov.$solver(A,b)
                bx = zeros(length(x))
                bstats = deepcopy(stats)
                if needs_primal(config)
                    return AugmentedReturn((x, stats), (bx, bstats), (A,x, Ref(bx)))
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
                _b::Annotation{<:Vector},
            ) where {RT, MT <: $AMT}
                psolver = $solver
                pamt = $AMT
                # println("($psolver, $pamt) reverse")
                (A,x,bx) = cache
                _b.dval .= $solver(transpose(A), bx[])[1]
                _A.dval .= -x .* _b.dval'
                return (nothing, nothing)
            end
        end
    end
end
