using Enzyme
import .EnzymeRules: forward, reverse, augmented_primal
using .EnzymeRules

for solver in (:cg, :bicgstab, :gmres)
    @eval begin
        function forward(
                func::Const{typeof(Krylov.$solver)},
                RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated}},
                _A::Union{Const, Duplicated},
                _b::Union{Const, Duplicated};
                options...
            )
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

export forward

function augmented_primal(
    config,
    func::Const{typeof(Krylov.gmres)},
    ret::Type{<:Duplicated},
    _A::Union{Const, Duplicated},
    _b::Union{Const, Duplicated}
)
    A = _A.val
    b = _b.val
    x, stats = Krylov.gmres(A,b)
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
    ::Const{typeof(Krylov.gmres)},
    dret,
    tape,
    _A,
    _b
)
    (A,x,bx) = tape
    _b.dval .= gmres(transpose(A), bx[])[1]
    _A.dval .= -x .* _b.dval'
    return (nothing, nothing)
end

export augmented_primal, reverse
