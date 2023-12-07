import ForwardDiff: Dual, Partials, partials, value

for solver in (:cg, :gmres, :bicgstab)
@eval begin
    function Krylov.$solver(_A::SparseMatrixCSC{V, Int64}, _b::Vector{Dual{T, V, N}}; options...) where {T, V, N}
    A = SparseMatrixCSC(_A.m, _A.n, _A.colptr, _A.rowval, value.(_A.nzval))
    b = value.(_b)
    m = length(b)
    dbs = Matrix{V}(undef, m, N)
    for i in 1:m
        dbs[i,:] = partials(_b[i])
    end
    x, stats = $solver(A,b; options...)
    dxs = Matrix{Float64}(undef, m, N)
    px = Vector{Partials{N,V}}(undef, m)
    for i in 1:N
        nb = dbs[:,i]
        dx, dstats = $solver(A,nb; options...)
        dxs[:,i] = dx
    end
    for i in 1:m
        px[i] = Partials{N,V}(Tuple(dxs[i,j] for j in 1:N))
    end
    duals = Dual{T,V,N}.(x, px)
    return (duals, stats)
    end

    function Krylov.$solver(A::Matrix{V}, _b::Vector{Dual{T, V, N}}; options...) where {T, V, N}
    b = value.(_b)
    m = length(b)
    dbs = Matrix{V}(undef, m, N)
    for i in 1:m
        dbs[i,:] = partials(_b[i])
    end
    x, stats = $solver(A,b; options...)
    dxs = Matrix{Float64}(undef, m, N)
    px = Vector{Partials{N,V}}(undef, m)
    for i in 1:N
        nb = dbs[:,i]
        dx, dstats = $solver(A,nb; options...)
        dxs[:,i] = dx
    end
    for i in 1:m
        px[i] = Partials{N,V}(Tuple(dxs[i,j] for j in 1:N))
    end
    duals = Dual{T,V,N}.(x, px)
    return (duals, stats)
    end
end

# function Krylov.cg(_A::SparseMatrixCSC{Dual{T, V, NA}, Int64}, _b::Vector{Dual{T, V, NB}}; options...) where {T, V, NA, NB}
#   A = SparseMatrixCSC(_A.m, _A.n, _A.colptr, _A.rowval, value.(_A.nzval))
#   dAs = Vector{SparseMatrixCSC{Float64, Int64}}(undef, NA)
#   for i in 1:NA
#     dAs[i] = SparseMatrixCSC(_A.m, _A.n, _A.colptr, _A.rowval, partials.(_A.nzval, i))
#   end
#   b = value.(_b)
#   m = length(b)
#   dbs = Matrix{V}(undef, m, NB)
#   for i in 1:m
#     dbs[i,:] = partials(_b[i])
#   end
#   x, stats = cg(A,b)
#   dxs = Matrix{Float64}(undef, m, N)
#   px = Vector{Partials{N,V}}(undef, n)
#   for i in 1:N
#     nb = dbs[:,i] - dAs[i]*x
#     dx, dstats = cg(A[i],nb)
#     dxs[:,i] = dx
#   end
#   for i in 1:m
#       px[i] = Partials{N,V}(Tuple(dxs[i,j] for j in 1:N))
#   end
#   duals = Dual{T,V,N}.(x, px)
#   return (duals, stats)
# end
end
