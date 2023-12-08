import ForwardDiff: Dual, Partials, partials, value

_matrix_values(A::SparseMatrixCSC{Dual{T, V, N}, IT}) where {T, V, N, IT} = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, value.(A.nzval))
_matrix_values(A::Matrix{Dual{T, V, N}}) where {T, V, N} = Matrix{V}(value.(A))
function _matrix_partials(A::SparseMatrixCSC{Dual{T, V, N}, IT}) where {T, V, N, IT}
    dAs = Vector{SparseMatrixCSC{V, IT}}(undef, N)
    for i in 1:N
        dAs[i] = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, partials.(A.nzval, i))
    end
    return dAs
end
function _matrix_partials(A::Matrix{Dual{T, V, N}}) where {T, V, N}
    dAs = Vector{Matrix{V}}(undef, N)
    for i in 1:N
        dAs[i] = Matrix(partials.(A, i))
    end
    return dAs
end


for solver in (:cg, :bicgstab)
    for matrix in (:(SparseMatrixCSC{V, IT}), :(Matrix{V}))
        @eval begin
            function Krylov.$solver(A::$matrix, _b::Vector{Dual{T, V, N}}; options...) where {T, V, N, IT}
                b = value.(_b)
                m = length(b)
                dbs = Matrix{V}(undef, m, N)
                for i in 1:m
                    dbs[i,:] = partials(_b[i])
                end
                x, stats = $solver(A,b; options...)
                dxs = Matrix{V}(undef, m, N)
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
    end

    for matrix in (:(SparseMatrixCSC{Dual{T,V,N}, IT}), :(Matrix{Dual{T,V,N}}))
        @eval begin
            function Krylov.$solver(_A::$matrix, b::Vector{V}; options...) where {T, V, N, IT}
                A = _matrix_values(_A)
                dAs = _matrix_partials(_A)
                m = length(b)
                x, stats = $solver(A,b; options...)
                dxs = Matrix{V}(undef, m, N)
                px = Vector{Partials{N,V}}(undef, m)
                for i in 1:N
                    nb = - dAs[i]*x
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
    end

    for matrix in (:(SparseMatrixCSC{Dual{T,V,N}, IT}), :(Matrix{Dual{T,V,N}}))
        @eval begin
            function Krylov.$solver(_A::$matrix, _b::Vector{Dual{T, V, N}}; options...) where {T, V, N, IT}
                A = _matrix_values(_A)
                dAs = _matrix_partials(_A)
                b = value.(_b)
                m = length(b)
                dbs = Matrix{V}(undef, m, N)
                for i in 1:m
                    dbs[i,:] = partials(_b[i])
                end
                x, stats = $solver(A,b; options...)
                dxs = Matrix{V}(undef, m, N)
                px = Vector{Partials{N,V}}(undef, m)
                for i in 1:N
                    nb = dbs[:,i] - dAs[i]*x
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
    end

end

for matrix in (:(SparseMatrixCSC{V, IT}), :(Matrix{V}))
    @eval begin
        function Krylov.gmres(A::$matrix, _b::Vector{Dual{T, V, N}}; options...) where {T, V, N, IT}
            b = value.(_b)
            m = length(b)
            dbs = Matrix{V}(undef, m, N)
            for i in 1:m
                dbs[i,:] = partials(_b[i])
            end
            x, stats = gmres(A,b; options...)
            dxs = Matrix{V}(undef, m, N)
            px = Vector{Partials{N,V}}(undef, m)
            if N != 0
                xs, dstats = block_gmres(A,dbs; options...)
                dxs .= xs
            end
            for i in 1:m
                px[i] = Partials{N,V}(Tuple(dxs[i,j] for j in 1:N))
            end
            duals = Dual{T,V,N}.(x, px)
            return (duals, stats)
        end
    end
end

for matrix in (:(SparseMatrixCSC{Dual{T,V,N}, IT}), :(Matrix{Dual{T,V,N}}))
    @eval begin
        function Krylov.gmres(_A::$matrix, b::Vector{V}; options...) where {T, V, N, IT}
            A = _matrix_values(_A)
            dAs = _matrix_partials(_A)
            m = length(b)
            dbs = Matrix{V}(undef, m, N)
            x, stats = gmres(A,b; options...)
            dxs = Matrix{V}(undef, m, N)
            px = Vector{Partials{N,V}}(undef, m)
            for i in 1:N
                dbs[:,i] = - dAs[i]*x
            end
            if N != 0
                dx, dstats = block_gmres(A,dbs; options...)
            end
            dxs .= dx
            for i in 1:m
                px[i] = Partials{N,V}(Tuple(dxs[i,j] for j in 1:N))
            end
            duals = Dual{T,V,N}.(x, px)
            return (duals, stats)
        end
    end
end

for matrix in (:(SparseMatrixCSC{Dual{T,V,N}, IT}), :(Matrix{Dual{T,V,N}}))
    @eval begin
        function Krylov.gmres(_A::$matrix, _b::Vector{Dual{T, V, N}}; options...) where {T, V, N, IT}
            A = _matrix_values(_A)
            dAs = _matrix_partials(_A)
            b = value.(_b)
            m = length(b)
            dbs = Matrix{V}(undef, m, N)
            for i in 1:m
                dbs[i,:] = partials(_b[i])
            end
            x, stats = gmres(A,b; options...)
            dxs = Matrix{V}(undef, m, N)
            px = Vector{Partials{N,V}}(undef, m)
            for i in 1:N
                dbs[:,i] -= dAs[i]*x
            end
            if N != 0
                dx, dstats = block_gmres(A,dbs; options...)
            end
            dxs .= dx
            for i in 1:m
                px[i] = Partials{N,V}(Tuple(dxs[i,j] for j in 1:N))
            end
            duals = Dual{T,V,N}.(x, px)
            return (duals, stats)
        end
    end
end
