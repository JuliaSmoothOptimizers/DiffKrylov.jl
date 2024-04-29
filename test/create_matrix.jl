function create_unsymmetric_matrix(n)
    # Ensure the size is at least 2
    if n < 2
        throw(ArgumentError("Matrix size should be at least 2x2"))
    end

    # Generate a random n x n matrix with entries from a normal distribution
    A = randn(n, n)

    # Perform Singular Value Decomposition
    U, S, V = svd(A)

    # Modify the singular values to make them close to each other but not too small
    # Here we set them all to be between 1 and 2
    S = Diagonal(range(1, stop=2, length=n))

    # Reconstruct the matrix
    well_conditioned_matrix = U * S * V'

    return well_conditioned_matrix
end
