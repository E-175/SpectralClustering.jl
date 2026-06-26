"""
    compute_eigenvectors(L::AbstractMatrix, k::Int)

Extracts the bottom `k` eigenvectors from the Laplacian `L`.

This function computes the eigenvector embedding used in spectral clustering.
The input `L` is expected to be a graph Laplacian or a normalized graph Laplacian.

# Arguments
- `L`: Square Laplacian matrix.
- `k`: Number of eigenvectors to return.

# Returns
A matrix whose rows are the selected eigenvectors. The matrix has one column per
sample and `k` rows (features × samples format).

# Throws
- `ArgumentError` if `L` is not square.
- `ArgumentError` if `k` is smaller than 1 or larger than the number of samples.
- `ArgumentError` if the selected eigenvectors contain non-negligible imaginary parts.
"""
function compute_eigenvectors(L::AbstractMatrix, k::Int)
    # Get the dimensions of the Laplacian matrix.
    # A Laplacian must be square because it represents relationships between samples.
    n, m = size(L)

    # Check that L is square.
    n == m || throw(ArgumentError("Laplacian matrix L must be square."))

    # Check that k is valid.
    # We cannot request fewer than 1 eigenvector or more eigenvectors than samples.
    1 <= k <= n || throw(ArgumentError("k must be between 1 and the number of samples."))

    # Solve the standard eigenvalue problem:
    #     L * y = λ * y
    #
    # Important:
    # We intentionally do not wrap L in Symmetric(L), because the random-walk
    # normalized Laplacian L_rw = I - D⁻¹W is generally not symmetric.
    decomposition = eigen(Matrix(L))

    # Sort eigenvalues from smallest to largest.
    # Spectral clustering uses the eigenvectors belonging to the smallest eigenvalues.
    indices = sortperm(real.(decomposition.values))[1:k]

    # Select the corresponding eigenvectors.
    V = decomposition.vectors[:, indices]

    # For the Laplacians used here, the relevant eigenvectors should be real.
    # Small imaginary parts can occur because of numerical rounding.
    # Large imaginary parts would indicate that something is wrong with the input matrix.
    if maximum(abs.(imag.(V))) > 1e-10
        throw(ArgumentError("Selected eigenvectors contain non-negligible imaginary parts."))
    end

    # Return only the real part and transpose to features × samples.
    # This keeps the output usable for the later K-Means step.
    return Matrix(real.(V)')
end