using TensorKit
using FastGaussQuadrature

function f(ℝϕ1, ℂϕ1, ℝϕ2, ℂϕ2, μ0, λ)
    return exp(
        -1 / 2 * ((ℝϕ1 - ℝϕ2)^2 + (ℂϕ1 - ℂϕ2)^2)
            - μ0 / 8 * (ℝϕ1^2 + ℂϕ1^2 + ℝϕ2^2 + ℂϕ2^2)
            - λ / 16 * ((ℝϕ1^2 + ℂϕ1^2)^2 + (ℝϕ2^2 + ℂϕ2^2)^2)
    )
end


function fmatrix(ys, μ0, λ)
    K = length(ys)
    matrix = zeros(K^2, K^2)
    for i in 1:K
        for j in i:K    # Optimazation
            for k in 1:K
                for l in 1:K
                    idx1 = (i - 1) * K + j
                    idx2 = (k - 1) * K + l
                    if idx2 >= idx1  # only compute upper triangle
                        val = f(ys[i], ys[j], ys[k], ys[l], μ0, λ)
                        matrix[idx1, idx2] = val
                        matrix[idx2, idx1] = val  # symmetric counterpart

                        # Based on the simultaneous symmetry of (i,j)<->(j,i) and (k,l)<->(l,k)
                        idx3 = (j - 1) * K + i
                        idx4 = (l - 1) * K + k
                        matrix[idx3, idx4] = val
                        matrix[idx4, idx3] = val  # symmetric counterpart
                    end
                end
            end
        end
    end
    return TensorMap(matrix, ℂ^(K^2) ← ℂ^(K^2))
end


function getTensor(K, μ0, λ)
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix(ys, μ0, λ)

    # SVD fmatrix
    U, S, V = tsvd(f)

    # Make tensor for one site
    T_arr = [
        sum(
                √(S[i, i] * S[j, j] * S[k, k] * S[l, l]) *
                ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) *
                U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] * V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
                for α in 1:K, β in 1:K
            )
            for i in 1:(K^2), j in 1:(K^2), k in 1:(K^2), l in 1:(K^2)
    ]
    T = TensorMap(T_arr, ℂ^(K^2) ⊗ ℂ^(K^2) ← ℂ^(K^2) ⊗ ℂ^(K^2))
    return T
end


function getImpϕTensor(K, μ0, λ)
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix(ys, μ0, λ)

    # SVD fmatrix
    U, S, V = tsvd(f)

    # Make tensor for one site
    T_arr = [
        sum(
                √(S[i, i] * S[j, j] * S[k, k] * S[l, l]) *
                (ys[α] + ys[β]im) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) *
                U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] * V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
                for α in 1:K, β in 1:K
            )
            for i in 1:(K^2), j in 1:(K^2), k in 1:(K^2), l in 1:(K^2)
    ]
    T = TensorMap(T_arr, ℂ^(K^2) ⊗ ℂ^(K^2) ← ℂ^(K^2) ⊗ ℂ^(K^2))
    return T
end

function getImpAbsϕTensor(K, μ0, λ)
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix(ys, μ0, λ)

    # SVD fmatrix
    U, S, V = tsvd(f)

    # Make tensor for one site
    T_arr = [
        sum(
                √(S[i, i] * S[j, j] * S[k, k] * S[l, l]) *
                sqrt(ys[α]^2 + ys[β]^2) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) *
                U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] * V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
                for α in 1:K, β in 1:K
            )
            for i in 1:(K^2), j in 1:(K^2), k in 1:(K^2), l in 1:(K^2)
    ]
    T = TensorMap(T_arr, ℂ^(K^2) ⊗ ℂ^(K^2) ← ℂ^(K^2) ⊗ ℂ^(K^2))
    return T
end

function getImpϕ2Tensor(K, μ0, λ)
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix(ys, μ0, λ)

    # SVD fmatrix
    U, S, V = tsvd(f)

    # Make tensor for one site
    T_arr = [
        sum(
                √(S[i, i] * S[j, j] * S[k, k] * S[l, l]) *
                (ys[α]^2 + ys[β]^2) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) *
                U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] * V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
                for α in 1:K, β in 1:K
            )
            for i in 1:(K^2), j in 1:(K^2), k in 1:(K^2), l in 1:(K^2)
    ]
    T = TensorMap(T_arr, ℂ^(K^2) ⊗ ℂ^(K^2) ← ℂ^(K^2) ⊗ ℂ^(K^2))
    return T
end

function getImpAll(K, μ0, λ)
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix(ys, μ0, λ)

    # SVD fmatrix
    U, S, V = tsvd(f)

    # Make tensor for pure
    T_arr = [
        sum(
                √(S[i, i] * S[j, j] * S[k, k] * S[l, l]) *
                ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) *
                U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] * V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
                for α in 1:K, β in 1:K
            )
            for i in 1:(K^2), j in 1:(K^2), k in 1:(K^2), l in 1:(K^2)
    ]
    T = TensorMap(T_arr, ℂ^(K^2) ⊗ ℂ^(K^2) ← ℂ^(K^2) ⊗ ℂ^(K^2))

    # Make tensor for |ϕ|
    T_impϕ_arr = [
        sum(
                √(S[i, i] * S[j, j] * S[k, k] * S[l, l]) *
                sqrt(ys[α]^2 + ys[β]^2) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) *
                U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] * V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
                for α in 1:K, β in 1:K
            )
            for i in 1:(K^2), j in 1:(K^2), k in 1:(K^2), l in 1:(K^2)
    ]
    T_impϕ = TensorMap(T_impϕ_arr, ℂ^(K^2) ⊗ ℂ^(K^2) ← ℂ^(K^2) ⊗ ℂ^(K^2))

    # Make tensor for |ϕ|²
    T_impϕ2_arr = [
        sum(
                √(S[i, i] * S[j, j] * S[k, k] * S[l, l]) *
                (ys[α]^2 + ys[β]^2) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) *
                U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] * V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
                for α in 1:K, β in 1:K
            )
            for i in 1:(K^2), j in 1:(K^2), k in 1:(K^2), l in 1:(K^2)
    ]
    T_impϕ2 = TensorMap(T_impϕ2_arr, ℂ^(K^2) ⊗ ℂ^(K^2) ← ℂ^(K^2) ⊗ ℂ^(K^2))
    return T, T_impϕ, T_impϕ2
end
