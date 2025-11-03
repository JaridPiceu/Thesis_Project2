using TensorKit
using FastGaussQuadrature

using Base.Threads
using Combinatorics: permutations



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
    @threads for i in 1:K
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
    
    N = K^2
    T_arr = zeros(eltype(S), N, N, N, N)

    weights = [ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]

    perms = collect(permutations(1:4))  # 24 total

    @threads for i in 1:N
        for j in i:N
            for k in j:N
                for l in k:N
                    s = 0.0
                    factor = √(S[i,i]*S[j,j]*S[k,k]*S[l,l])
                    for α in 1:K, β in 1:K
                        s += factor *
                             weights[α, β] *
                             U[(α-1)*K+β, i]*U[(α-1)*K+β, j] *
                             V[k, (α-1)*K+β]*V[l, (α-1)*K+β]
                    end

                    # Fill all 24 symmetric permutations
                    idxs = (i,j,k,l)
                    for p in perms
                        ii, jj, kk, ll = idxs[p[1]], idxs[p[2]], idxs[p[3]], idxs[p[4]]
                        T_arr[ii,jj,kk,ll] = s
                    end
                end
            end
        end
    end

    T = TensorMap(T_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    return T
end


function getImpϕTensor(K, μ0, λ)
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix(ys, μ0, λ)

    # SVD fmatrix
    U, S, V = tsvd(f)
    
    N = K^2
    T_arr = zeros(ComplexF64, N, N, N, N)

    weights = [(ys[α] + ys[β]im) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]

    perms = collect(permutations(1:4))  # 24 total

    @threads for i in 1:N
        for j in i:N
            for k in j:N
                for l in k:N
                    s = 0.0
                    factor = √(S[i,i]*S[j,j]*S[k,k]*S[l,l])
                    for α in 1:K, β in 1:K
                        s += factor *
                             weights[α, β] *
                             U[(α-1)*K+β, i]*U[(α-1)*K+β, j] *
                             V[k, (α-1)*K+β]*V[l, (α-1)*K+β]
                    end

                    # Fill all 24 symmetric permutations
                    idxs = (i,j,k,l)
                    for p in perms
                        ii, jj, kk, ll = idxs[p[1]], idxs[p[2]], idxs[p[3]], idxs[p[4]]
                        T_arr[ii,jj,kk,ll] = s
                    end
                end
            end
        end
    end

    T = TensorMap(T_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    return T
end



function getImpϕdagTensor(K, μ0, λ)
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix(ys, μ0, λ)

    # SVD fmatrix
    U, S, V = tsvd(f)
    
    N = K^2
    T_arr = zeros(ComplexF64, N, N, N, N)

    weights = [(ys[α] - ys[β]im) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]

    perms = collect(permutations(1:4))  # 24 total

    @threads for i in 1:N
        for j in i:N
            for k in j:N
                for l in k:N
                    s = 0.0
                    factor = √(S[i,i]*S[j,j]*S[k,k]*S[l,l])
                    for α in 1:K, β in 1:K
                        s += factor *
                             weights[α, β] *
                             U[(α-1)*K+β, i]*U[(α-1)*K+β, j] *
                             V[k, (α-1)*K+β]*V[l, (α-1)*K+β]
                    end

                    # Fill all 24 symmetric permutations
                    idxs = (i,j,k,l)
                    for p in perms
                        ii, jj, kk, ll = idxs[p[1]], idxs[p[2]], idxs[p[3]], idxs[p[4]]
                        T_arr[ii,jj,kk,ll] = s
                    end
                end
            end
        end
    end

    T = TensorMap(T_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    return T
end


function getImpAbsϕTensor(K, μ0, λ)
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix(ys, μ0, λ)

    # SVD fmatrix
    U, S, V = tsvd(f)
    
    N = K^2
    T_arr = zeros(ComplexF64, N, N, N, N)

    weights = [sqrt(ys[α]^2 + ys[β]^2) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]

    perms = collect(permutations(1:4))  # 24 total

    @threads for i in 1:N
        for j in i:N
            for k in j:N
                for l in k:N
                    s = 0.0
                    factor = √(S[i,i]*S[j,j]*S[k,k]*S[l,l])
                    for α in 1:K, β in 1:K
                        s += factor *
                             weights[α, β] *
                             U[(α-1)*K+β, i]*U[(α-1)*K+β, j] *
                             V[k, (α-1)*K+β]*V[l, (α-1)*K+β]
                    end

                    # Fill all 24 symmetric permutations
                    idxs = (i,j,k,l)
                    for p in perms
                        ii, jj, kk, ll = idxs[p[1]], idxs[p[2]], idxs[p[3]], idxs[p[4]]
                        T_arr[ii,jj,kk,ll] = s
                    end
                end
            end
        end
    end

    T = TensorMap(T_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    return T
end


function getImpϕ2Tensor(K, μ0, λ)
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix(ys, μ0, λ)

    # SVD fmatrix
    U, S, V = tsvd(f)
    
    N = K^2
    T_arr = zeros(ComplexF64, N, N, N, N)

    weights = [sqrt(ys[α]^2 + ys[β]^2) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]

    perms = collect(permutations(1:4))  # 24 total

    @threads for i in 1:N
        for j in i:N
            for k in j:N
                for l in k:N
                    s = 0.0
                    factor = √(S[i,i]*S[j,j]*S[k,k]*S[l,l])
                    for α in 1:K, β in 1:K
                        s += factor *
                             weights[α, β] *
                             U[(α-1)*K+β, i]*U[(α-1)*K+β, j] *
                             V[k, (α-1)*K+β]*V[l, (α-1)*K+β]
                    end

                    # Fill all 24 symmetric permutations
                    idxs = (i,j,k,l)
                    for p in perms
                        ii, jj, kk, ll = idxs[p[1]], idxs[p[2]], idxs[p[3]], idxs[p[4]]
                        T_arr[ii,jj,kk,ll] = s
                    end
                end
            end
        end
    end

    T = TensorMap(T_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    return T
end


function getImpAll(K, μ0, λ)
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix(ys, μ0, λ)

    # SVD fmatrix
    U, S, V = tsvd(f)
    
    N = K^2


    T_arr = zeros(ComplexF64, N, N, N, N)
    T_ϕ_arr = zeros(ComplexF64, N, N, N, N)
    T_ϕdag_arr = zeros(ComplexF64, N, N, N, N)
    T_ϕabs_arr = zeros(ComplexF64, N, N, N, N)
    T_ϕ2_arr = zeros(ComplexF64, N, N, N, N)

    w = [ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]
    w_ϕ = [(ys[α] + ys[β]im) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]
    w_ϕdag = [(ys[α] - ys[β]im) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]
    w_ϕabs = [sqrt(ys[α]^2 + ys[β]^2) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]
    w_ϕ2 = [(ys[α]^2 + ys[β]^2) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]


    perms = collect(permutations(1:4))  # 24 total

    @threads for i in 1:N
        for j in i:N
            for k in j:N
                for l in k:N
                    s = 0.0
                    s_ϕ = 0.0
                    s_ϕdag = 0.0
                    s_ϕabs = 0.0
                    s_ϕ2 = 0.0
                    factor = √(S[i,i]*S[j,j]*S[k,k]*S[l,l])
                    for α in 1:K, β in 1:K
                        s += factor * w[α, β] * U[(α-1)*K+β, i]*U[(α-1)*K+β, j] * V[k, (α-1)*K+β]*V[l, (α-1)*K+β]
                        s_ϕ += factor * w_ϕ[α, β] * U[(α-1)*K+β, i]*U[(α-1)*K+β, j] * V[k, (α-1)*K+β]*V[l, (α-1)*K+β]
                        s_ϕdag += factor * w_ϕdag[α, β] * U[(α-1)*K+β, i]*U[(α-1)*K+β, j] * V[k, (α-1)*K+β]*V[l, (α-1)*K+β]
                        s_ϕabs += factor * w_ϕabs[α, β] * U[(α-1)*K+β, i]*U[(α-1)*K+β, j] * V[k, (α-1)*K+β]*V[l, (α-1)*K+β]
                        s_ϕ2 += factor * w_ϕ2[α, β] * U[(α-1)*K+β, i]*U[(α-1)*K+β, j] * V[k, (α-1)*K+β]*V[l, (α-1)*K+β]
                    end

                    # Fill all 24 symmetric permutations
                    idxs = (i,j,k,l)
                    for p in perms
                        ii, jj, kk, ll = idxs[p[1]], idxs[p[2]], idxs[p[3]], idxs[p[4]]
                        T_arr[ii,jj,kk,ll] = s
                        T_ϕ_arr[ii,jj,kk,ll] = s_ϕ
                        T_ϕdag_arr[ii,jj,kk,ll] = s_ϕdag
                        T_ϕabs_arr[ii,jj,kk,ll] = s_ϕabs
                        T_ϕ2_arr[ii,jj,kk,ll] = s_ϕ2
                    end
                end
            end
        end
    end

    T = TensorMap(T_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    T_ϕ = TensorMap(T_ϕ_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    T_ϕdag = TensorMap(T_ϕdag_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    T_ϕabs = TensorMap(T_ϕabs_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    T_ϕ2 = TensorMap(T_ϕ2_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    return T, T_ϕ, T_ϕdag, T_ϕabs, T_ϕ2
end
