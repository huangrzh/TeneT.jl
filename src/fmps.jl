module FiniteMPS

export MPS, initialize_mps, apply_operator, measure, canonicalize_mps, truncate_bond_dimension

using LinearAlgebra

struct MPS
    tensors::Vector{Array{ComplexF64, 3}}
end

function initialize_mps(d::Int, D::Int, L::Int) :: MPS
    @assert D>=d && L>2
    tensors = [rand(ComplexF64, D, d, D) for _ in 1:L]
    tensors[1] = rand(ComplexF64, 1, d, d)
    tensors[end] = rand(ComplexF64, d, d, 1)
    
    for i0 in 2:L/2
        i = Int(i0)
        Dl = size(tensors[i-1], 3)
        Dr = min(D, Dl*d)
        tensors[i] = rand(ComplexF64, Dl, d, Dr)
        tensors[L-i+1] = rand(ComplexF64, Dr, d, Dl)
    end

    if mod(L,2) == 1
        i = Int((L+1)/2)
        Dl = size(tensors[i-1], 3)
        Dr = Dl
        tensors[i] = rand(ComplexF64, Dl, d, Dr)
    end

    return MPS(tensors)
end

function canonicalize_mps(mps::MPS) :: MPS
    L = length(mps.tensors)
    tensors = copy(mps.tensors)
    
    # Left canonical form using QR decomposition
    for i in 1:L
        D1, d, D2 = size(mps.tensors[i])
        Q, R = qrpos(reshape(mps.tensors[i], D1*d, D2))
        tensors[i] = reshape(Q, D1, d, size(Q, 2))
        if i < L
            tensors[i+1] = contract_tensors(R, mps.tensors[i+1])
        end
    end
    
    return MPS(tensors)
end


function truncate_bond_dimension(mps0::MPS, max_bond_dim::Int) :: MPS
    mps = canonicalize_mps(mps0)
    tensors = copy(mps.tensors)
    
    # Right canonical form with truncation
    for i in L:-1:2
        tensor = tensors[i]
        D1, d, D2 = size(tensor)
        rho = reshape(tensor, D1*d, D2) * conj(reshape(tensor, D1*d, D2))
        V,D = eigen(rho)
        D = sort(D, rev=true)
        D = cumsum(D) / sum(D)
        bond_dim = min(max_bond_dim, length(D))
        U = V[:, 1:bond_dim]
        
        tensors[i] = reshape(V', bond_dim, d, D2)
        tensors[i-1] = contract_tensors(tensors[i-1], U * Diagonal(S))
    end
    
    return MPS(tensors)
end

function contract_tensors(tensor1::Array{ComplexF64, 3}, tensor2::Array{ComplexF64, 3}) :: Array{ComplexF64, 3}
    D1, d1, D2 = size(tensor1)
    D3, d2, D4 = size(tensor2)
    return reshape(tensor1 * reshape(tensor2, D3, d2*D4), D1, d1, d2, D4)
end

function contract_tensors(tensor1::Array{ComplexF64, 2}, tensor2::Array{ComplexF64, 3}) :: Array{ComplexF64, 3}
    D1,D2,D3 = size(tensor2)
    return reshape(tensor1 * reshape(tensor2, D1, D2*D3), :, D2, D3)
end

end # module FiniteMPS