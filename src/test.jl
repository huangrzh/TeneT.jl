#include("patch.jl")
#include("environment.jl")
#include("fixedpoint.jl")
#include("vumpsruntime.jl")
#include("autodiff.jl")

using Test,Optim,Random,TeneT,Revise

include("./hamiltonian_models.jl")
include("./construct_M.jl")
include("./observable.jl")
include("./optimise_ipeps.jl")


h = hamiltonian(Heisenberg(1, 1))


@test size(h) == (2, 2, 2, 2)
rh = reshape(permutedims(h, (1, 3, 2, 4)), 4, 4)
@test rh' == rh

Ni = 1
Nj = 1
D = 2
χ = 10
model = Heisenberg(Ni, Nj)
A, key = init_ipeps(model; Ni=Ni, Nj=Nj, D=D, χ=χ);

model = Heisenberg(Ni, Nj)
A, key = init_ipeps(model; Ni=Ni, Nj=Nj, D=D, χ=χ)
A = real(A)

oc = optcont(D, χ)
h = hamiltonian(model)
@show energy(h, A, oc, key; verbose = true, savefile = true)

Random.seed!(100)
model = Heisenberg(Ni, Nj, 1.0, 1.0, 1.0)
A, key = init_ipeps(model; Ni=Ni, Nj=Nj, D=D, χ=χ, verbose=false)
A = real(A)
optimise_ipeps(A, key; f_tol=1e-6, opiter=20, optimmethod=LBFGS(m=20))
