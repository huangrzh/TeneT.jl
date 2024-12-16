#using TeneT
using OMEinsum
using Zygote, Test,Random

include("./lib/hamiltonian_models.jl")
include("./lib/optimise_ipeps.jl")
include("./lib/construct_M.jl")
include("./lib/observable.jl")


using Optim
using Random

for Ni = [1], Nj = [1], D in [2], χ in [10]
    Random.seed!(100)
    model = Heisenberg(Ni,Nj,-1.0,-1.0,1.0)
    A, key = init_ipeps(model; Ni=Ni, Nj=Nj, D=D, χ=χ, verbose= false)
    optimise_ipeps(A, key; f_tol = 1e-6, opiter = 20, optimmethod = LBFGS(m = 100))
end
