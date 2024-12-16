using TeneT,Optim,Random,Zygote,TensorOperations,JLD2

include("./get_gs.jl")

Ni = 1
Nj = 1
D = 2
χ = 10

Sx = Float64[0 1; 1 0]/2
Sy = ComplexF64[0 -1im; 1im 0]/2
Sz = Float64[1 0; 0 -1]/2
f(oi,oj) = @tensor oij[-1 -2 -3 -4] := oi[-1 -2]*oj[-3 -4]
h = f(Sx,-Sx) + f(Sy,Sy) + f(Sz,-Sz)


folder = joinpath("./data/", "$(Ni)x$(Nj)/")
mkpath(folder)
#A = rand(ComplexF64,D,D,D,D,2,Ni,Nj)
#A = load_object("A.jld2")
key = (folder, atype=Array, Ni, Nj, D, χ, tol=1e-10, maxiter=10, miniter=1, verbose=false)
oc = optcont()
A = load(joinpath(folder, "D$(D)_χ$(χ)_tol$(key.tol)_maxiter$(key.maxiter).jld2"), "bcipeps")

@show energy(h, A, oc, key; verbose = true, savefile = true)
oc = optcont()
f(x) = real(energy(h, x, oc, key))
g(x) = Zygote.gradient(f,x)[1]
#g(A)
optimise_ipeps(A, h, key; f_tol = 1e-6, opiter = 20, optimmethod = LBFGS(m = 100))