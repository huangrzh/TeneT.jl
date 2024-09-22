using FileIO
using LinearAlgebra: norm, det, svd
using LineSearches
using Random
using Optim
using KrylovKit
using Printf
using OMEinsum, OMEinsumContractionOrders,Revise

#=
"""
```
        4
        │
 1 ── ipeps ── 3
        │
        2
```
"""


"""
```
                                            a ────┬──── c          
a ────┬──c ──┬──── f                        │     b     │  
│     b      e     │                        ├─ e ─┼─ f ─┤  
├─ g ─┼─  h ─┼─ i ─┤                        g     h     i 
│     k      n     │                        ├─ j ─┼─ k ─┤ 
j ────┴──l ──┴──── o                        │     m     │ 
                                            l ────┴──── n 
```
where the central two block are six order tensor have extra bond `pq` and `rs`
"""
=#

function mps_2x3(FL, ACu, api, apj, ACd, ARu, ARd, FR)
    @tensor lr[-1 -2 -3 -4] := FL[1 3 4]*ACu[1 2 6]*
                        api[3 5 7 2 -1 -2]*
                        ACd[4 5 9]*
                        ARu[6 8 12]*
                        apj[7 11 13 8 -3 -4]*
                        ARd[9 11 14]*
                        FR[12 13 14]
end

#	oc_V = ein"(((abc,aeg),ehfbpq),cfi),(gjl,(jmkhrs,(ikn,lmn))) -> pqrs"
function mps_3x2(ACu, FLu, api, FRu, FL, apj, FR, ACd)
    @tensor rho[-1 -2 -3 -4] := ACu[1 2 4]*FLu[1 3 6]*
                        api[3 7 5 2 -1 -2]*
                        FRu[4 5 10]*
                        FL[6 8 11]*
                        apj[8 12 9 7 -3 -4]*
                        FR[10 9 13]*
                        ACd[11 12 13]
end

function expectation_value2(h, ap, env, oc, key)
    _, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, FLu, FRu = env
    oc_H, oc_V = oc
    ACu = TeneT.ALCtoAC(ALu, Cu)
    ACd = TeneT.ALCtoAC(ALd, Cd)
    
    etol = 0
    Ni = key.Ni
    Nj = key.Nj
    verbose = key.verbose
    for j = 1:Nj, i = 1:Ni
        verbose && println("===========$i,$j===========")
        ir = Ni + 1 - i
        jr = j + 1 - (j==Nj) * Nj
        #    oc_H = ein"(((agj,abc),gkhbpq),jkl),(((fio,cef),hniers),lno) -> pqrs"
        #lr = oc_H(FL[:,:,:,i,j],ACu[:,:,:,i,j],ap[:,:,:,:,:,:,i,j],ACd[:,:,:,ir,j],FR[:,:,:,i,jr],ARu[:,:,:,i,jr],ap[:,:,:,:,:,:,i,jr],ARd[:,:,:,ir,jr])
        lr = mps_2x3(FL[:,:,:,i,j], ACu[:,:,:,i,j], ap[:,:,:,:,:,:,i,j], ap[:,:,:,:,:,:,i,jr], ACd[:,:,:,ir,j], ARu[:,:,:,i,jr], ARd[:,:,:,ir,jr], FR[:,:,:,i,jr])

                        
        #e = Array(ein"pqrs, pqrs -> "(lr,h))[]
        n =  Array(ein"pprr -> "(lr))[]
        e = @tensor lr[1 2 3 4]*h[1 2 3 4]
        #n = @tensor lr[1,1;2,2]
        verbose && println("Horizontal energy = $(e/n)")
        etol += e/n

        ir  =  i + 1 - (i==Ni) * Ni
        irr = Ni - i + (i==Ni) * Ni
        #lr = oc_V(ACu[:,:,:,i,j],FLu[:,:,:,i,j],ap[:,:,:,:,:,:,i,j],FRu[:,:,:,i,j],FL[:,:,:,ir,j],ap[:,:,:,:,:,:,ir,j],FR[:,:,:,ir,j],ACd[:,:,:,irr,j])
        lr = mps_3x2(ACu[:,:,:,i,j], FLu[:,:,:,i,j], ap[:,:,:,:,:,:,i,j], FRu[:,:,:,i,j],FL[:,:,:,ir,j],ap[:,:,:,:,:,:,ir,j],FR[:,:,:,ir,j],ACd[:,:,:,irr,j])
        #e = Array(ein"pqrs, pqrs -> "(lr,h))[]
        n = Array(ein"pprr -> "(lr))[]
        e = @tensor lr[1 2 3 4]*h[1 2 3 4]
        #n = @tensor h[1 1 2 2]
        verbose && println("Vertical energy = $(e/n)")
        etol += e/n
    end

    verbose && println("e = $(etol/Ni/Nj)")
    return etol/Ni/Nj
    
end


function optcont()
    oc_H = ein"(((agj,abc),gkhbpq),jkl),(((fio,cef),hniers),lno) -> pqrs"
	oc_V = ein"(((abc,aeg),ehfbpq),cfi),(gjl,(jmkhrs,(ikn,lmn))) -> pqrs"
    oc_H, oc_V
end


function energy(h, A, oc, key; verbose = key.verbose, savefile = true)
    # A = indexperm_symmetrize(A)
    D = key.D
    ap = ein"abcdeij,fghmnij->afbgchdmenij"(A, conj(A))
    ap = reshape(ap, D^2, D^2, D^2, D^2, 2, 2, key.Ni, key.Nj)
    M = ein"abcdeeij->abcdij"(ap)

    env = obs_env(M; updown = true, χ = key.χ, tol = key.tol, maxiter = key.maxiter, miniter = key.miniter, verbose = verbose, savefile = savefile, infolder = key.folder, outfolder = key.folder)
    GsE = expectation_value2(h, ap, env, oc, key)
    return GsE
end

function expectation_value(h, ap, env, oc, key)
    _, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, FLu, FRu = env
    oc_H, oc_V = oc
    ACu = TeneT.ALCtoAC(ALu, Cu)
    ACd = TeneT.ALCtoAC(ALd, Cd)

    etol = 0
    Ni = key.Ni
    Nj = key.Nj
    verbose = key.verbose
    for j = 1:Nj, i = 1:Ni
        verbose && println("===========$i,$j===========")
        ir = Ni + 1 - i
        jr = j + 1 - (j==Nj) * Nj
        lr = oc_H(FL[:,:,:,i,j],ACu[:,:,:,i,j],ap[:,:,:,:,:,:,i,j],ACd[:,:,:,ir,j],FR[:,:,:,i,jr],ARu[:,:,:,i,jr],ap[:,:,:,:,:,:,i,jr],ARd[:,:,:,ir,jr])
        e = Array(ein"pqrs, pqrs -> "(lr,h))[]
        n =  Array(ein"pprr -> "(lr))[]
        verbose && println("Horizontal energy = $(e/n)")
        etol += e/n

        ir  =  i + 1 - (i==Ni) * Ni
        irr = Ni - i + (i==Ni) * Ni
        lr = oc_V(ACu[:,:,:,i,j],FLu[:,:,:,i,j],ap[:,:,:,:,:,:,i,j],FRu[:,:,:,i,j],FL[:,:,:,ir,j],ap[:,:,:,:,:,:,ir,j],FR[:,:,:,ir,j],ACd[:,:,:,irr,j])
        e = Array(ein"pqrs, pqrs -> "(lr,h))[]
        n = Array(ein"pprr -> "(lr))[]
        verbose && println("Vertical energy = $(e/n)")
        etol += e/n
    end

    verbose && println("e = $(etol/Ni/Nj)")
    return etol/Ni/Nj
end


function optimise_ipeps(A::AbstractArray, h, key; f_tol = 1e-6, opiter = 100, optimmethod = LBFGS(m = 20))
    oc = optcont()
    f(x) = real(energy(h, x, oc, key))
    g(x) = Zygote.gradient(f,x)[1]

    res = optimize(f, g, 
        A, optimmethod, inplace = false,
        Optim.Options(f_tol=f_tol, iterations=opiter,
        extended_trace=true,
        callback=os->writelog(os, key)),
        )
    return res
end

function writelog(os::OptimizationState, key=nothing)
    message = @sprintf("i = %5d\tt = %0.2f\tenergy = %.15f \tgnorm = %.3e\n", os.iteration, os.metadata["time"], os.value, os.g_norm)

    printstyled(message; bold=true, color=:red)
    flush(stdout)

    folder, atype, Ni, Nj, D, χ, tol, maxiter, miniter, verbose = key
    !(isdir(folder)) && mkdir(folder)
    if !(key === nothing)
        logfile = open(joinpath(folder, "D$(D)_χ$(χ)_tol$(tol)_maxiter$(maxiter).log"), "a")
        write(logfile, message)
        close(logfile)
        save(joinpath(folder, "D$(D)_χ$(χ)_tol$(tol)_maxiter$(maxiter).jld2"), "bcipeps", os.metadata["x"])
    end
    return false
end