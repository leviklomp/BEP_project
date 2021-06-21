using Random
using Statistics
using Distributions
using StatsBase
using LinearAlgebra
include("CreateCovarianceMatrices.jl")

function SimGaussian1D(L,dx,ρ_X)
    """Generates a 1D random standard Gaussian field with covariance function ξ
    (in this case same as correlation function ρ)"""
    xas = collect(-L:dx:L)
    N = length(xas)

    Σ = CreateKernelMatrix1D(ρ_X,xas)

    F = eigen(Σ)
    U = F.vectors
    Λ = Diagonal(F.values)
    sqrtΛ = Diagonal(sqrt.(Complex.(F.values)))

    A = U*sqrtΛ
    z = rand(Normal(0,1),N,1)
    x =  A*z

    x = real(x)
    return x
end

function SimGaussian2D(L,M,dx,dy,ρ_X)
    """Generates a 2D random standard Gaussian field with covariance function ξ
    (in this case same as correlation function ρ)"""
    xas = collect(-L:dx:L)
    yas = collect(-M:dy:M)
    Nx = length(xas)
    Ny = length(yas)

    Σ = CreateKernelMatrix2D(ρ_X,xas,yas)

    F = eigen(Σ)
    U = F.vectors
    Λ = Diagonal(F.values)
    sqrtΛ = Diagonal(sqrt.(Complex.(F.values)))

    A = U*sqrtΛ
    z = rand(Normal(0,1),Nx*Ny,1)
    x = zeros(Nx*Ny) + A*z

    X = zeros(Ny,Nx)*im
    for i = 1:Nx
        X[:,i] = x[1+Ny*(i-1):Ny*i]
    end
    X = real(X)
    return X
end

function SimGamma1D(m,L,dx,ρ_X)
    xas = collect(-L:dx:L)
    N = length(xas)

    G = zeros(N)
    for s = 1:2*m
        X = SimGaussian1D(L,dx,ρ_X)
        G += (1/2)*X.^2
    end
    return G
end

function SimGamma2D(m,L,M,dx,dy,ρ_X)
    xas = collect(-L:dx:L)
    yas = collect(-M:dy:M)
    Nx = length(xas)
    Ny = length(yas)

    G = zeros(Ny,Nx)
    for s = 1:2*m
        X = SimGaussian2D(L,M,dx,dy,ρ_X)
        G += (1/2)*X.^2
    end
    return G
end

function SimBeta1D(m,n,L,dx,ρ_X)

    G_m = SimGamma1D(m,L,dx,ρ_X)
    G_n = SimGamma1D(n,L,dx,ρ_X)

    B_nm =  G_m./(G_m + G_n)
    return B_nm
end

function SimBeta2D(m,n,L,M,dx,dy,ρ_X)

    G_m = SimGamma2D(m,L,M,dx,dy,ρ_X)
    G_n = SimGamma2D(n,L,M,dx,dy,ρ_X)

    B_nm = G_m./(G_m + G_n)
    return B_nm
end

function SimLognormal1D(μ,σ,L,dx,ρ_X)
    X = SimGaussian1D(L,dx,ρ_X)
    L = exp.(μ.+σ*X)
    return L
end

function SimLognormal2D(μ,σ,L,M,dx,dy,ρ_X)
    X = SimGaussian2D(L,M,dx,dy,ρ_X)
    L = exp.(μ.+σ*X)
    return L
end

##Extra Random Fields
#=
function SimT1D(ν,L,dx,ρ_X
    N = length(collect(-L:dx:L))
    w = rand(Chisq(ν))
    Z = SimGaussian1D(L,dx,ρ_X)
    T = Z./(sqrt(w/ν))
    return T
end
=#
function SimT2D(ν,L,M,dx,dy,ρ_X)
    Nx = length(collect(-L:dx:L))
    Ny = length(collect(-M:dy:M))
    W = rand(Chisq(ν))
    Z = SimGaussian2D(L,M,dx,dy,ρ_X)
    T = Z./(sqrt(W/ν))
    return T
end
## Not correct probably
function SimChiSq1D(n,L,dx,ρ_X)
    xas = collect(-L:dx:L)
    N = length(xas)

    W = zeros(N)
    for s = 1:n
        X = SimGaussian1D(L,dx,ρ_X)
        W += X.^2
    end
    return W
end

function SimChiSq2D(n,L,M,dx,dy,ρ_X)
    xas = collect(-L:dx:L)
    yas = collect(-M:dy:M)
    Nx = length(xas)
    Ny = length(yas)

    W = zeros(Ny,Nx)
    for s = 1:n
        X = SimGaussian2D(L,M,dx,dy,ρ_X)
        W += X.^2
    end
    return W
end


function SimFisher1D(m,n,L,dx,ρ_X)
    W_m = SimChiSq1D(m,L,dx,ρ_X)
    W_n = SimChiSq1D(n,L,dx,ρ_X)
    F = (W_m/m)./(W_n/n)
    return F
end

function SimFisher2D(m,n,L,M,dx,dy,ρ_X)
    W_m = SimChiSq2D(m,L,M,dx,dy,ρ_X)
    W_n = SimChiSq2D(n,L,M,dx,dy,ρ_X)
    F = (W_m/m)./(W_n/n)
    return F
end
