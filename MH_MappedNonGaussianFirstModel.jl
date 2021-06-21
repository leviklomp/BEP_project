using Random
using Statistics
using Distributions
using StatsBase
using Plots
using LinearAlgebra
using JLD
using ForwardDiff
include("SimulateRF.jl")
include("MetropolisHastings.jl")
include("CreateSystem.jl")

##
function logdetJacobian_g(θ,R)
    f(x) = inv_mapping_g(θ,x)
    return real(logdet(ForwardDiff.jacobian(x->f(x), R)))
end

function mapping_g_θ(θ,x)
    μ = θ[1]
    σ = θ[2]
    return exp.(μ .+σ*x)
end

function inv_mapping_g(θ,r)
    μ = θ[1]
    σ = θ[2]
    return real(log.(r) .-μ)/σ #lognormal example
end
##Posterior calculation for a given kernel density k
function posteriorexample2d(ρ,Y,xas,yas) #Example
    """Example"""
    Nx = length(xas)
    Ny = length(yas)
    N = Nx*Ny
    #ρ = [μ;A;B;C;ν2] parameters
    R = ρ[1:N]
    η = ρ[N+1:N+3]
    θ = ρ[N+4:N+5]
    ν2 = ρ[end]
    #creating covariance matrix using a kernel
    kernelexample2d(t_n,t_m) = ρ_X(t_n,t_m,η)
    Σ_η = CreateKernelMatrix2D(kernelexample2d,xas,yas)
    #calculating posterior ∝ scalar
    if minimum(R)<=0
        log_likelihood = -Inf
    else
        log_likelihood = MvNormalpdf(zeros(N),ν2*Matrix(I,N,N),Y-R)+MvNormalpdf(zeros(N),Σ_η,inv_mapping_g(θ,R))+logdetJacobian_g(θ,R)
    end
    v_η = .25*[1;1;1]
    v_θ = [0.5;0.5]
    η_tilde = [1;1;0.5]
    θ_tilde = [-1;1]
    μ_ν2 = 1
    σ_ν2 = 0.1

    log_prior_η =  MvNormalpdf(η_tilde,v_η.*Matrix(I,length(η),length(η)),η)
    log_prior_θ = log(prod(pdf.(Normal.(θ_tilde,v_θ),θ)))
    log_prior_ν2 = log(pdf(Normal(μ_ν2,σ_ν2),ν2))


    return log_likelihood+log_prior_η+log_prior_ν2+log_prior_θ
end

##Data we research
function ρ_X(t1,t2,η) #Correlation function (in interval [-1,1])

    return η[1]*cos(η[2]*norm(t1-t2))*exp(-η[3]*norm(t1-t2))
end
function ρ_Xτ(τ,η) #Correlation function (in interval [-1,1])


    return η[1]*cos(η[2]*norm(τ))*exp(-η[3]*norm(τ))
end

## Creating the data
L = 1
M = L
dx = 0.25
dy = dx
xas = collect(-L:dx:L)
yas = collect(-M:dy:M)
Nx = length(xas)
Ny = length(yas)
ν2 = 1 #noise variance
N = Nx*Ny
d = N+6 #number of parameters [μ,A,B,C,σ]
η = [1;1;0.5]
θ = [-1;1]
ρX(t1,t2) = ρ_X(t1,t2,η)

R = SimLognormal2D(θ[1],θ[2],L,M,dx,dy,ρX)
ϵ = rand(Normal(0,ν2),Ny,Nx)

SampleData_mat = R + ϵ
Plots.heatmap(xas,yas,R,title="Simulated RF",xlabel="x",ylabel="y",color = cgrad(:rainbow))
Plots.heatmap(xas,yas,SampleData_mat,title="Simulated observed data",xlabel="x",ylabel="y",color = cgrad(:rainbow))
SampleData = MatrixToVector(SampleData_mat)
##Defininng posterior, and transition (ν is important for searching ρ)
β = 0.005
p(ρ) = posteriorexample2d(ρ,SampleData,xas,yas)
q(ρ1,ρ2) = MvNormalpdf(ρ1,β*Matrix(I,d,d),ρ2)
q(ρ1,ρ2) = 1 #transition_density is symmetric so use simple function to prevent calculations
Q(ρ) = ρ+rand(Normal(0,β),d,1) #transition sample
##Initial values
#x0 = rand(Uniform(),d)
R_0 = zeros(N)
for i = 1:N
    R_0[i] = maximum([SampleData[i] 0.01])
end
x0 = [R_0;.5;.5;.5;-.5;.5;.5] #initial point is important (vary with stepsize in transition_density (depends on σ))
n = 50000
burnin = 500 #Int(floor(0.1*n))
logvariant = true

##Metropolis-Hastings algorithm
MH_data,acceptancerate = MetropolisHastingsAlgorithm(p,x0,q,Q,n,burnin,logvariant)

MH_data_T = transpose(MH_data)
##Parameter path (a line per dimension)
##Estimated parameters using sample mean
ρ_MH = mean(MH_data,dims = 2)
R_MH = ρ_MH[1:N]

R_MH_mat = zeros(Ny,Nx)
for i = 1:Nx
    R_MH_mat[:,i] = R_MH[(i-1)*Ny+1:i*Ny]
end

p1hm = Plots.heatmap(xas,yas,R_MH_mat,fill=true,color = cgrad(:rainbow),title = "Estimated RF" )
p2hm = Plots.heatmap(xas,yas,R,fill=true,color = cgrad(:rainbow),title = "Real RF")
p12hm = plot(p1hm,p2hm)
savefig(p12hm,"MappedNonGaussianFirstEstANDRealRF.png")

p_obs = Plots.heatmap(xas,yas,SampleData_mat,fill=true,color = cgrad(:rainbow),title = "Obs. Data")
plot(p_obs)
savefig(p_obs,"MappedNonGaussianMHobsdata.png")

#mean(μ_MH),var(μ_MH)
A_MH = ρ_MH[N+1]
B_MH = ρ_MH[N+2]
C_MH = ρ_MH[N+3]
μ_MH = ρ_MH[N+4]
σ_MH = ρ_MH[N+5]
ν2_MH = ρ_MH[end]
##
p1 = Plots.plot(collect(burnin:1:n+1),MH_data_T,legend = false
    ,title = "Random walk through parameter space P using MH  algorithm"
    ,titlefont = font(10))
savefig(p1,"MappedNonGaussianFirstMHwholeWalk.png")

N_mh = length(collect(burnin:1:n+1))
pA = Plots.plot(collect(burnin:1:n+1),[MH_data_T[:,N+1],ones(N_mh)*η[1],ones(N_mh)*A_MH],legend = false
    ,title = "A"
    ,titlefont = font(20))
pB = Plots.plot(collect(burnin:1:n+1),[MH_data_T[:,N+2],ones(N_mh)*η[2],ones(N_mh)*B_MH],legend = false
    ,title = "B"
    ,titlefont = font(20))
pC = Plots.plot(collect(burnin:1:n+1),[MH_data_T[:,N+3],ones(N_mh)*η[3],ones(N_mh)*C_MH],legend = true
    ,title = "C"
    ,titlefont = font(20), legendfont = font(15),labels = ["path" "real value" "Bayes est."])
pμ = Plots.plot(collect(burnin:1:n+1),[MH_data_T[:,N+4],ones(N_mh)*θ[1],ones(N_mh)*μ_MH],legend = false
    ,title = "mu"
    ,titlefont = font(20))
pσ = Plots.plot(collect(burnin:1:n+1),[MH_data_T[:,N+5],ones(N_mh)*θ[2],ones(N_mh)*σ_MH],legend = false
    ,title = "sigma"
    ,titlefont = font(20))
pν2 = Plots.plot(collect(burnin:1:n+1),[MH_data_T[:,end],ones(N_mh)*ν2,ones(N_mh)*ν2_MH],legend = false
    ,title = "nu^2"
    ,titlefont = font(20))

pparams = plot(pA,pB,pC,pμ,pσ,pν2,size = (3000,1500))

savefig(pparams,"MappedNonGaussianFirstMHparamsWalk.png")


pABCν2 = Plots.plot(collect(burnin:1:n+1),MH_data_T[:,N+1:end],legend = true, labels = ["A" "B" "C" "mu" "sigma" "nu"]
    ,title = "Random walk through parameter space P using MH-algorithm  "
    ,titlefont = font(10),xlabel = "step",ylabel = "value per ρ")
