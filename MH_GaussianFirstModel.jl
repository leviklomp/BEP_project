using Random
using Statistics
using Distributions
using Plots
using LinearAlgebra
using JLD
include("SimulateRF.jl")
include("MetropolisHastings.jl")
include("CreateSystem.jl")

##Posterior calculation for a given kernel density k
function posteriorexample2d(ρ,Y,xas,yas) #Example
    """Example"""
    Nx = length(xas)
    Ny = length(yas)
    N = Nx*Ny
    #ρ = [μ;A;B;C;ν2] parameters
    R = ρ[1:N] 
    η = ρ[N+1:N+3]
    ν2 = ρ[end]

    #creating covariance matrix using a kernel
    kernelexample2d(t_n,t_m) = ξ_X(t_n,t_m,η)
    Σ_η = CreateKernelMatrix2D(kernelexample2d,xas,yas)

    #likelihood
    log_likelihood = MvNormalpdf(zeros(N),ν2*Matrix(I,N,N),Y-R)+MvNormalpdf(zeros(N),Σ_η,R)

    v_η = 0.25*[1;1;1]
    η_tilde = [1;1;0.1]
    μ_ν2 = 1
    σ_ν2 = 0.1

    #priors
    log_prior_η =  MvNormalpdf(η_tilde,v_η.*Matrix(I,length(η),length(η)),η)
    log_prior_ν2 = log(pdf(Normal(μ_ν2,σ_ν2),ν2))

    return log_likelihood+log_prior_η+log_prior_ν2
end

##Gaussian covariance function
function ξ_X(t1,t2,η)

    return η[1]*cos(η[2]*norm(t1-t2))*exp(-η[3]*norm(t1-t2))
end
function ξ_Xτ(τ,η)


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
d = N+4 #number of parameters [μ,A,B,C,σ]
η = [1;1;0.5]

ξX(t1,t2) = ξ_X(t1,t2,η)

R = SimGaussian2D(L,M,dx,dy,ξX)
ϵ = rand(Normal(0,σ),Ny,Nx)

SampleData_mat = R + ϵ
Plots.heatmap(xas,yas,R,title="Simulated RF",xlabel="x",ylabel="y",color = cgrad(:rainbow))
Plots.heatmap(xas,yas,SampleData_mat,title="Simulated observed data",xlabel="x",ylabel="y",color = cgrad(:rainbow))
SampleData = MatrixToVector(SampleData_mat)
        
##Defininng posterior, and transition (ν is important for searching ρ)
β = 0.02
p(ρ) = posteriorexample2d(ρ,SampleData,xas,yas)
q(ρ1,ρ2) = logMvNormalpdf(ρ1,β*Matrix(I,d,d),ρ2)
q(ρ1,ρ2) = 1 #transition_density is symmetric so use simple form to reduce calculations
Q(ρ) = ρ+rand(Normal(0,β),d,1) #transition sample
##Initial values
x0 = [SampleData;1;1;1;.75]

n = 25000
burnin = 500
logvariant = true

##Metropolis-Hastings algorithm
MH_data,acceptancerate = MetropolisHastingsAlgorithm(p,x0,q,Q,n,burnin,logvariant)
MH_data_T = transpose(MH_data)

##Estimated parameters using sample mean
ρ_MH = mean(MH_data,dims = 2)
R_MH = ρ_MH[1:N]

R_MH_mat = zeros(Ny,Nx)
for i = 1:Nx
    R_MH_mat[:,i] = R_MH[(i-1)*Ny+1:i*Ny]
end


p1hm = Plots.heatmap(xas,yas,R_MH_mat,fill=true,color = cgrad(:rainbow),title = "Bayes estimated RF" )
p2hm = Plots.heatmap(xas,yas,R,fill=true,color = cgrad(:rainbow),title = "Real RF")
p12hm = plot(p1hm,p2hm,size = (600,250))
#savefig(p12hm,"EstANDRealRFGaussianFirstmodel.png")
p_obs = Plots.heatmap(xas,yas,SampleData_mat,fill=true,color = cgrad(:rainbow),title = "Obs. Data")
plot(p1hm,p_obs)

#Bayes estimates
A_MH = ρ_MH[N+1]
B_MH = ρ_MH[N+2]
C_MH = ρ_MH[N+3]
ν2_MH = ρ_MH[N+4]
##Parameter path (a line per dimension)
p1 = Plots.plot(collect(burnin:1:n+1),MH_data_T,legend = false
    ,title = "Random walk through parameter space P using MH  algorithm"
    ,titlefont = font(12))

#savefig(p1,"MHwholeWalkGaussianFirst.png")
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
pν2 = Plots.plot(collect(burnin:1:n+1),[MH_data_T[:,N+4],ones(N_mh)*ν2,ones(N_mh)*ν2_MH],legend = false
    ,title = "nu^2"
    ,titlefont = font(20))

pparams = plot(pA,pB,pC,pν2,size = (1500,1000))

#savefig(pparams,"MHparamsWalkGaussianFirst.png")

pABCν2 = Plots.plot(collect(burnin:1:n+1),MH_data_T[:,N+1:end],legend = true, labels = ["A" "B" "C" "nu"]
    ,title = "Random walk through parameter space P using MH-algorithm  "
    ,titlefont = font(10),xlabel = "step",ylabel = "value per ρ")
