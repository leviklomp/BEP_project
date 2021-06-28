include("MaskOnData.jl")
include("AuxiliaryFunctions.jl")
include("CreateCovarianceMatrices.jl")
include("MetropolisHastings.jl")
include("WaveRepresentation.jl")
using StatPlots
using Statistics
using JLD
function eigenf(x,y,l,m)
    return exp(-im*π*(x*l/L+y*m/M))
end

##
function F(l,m,l2,m2,γ)
    return (l==l2)*(m==m2)*γ[l,m]#γ[1]*cos(γ[2]*abs(l-l2))+exp(-γ[3]*abs(m-m2))
end

## Defining the coordinate field
L = 5
M = L
dx = 0.25
dy = dx
xas = collect(-L:dx:L)
yas = collect(-M:dy:M)
Nx  = length(xas)
Ny  = length(yas)
N = Nx*Ny
##Generating the Gaussian RF
n_real = 5 #number of eigenfunctions
E_real = GridEigenFunctionMatrix2D(xas,yas,eigenf,n_real) #EF matrix

#Generating Coefficients
γ_real = [1/(l*m) for l = 1:n_real, m = 1:n_real]
F_γ(l,m,l2,m2) = F(l,m,l2,m2,γ_real)
C_γ = CovarianceMatrixCoefficients2D(n_real,F_γ) #Coefficients cov matrix defined by function F
a_real = MvNormalSample(zeros(n_real^2),C_γ)

#Gaussian RF
X = E_real*a_real
X_real = zeros(Ny,Nx)
for i = 1:Nx
    X_real[:,i] = X[1+(i-1)*Ny:i*Ny]
end
#Noise
ν2_real = 1
ϵ = rand(Normal(0,sqrt(ν2_real)),Ny,Nx)

#Generated data
Y = X_real + ϵ
## Adding a rectangular mask
LL = (-2,-2)
UR = (3,3)

(Mask,Coordset) = CreateRectangularMask2D(xas,yas,LL,UR)
NoMask = Matrix(I,N,N) #identity matrix if there is no mask!

Y_Mask = Mask*MatrixToVector(Y)
Y_NoMask = MatrixToVector(Y) #Data points after mask, with coordinates Coordset

p1 = PlotDataWithRectangularMask2D(Y,xas,yas,LL,UR)

savefig(p1,"WithWithoutMaskPlotGaussianEF.png")
## Applying Metropoolis-Hastings algorithm and compare the estimation with and without mask
include("MH_scheme_GaussianSecondModel.jl")

n_guess = 5 #our assumption about the number of different eigenfunctions n^2 (in one dimension)
E_guess = GridEigenFunctionMatrix2D(xas,yas,eigenf,n_guess)

#initial values
a_0 = zeros(n_guess^2)
γ_0 = ones(n_guess,n_guess)
γ_0 = MatrixToVector(γ_0)
ν2_0 = 1
ρ_0 = [a_0;γ_0;ν2_0]
d = length(ρ_0)

#transition kernel/density
q(ρ1,ρ2) = 1 #transition prob (symmetrical so assume trivial, because this prevents calculations!)
β = .02
Q(ρ) = ρ+rand(Normal(0,β),d,1) #transition sample

#posteriors
log_posterior_Mask(ρ) = log_posterior(Y_Mask,ρ[end],ρ[1:n_guess^2],ρ[n_guess^2+1:end-1],Mask,E_guess,n_guess,F)
log_posterior_NoMask(ρ) = log_posterior(Y_NoMask,ρ[end],ρ[1:n_guess^2],ρ[n_guess^2+1:end-1],NoMask,E_guess,n_guess,F)

steps = 25000
burnin = 500

# Metropolis-Hastings applied to two cases (mask and no-mask) but same observed data
(PathMask,arateMask) = MetropolisHastingsAlgorithm(log_posterior_Mask,ρ_0,q,Q,steps,burnin,true)
MH_data_Mask = transpose(PathMask)
(PathNoMask,arateNoMask) = MetropolisHastingsAlgorithm(log_posterior_NoMask,ρ_0,q,Q,steps,burnin,true)
MH_data_NoMask = transpose(PathNoMask)

#=
save("MaskMHdataEFGaussian.jld","data",MH_data_Mask)
save("NoMaskMHdataEFGaussian.jld","data",MH_data_NoMask)
=#
##Calculate estimates (posterior mean)

#=
MH_data_Mask = load("MaskMHdataEFGaussian.jld")["data"]
MH_data_NoMask = load("NoMaskMHdataEFGaussian.jld")["data"]
PathMask = transpose(MH_data_Mask)
PathNoMask = transpose(MH_data_NoMask)
=#
#MASK

mean_Mask_MH = mean(PathMask,dims = 2)
var_Mask_MH = var(PathMask,dims = 2)
cor_Mask_MH = cor(PathMask,dims = 2)

a_hat_Mask = mean_Mask_MH[1:n_guess^2]
γ_hat_Mask = mean_Mask_MH[n_guess^2+1:end-1]
ν2_hat_Mask = mean_Mask_MH[end]

γ_mat_hat_Mask = zeros(n_guess,n_guess)
for i = 1:n_guess
    γ_mat_hat_Mask[i,:] = γ_hat_Mask[(i-1)*n_guess+1:i*n_guess]
end

#NO MASK
mean_NoMask_MH = mean(PathNoMask,dims = 2)
var_NoMask_MH = var(PathNoMask,dims = 2)
cor_NoMask_MH = cor(PathNoMask,dims = 2)

a_hat_NoMask = mean_NoMask_MH[1:n_guess^2]
γ_hat_NoMask = mean_NoMask_MH[n_guess^2+1:end-1]
ν2_hat_NoMask = mean_NoMask_MH[end]

γ_mat_hat_NoMask = zeros(n_guess,n_guess)
for i = 1:n_guess
    γ_mat_hat_NoMask[i,:] = γ_hat_NoMask[(i-1)*n_guess+1:i*n_guess]
end


real_params = [a_real;MatrixToVector(γ_real);ν2_real]

mse_mask = (1/25)*sum((real_params[26:50]-mean_Mask_MH[26:50]).^2)
mse_nomask = (1/25)*sum((real_params[26:50]-mean_NoMask_MH[26:50]).^2)

##calculating estimated RF's

A_hat_Mask = zeros(n_guess,n_guess)
A_hat_NoMask = zeros(n_guess,n_guess)
for i = 1:n_guess
    A_hat_Mask[:,i] = a_hat_Mask[(i-1)*n_guess+1:i*n_guess]
    A_hat_NoMask[:,i] = a_hat_NoMask[(i-1)*n_guess+1:i*n_guess]
end

X_hat_Mask = Fourier_RF_2D(A_hat_Mask,xas,yas,L,M)
X_hat_NoMask = Fourier_RF_2D(A_hat_NoMask,xas,yas,L,M)

pMask = Plots.heatmap(xas,yas,X_hat_Mask,title = "Estimated RF with Mask",color = cgrad(:rainbow));
pNoMask = Plots.heatmap(xas,yas,X_hat_NoMask, title = "Estimated RF without Mask",color = cgrad(:rainbow));
pReal= Plots.heatmap(xas,yas,X_real, title = "Real RF",color = cgrad(:rainbow));

p3plots = plot(pMask,pNoMask,pReal)

savefig(p3plots,"WithWihoutMaskEstANDRealRF.png")

##Parameter path Mask MH (a line per dimension)
N_MH = length(collect(burnin:1:steps+1))


p1mask = Plots.plot(collect(burnin:1:steps+1),MH_data_Mask,legend = false
    ,title = "Random walk through parameter space P using MH  algorithm"
    ,titlefont = font(15),xlabel = "step",ylabel = "value")
savefig(p1mask,"MaskWholeTraceEFGaussian.png")
pamask = Plots.plot(collect(burnin:1:steps+1),MH_data_Mask[:,1:n_guess^2],legend = false
    ,title = "EF coeffs a"
    ,titlefont = font(15),xaxis = nothing,xlabel = "step",ylabel = "value")
pγmask = Plots.plot(collect(burnin:1:steps+1),MH_data_Mask[:,n_guess^2+1:end-1],legend = false
    ,title = "COV coeffs gamma"
    ,titlefont = font(15),xaxis = nothing,xlabel = "step",ylabel = "value")
#Plots.plot(collect(burnin:1:steps+1),est_values_γ)

pν2mask = Plots.plot(collect(burnin:1:steps+1),[MH_data_Mask[:,end],ones(N_MH)*ν2_real,ones(N_MH)*ν2_hat_Mask,MovingAverage(MH_data_Mask[:,end])],legend = false
    ,title = "nu^2"
    ,titlefont = font(15),xaxis = nothing,xlabel = "step",ylabel = "value")

p_paramsMask = plot(pamask,pγmask,pν2mask)
savefig(p_paramsMask,"MaskSeperateTracePlotGaussianEF.png")
savefig(pν2mask,"Masknu2traceplotGaussianEF.png")
##Parameter path No Mask(a line per dimension)
N_MH = length(collect(burnin:1:steps+1))
γ_ones_mat = ones(length(γ_hat),N_MH)
est_values_γ = transpose(γ_hat.*γ_ones_mat)

p1nomask = Plots.plot(collect(burnin:1:steps+1),MH_data_NoMask,legend = false
    ,title = "Random walk through parameter space P using MH  algorithm"
    ,titlefont = font(15),xlabel = "step",ylabel = "value")
savefig(p1nomask,"NoMaskWholeTraceEFGaussian.png")
panomask = Plots.plot(collect(burnin:1:steps+1),MH_data_NoMask[:,1:n_guess^2],legend = false
    ,title = "EF coeffs a"
    ,titlefont = font(15),xaxis = nothing,xlabel = "step",ylabel = "value")
pγnomask = Plots.plot(collect(burnin:1:steps+1),MH_data_NoMask[:,n_guess^2+1:end-1],legend = false
    ,title = "COV coeffs gamma"
    ,titlefont = font(15),xaxis = nothing,xlabel = "step",ylabel = "value")
#Plots.plot(collect(burnin:1:steps+1),est_values_γ)

pν2nomask = Plots.plot(collect(burnin:1:steps+1),[MH_data_NoMask[:,end],ones(N_MH)*ν2_real,ones(N_MH)*ν2_hat_NoMask,MovingAverage(MH_data_NoMask[:,end])],legend = false
    ,title = "nu^2"
    ,titlefont = font(15),xaxis = nothing,xlabel = "step",ylabel = "value")

p_paramsNoMask = plot(panomask,pγnomask,pν2nomask)
savefig(p_paramsNoMask,"NoMaskSeperateTracePlotGaussianEF.png")
savefig(pν2nomask,"NoMasknu2traceplotGaussianEF.png")
##
pdens = density(MH_data_Mask[:,end],label = ["Est. with Mask"],color="blue")
density!(MH_data_NoMask[:,end],label = ["Est. without Mask"],color = "green")
Plots.plot!([ν2_real;ν2_real],[0;22],label = ["Real value"],color = "red")
Plots.plot!([ν2_hat_Mask;ν2_hat_Mask],[0;22],label = ["Est. on mask"],color = "blue")
Plots.plot!([ν2_hat_NoMask;ν2_hat_NoMask],[0;22],label = ["Est. on no mask"],color = "green")
#Plots.plot!(a_real[1]*ones(11),collect(0:0.1:1),label = ["Real value"])
savefig(pdens,"nu2EPDFComparisonMaskNoMask.png")
