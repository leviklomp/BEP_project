"""Metropolis Algorithm on an eigenfunction representation with Gaussian coefficient a"""
using Random
using Plots
using StatPlots

include("MetropolisHastings.jl")
include("MaskOnData.jl")
include("SimulateRF.jl")
include("CreateCovarianceMatrices.jl")
include("AuxiliaryFunctions.jl")
include("WaveRepresentation.jl")

function log_posterior(Y,ν2,a,γ,Mask,E,n,func_γ)

    γ_mat = zeros(n,n)

    for i = 1:n
        γ_mat[i,:] = γ[(i-1)*n+1:i*n]
    end
    γ = γ_mat
    return log_Y(Y,ν2,a,E,Mask)+log_a(a,γ,n,func_γ)+log_π_ν2(ν2)+log_π_γ(γ)
end

function log_Y(Y,ν2,a,E,Mask)
    N = length(Y)
    log_Y = MvNormalpdf(Mask*E*a,ν2*Matrix(I,N,N),Y)
    return log_Y
end

function log_a(a,γ,n,func_γ)
    func_given_γ(l,m,l2,m2) = func_γ(l,m,l2,m2,γ)
    C_γ = CovarianceMatrixCoefficients2D(n,func_given_γ)
    N = length(a)
    return MvNormalpdf(zeros(N),C_γ,a)
end

function log_π_ν2(ν2)
    return log(pdf(Normal(5,1),ν2))
end

function log_π_γ(γ)
    N = size(γ)[1]
    val = 1
    for i = 1:N, j = 1:N
        val = val*pdf(Exponential(1/(i*j)),γ[i,j])
    end
    return log(val)
end



#Plots.heatmap(collect(1:size(E)[1]),collect(1:size(E)[2]),E)
global L = 2.5
global M = L
dx = 0.25
dy = dx
xas = collect(-L:dx:L)
yas = collect(-M:dy:M)
Nx  = length(xas)
Ny  = length(yas)
N = Nx*Ny
LL = (-.9,-.9)
UR = (.5,.5)
(Mask,Coordset) = CreateRectangularMask2D(xas,yas,LL,UR)

#if we dont want to use mask take Mask = Identity matrix
Mask = Matrix(I,N,N)
##
function eigenf(x,y,l,m)
    return exp(-im*π*(x*l/L+y*m/M))
end

n_real = 5 #number of eigenfunctions
##
Mask
E_real = GridEigenFunctionMatrix2D(xas,yas,eigenf,n_real)

##function for covmat of the coefficients
γ_real = [1/(l*m) for l = 1:n_real, m = 1:n_real]

function F(l,m,l2,m2,γ)
    return (l==l2)*(m==m2)*γ[l,m]#γ[1]*cos(γ[2]*abs(l-l2))+exp(-γ[3]*abs(m-m2))
end
F_γ(l,m,l2,m2) = F(l,m,l2,m2,γ_real)

C_γ = CovarianceMatrixCoefficients2D(n_real,F_γ)
#a = rand(MvNormal(zeros(n^2),C_γ))

a_real = MvNormalSample(zeros(n_real^2),C_γ)
##generating data
R = E_real*a_real
Nx = length(xas)
Ny = length(yas)
R_real = zeros(Ny,Nx)
for i = 1:Nx
    R_real[:,i] = R[1+(i-1)*Nx:i*Nx]
end

ν2_real = 5
ϵ = rand(Normal(0,sqrt(ν2_real)),Ny,Nx)
Y = R_real+ϵ
p1 = Plots.heatmap(xas,yas,R_real,fill=true,fillcolor = cgrad(:rainbow),title="Gaussian RF without noise",xlabel = "x",ylabel="y",aspect_ratio=:equal,titlefont = font(20))
p2 = Plots.heatmap(xas,yas,Y,fill=true,fillcolor = cgrad(:rainbow),title="Gaussian RF with noise",xlabel = "x",ylabel="y",aspect_ratio=:equal,grid = false,titlefont = font(20))
p3 = plot(p1,p2,size = (1200,600))
savefig(p3,"EFRFGaussian2figures.png")
## additional mask
maskxas = collect(LL[1]:dx:UR[1])
NMx = length(maskxas)
maskyas = collect(LL[2]:dy:UR[2])
NMy = length(maskyas)
maskval = ones(NMy,NMx)
Plots.heatmap!(maskxas,maskyas,maskval,color = "black",colorbar = false)


Ynew = Mask*MatrixToVector(Y)
Coordset

n_guess = 5 #our assumption about the number of different eigenfunctions n^2 (in one dimension)
E_guess = GridEigenFunctionMatrix2D(xas,yas,eigenf,n_guess)
γ_0 = ones(n_guess,n_guess)
γ_0 = MatrixToVector(γ_0)
θ_0 = [zeros(n_guess^2);γ_0;1]
β = .02
d = length(θ_0)
q(θ1,θ2) = 1 #transition prob
Q(θ) = θ+rand(Normal(0,β),d,1) #transition sample
log_posterior_MH(θ) = log_posterior(Ynew,θ[end],θ[1:n_guess^2],θ[n_guess^2+1:end-1],Mask,E_guess,n_guess,F)
steps = 25000
burnin = 500
##
(Path,arate) = MetropolisHastingsAlgorithm(log_posterior_MH,θ_0,q,Q,steps,burnin,true)
MH_data_T = transpose(Path)
##

mean_MH = mean(Path,dims = 2)
var_MH = var(Path,dims = 2)
cor_MH = cor(Path,dims = 2)

a_hat = mean_MH[1:n_guess^2]
γ_hat = mean_MH[n_guess^2+1:end-1]
ν2_hat = mean_MH[end]


γ_mat_hat = zeros(n_guess,n_guess)
for i = 1:n_guess
    γ_mat_hat[i,:] = γ_hat[(i-1)*n_guess+1:i*n_guess]
end



mean(a_hat-a_real)
var(a_hat-a_real)

mean(γ_mat_hat-γ_real)
var(γ_mat_hat-γ_real)

A_hat = zeros(n_guess,n_guess)
for i = 1:n_guess
    A_hat[:,i] = a_hat[(i-1)*n_guess+1:i*n_guess]
end

R_hat = Fourier_RF_2D(A_hat,xas,yas,L,M)
p1 = Plots.heatmap(xas,yas,R_hat,title = "Estimated R",color = cgrad(:rainbow));
p2 = Plots.heatmap(xas,yas,R_real, title = "Real R",color = cgrad(:rainbow));
p3=plot(p1,p2,size = (600,300))
savefig(p3,"EFRFGaussianEstANDRealRF.png")
##Parameter path (a line per dimension)
N_MH = length(collect(burnin:1:steps+1))
γ_ones_mat = ones(length(γ_hat),N_MH)
est_values_γ = transpose(γ_hat.*γ_ones_mat)

p1 = Plots.plot(collect(burnin:1:steps+1),MH_data_T,legend = false
    ,title = "Random walk through parameter space P using MH  algorithm"
    ,titlefont = font(20),xlabel = "step",ylabel = "value")
savefig(p1,"EFRFGaussianWholeWalk.png")
pa = Plots.plot(collect(burnin:1:steps+1),MH_data_T[:,1:n_guess^2],legend = false
    ,title = "EF coeffs a"
    ,titlefont = font(15),xlabel = "step",ylabel = "value")
pγ = Plots.plot(collect(burnin:1:steps+1),MH_data_T[:,n_guess^2+1:end-1],legend = false
    ,title = "COV coeffs gamma"
    ,titlefont = font(15),xlabel = "step",ylabel = "value")
Plots.plot(collect(burnin:1:steps+1),est_values_γ)

pν2 = Plots.plot(collect(burnin:1:steps+1),[MH_data_T[:,end],ones(N_MH)*ν2_real,ones(N_MH)*ν2_hat],legend = false
    ,title = "nu^2"
    ,titlefont = font(15),xlabel = "step",ylabel = "value")

p_params = plot(pa,pγ,pν2)

savefig(p_params,"EFMHGaussianSeperatedTrace.png")
#corrplot(MH_data_T[:,n_guess^2+1:end],label = append!(["gam$i" for i = 1:n_guess],["v2"]),fillcolor=cgrad(),title = "Matrix Scatterplot for
#γ and ν")
#cor(MH_data_T)




##creating several masks and comparing the variances
#function TestMasksMH(log_posterior_MH,θ_0,q,Q,steps,burnin,xas,yas,Ynew,E_guess,n_guess,F,num_masks,mask_step)
θ_0 = [zeros(n_guess^2);ones(n_guess+1)]
β = .01
d = length(θ_0)
q(θ1,θ2) = 1 #transition prob
Q(θ) = θ+rand(Normal(0,β),d,1)

mask_step = dx
num_masks = Int(floor(minimum([L,M])/dx)-1)

num_params = length(θ_0)

MEAN_matrix = zeros(num_params,num_masks)
VAR_matrix = zeros(num_params,num_masks)
SURF_vec = zeros(num_masks)
a_rate_vec = zeros(num_masks)
steps = 1000
burnin = 100

for m = 1:num_masks
    #Random.seed!(1)
    println("--------------------------------------------------------------------------------")
    LL = (-mask_step*m,-mask_step*m)
    UR = (mask_step*m,mask_step*m)
    SURF_vec[m] = (2*mask_step*m)^2
    (Mask,Coordset) = CreateRectangularMask2D(xas,yas,LL,UR)
    Ynew = Mask*E_real*a_real
    E_guess = CoordsetEigenFunctionMatrix2D(Coordset,eigenf,n_guess)
    #running MH algo for given mask
    log_posterior_MH(θ) = log_posterior(Ynew,θ[end],θ[1:n_guess^2],θ[n_guess^2+1:end-1],Mask,E_guess,n_guess,F)
    (Path,arate) = MetropolisHastingsAlgorithm(log_posterior_MH,θ_0,q,Q,steps,burnin,true)
    #calculating mean and variances of parameters
    a_rate_vec[m] = arate
    MEAN_matrix[:,m] = mean(Path,dims = 2)
    VAR_matrix[:,m] = var(Path,dims = 2)
end
    #return MEAN_matrix,VAR_matrix,SURF_vec
#end
Plots.plot(SURF_vec,transpose(VAR_matrix),legend = false)

a_rate_vec
