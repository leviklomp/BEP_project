using Random
using Plots
using StatPlots

include("MetropolisHastings.jl")
include("MaskOnData.jl")
include("SimulateRF.jl")
include("CreateCovarianceMatrices.jl")
include("AuxiliaryFunctions.jl")
include("WaveRepresentation.jl")

function log_posterior(Y,ν2,b,γ,v,Mask,E,n,func_γ)
    γ_mat = zeros(n,n)

    for i = 1:n
        γ_mat[i,:] = γ[(i-1)*n+1:i*n]
    end
    γ = γ_mat
    return log_Y(Y,ν2,b,E,Mask)+log_b(b,γ,v,n,func_γ)+log_π_ν2(ν2)+log_π_γ(γ)+log_π_v(v)
end

function log_Y(Y,ν2,b,E,Mask)
    N = length(Y)
    log_Y = MvNormalpdf(Mask*E*b,ν2*Matrix(I,N,N),Y)
    return log_Y
end

function log_b(b,γ,v,n,func_γ)
    func_given_γ(l,m,l2,m2) = func_γ(l,m,l2,m2,γ)
    C_γ = CovarianceMatrixCoefficients2D(n,func_given_γ)
    N = length(b)
    return MvTDistpdf(v,zeros(N),C_γ,b)
end

function log_π_ν2(ν2)
    return log(pdf(Normal(2.5,0.5),ν2))
end

function log_π_γ(γ)
    N = size(γ)[1]
    val = 1
    for i = 1:N, j = 1:N
        val = val*pdf(Exponential(1/(i*j)),γ[i,j])
    end
    return log(val)
end

function log_π_v(v)
    μ_v = 10
    σ_v = 0.5
    return log(pdf(Normal(μ_v,σ_v),v))
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
#LL = (-.9,-.9)
#UR = (.5,.5)

#(Mask,Coordset) = CreateRectangularMask2D(xas,yas,LL,UR)

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
v_real = 10 #real degrees of freedom
a_real = MvNormalSample(zeros(n_real^2),C_γ)
b_real = a_real/(sqrt(rand(Chisq(v_real))/v_real))
##generating data
R = E_real*b_real
Nx = length(xas)
Ny = length(yas)
R_real = zeros(Ny,Nx)
for i = 1:Nx
    R_real[:,i] = R[1+(i-1)*Nx:i*Nx]
end

ν2_real = 2.5
ϵ = rand(Normal(0,sqrt(ν2_real)),Ny,Nx)
Y = R_real+ϵ
p1 = Plots.heatmap(xas,yas,R_real,fill=true,fillcolor = cgrad(:rainbow),title="Eigenfunction representation of a 2D random field \n with Gaussian coefficients",xlabel = "x",ylabel="y",aspect_ratio=:equal,titlefont = font(8))
p2 = Plots.heatmap(xas,yas,Y,fill=true,fillcolor = cgrad(:rainbow),title="Eigenfunction representation of a 2D random field (with noise) \n with Gaussian coefficients",xlabel = "x",ylabel="y",aspect_ratio=:equal,grid = false,titlefont = font(8))
p3 = plot(p1,p2)
savefig(p3,"EFMHOtherNonGaussian2figures.png")
            
## additional mask
#=            
maskxas = collect(LL[1]:dx:UR[1])
NMx = length(maskxas)
maskyas = collect(LL[2]:dy:UR[2])
NMy = length(maskyas)
maskval = ones(NMy,NMx)
Plots.heatmap!(maskxas,maskyas,maskval,color = "black")


Coordset
=#
Ynew = Mask*MatrixToVector(Y)         
n_guess = 5
E_guess = GridEigenFunctionMatrix2D(xas,yas,eigenf,n_guess) 
γ_0 = ones(n_guess,n_guess)
γ_0 = MatrixToVector(γ_0)
ρ_0 = [zeros(n_guess^2);γ_0;10;1]
β = .02
d = length(ρ_0)
q(ρ1,ρ2) = 1 #transition prob
Q(ρ) = ρ+rand(Normal(0,β),d,1) #transition sample
log_posterior_MH(ρ) = log_posterior(Ynew,ρ[end],ρ[1:n_guess^2],ρ[n_guess^2+1:end-1],ρ[end-2],Mask,E_guess,n_guess,F)
steps = 25000
burnin = 500
(Path,arate) = MetropolisHastingsAlgorithm(log_posterior_MH,ρ_0,q,Q,steps,burnin,true)
MH_data_T = transpose(Path)
##Parameter etimates

mean_MH = mean(Path,dims = 2)
var_MH = var(Path,dims = 2)
cor_MH = cor(Path,dims = 2)

b_hat = mean_MH[1:n_guess^2]
γ_hat = mean_MH[n_guess^2+1:n_guess^2+length(γ_0)]
v_hat = mean_MH[end-1]
ν2_hat = mean_MH[end]

γ_mat_hat = zeros(n_guess,n_guess)
for i = 1:n_guess
    γ_mat_hat[i,:] = γ_hat[(i-1)*n_guess+1:i*n_guess]
end

mean(b_hat-b_real)
var(b_hat-b_real)

mean(γ_mat_hat-γ_real)
var(γ_mat_hat-γ_real)



##plotting the estimated RF
B_hat = zeros(n_guess,n_guess)
for i = 1:n_guess
    B_hat[:,i] = b_hat[(i-1)*n_guess+1:i*n_guess]
end

R_hat = Fourier_RF_2D(B_hat,xas,yas,L,M)
p1 = Plots.heatmap(xas,yas,R_hat,title = "Estimated RF",color = cgrad(:rainbow));
p2 = Plots.heatmap(xas,yas,R_real,title = "Real RF",color = cgrad(:rainbow));
p3 = plot(p1,p2,size=(600,300))
savefig(p3,"EFMHOtherNonGaussianEstANDRealRF.png")
##
N_MH = length(collect(burnin:1:steps+1))
p1 = Plots.plot(collect(burnin:1:steps+1),MH_data_T,legend = false
    ,title = "Random walk through parameter space P using MH  algorithm"
    ,titlefont = font(15),xlabel = "step",ylabel = "value")
savefig(p1,"EFMHOtherNonGaussianWholeWalk.png")
pb = Plots.plot(collect(burnin:1:steps+1), MH_data_T[:,1:n_guess^2],legend = false
    ,title = "EF coeffs b"
    ,titlefont = font(15),xlabel = "step",ylabel = "value", xaxis = nothing )
pγ = Plots.plot(collect(burnin:1:steps+1),MH_data_T[:,n_guess^2+1:end-2],legend = false
    ,title = "COV coeffs gamma"
    ,titlefont = font(15),xlabel = "step",ylabel = "value", xaxis = nothing)
pv = Plots.plot(collect(burnin:1:steps+1),[MH_data_T[:,end-1] v_real*ones(N_MH) v_hat*ones(N_MH)],legend = false
    ,title = "Degree of freedom v"
    ,titlefont = font(15),xlabel = "step",ylabel = "value", xaxis = nothing)
pν2 = Plots.plot(collect(burnin:1:steps+1),[MH_data_T[:,end] ν2_real*ones(N_MH) ν2_hat*ones(N_MH)],legend = false
    ,title = "nu^2"
    ,titlefont = font(15),xlabel = "step",ylabel = "value", xaxis = nothing)


p_params = plot(pb,pγ,pv,pν2)
savefig(p_params,"EFMHOtherNonGaussianSeperatedTraceplot.png")



