include("MaskOnData.jl")
include("AuxiliaryFunctions.jl")
include("CreateCovarianceMatrices.jl")
include("MH_scheme_GaussianSecondModel.jl")
using StatPlots
using JLD
##
function F(l,m,l2,m2,γ)
    return (l==l2)*(m==m2)*γ[l,m]#γ[1]*cos(γ[2]*abs(l-l2))+exp(-γ[3]*abs(m-m2))
end

## Defining the coordinate field
L = 4
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
Plots.heatmap(xas,yas,Y,color=cgrad(:rainbow))
Y_vec = MatrixToVector(Y)
##
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
β = .015
Q(ρ) = ρ+rand(Normal(0,β),d,1) #transition sample

steps = 25000
burnin = 500
N_sim = Int(minimum([floor((L-dx)/dx),floor((M-dy)/dy)]))

Num_Pixels_arr = zeros(N_sim+1)
MEAN_MH = zeros(d,N_sim+1)
STD_MH = zeros(d,N_sim+1)
VAR_MH = zeros(d,N_sim+1)
Pathsteps = steps-burnin+2
Path_arr = zeros(Pathsteps,d,N_sim+1)

for Sim = 0:N_sim
    println("NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW")
    println("$Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim $Sim")
    LL = (-Sim*dx,-Sim*dy)
    UR = (Sim*dx,Sim*dy)

    if Sim == 0
        Mask = Matrix(I,length(Y_vec),length(Y_vec))
    else
        (Mask,Coordset) = CreateRectangularMask2D(xas,yas,LL,UR)
    end

    Ynew = Mask*Y_vec

    Num_Pixels_arr[Sim+1] = length(Ynew)

    log_posterior(ρ) = log_posterior(Ynew,ρ[end],ρ[1:n_guess^2],ρ[n_guess^2+1:end-1],Mask,E_guess,n_guess,F)

    (Path,arate) = MetropolisHastingsAlgorithm(log_posterior,ρ_0,q,Q,steps,burnin,true)
    MH_data_T = transpose(Path)
    MEAN_MH[:,Sim+1] = mean(MH_data_T,dims = 1)
    STD_MH[:,Sim+1] = std(MH_data_T,dims = 1)
    VAR_MH[:,Sim+1] = var(MH_data_T,dims = 1)
    Path_arr[:,:,Sim+1] = MH_data_T


end



N_sim = 15
d = 51
Num_Pixels_arr

#=
MEAN_MH = load("PerfCompMHexample2.jld")["MEANMH"]
VAR_MH = load("PerfCompMHexample2.jld")["VARMH"]
STD_MH = load("PerfCompMHexample2.jld")["STDMH"]
a_real = load("PerfCompMHexample2.jld")["real_a"]
=#

#calculating MSE
real_params = [a_real;MatrixToVector(γ_real);ν2_real]
MSE_array = zeros(N_sim+1)
for i = 1:N_sim+1
    MSE_array[i] = (1/d)*sum((MEAN_MH[:,i] .- real_params).^2)
end
MSE_array

##
real_params = [a_real;MatrixToVector(γ_real);ν2_real]
reverse(Num_Pixels_arr)
MEAN_measure = abs.(MEAN_MH.-real_params) #deviation of estimate from real value
VAR_measure = VAR_MH
#save("PerfCompMHexampl2.jld","Path",Path_arr,"numpix",Num_Pixels_arr,"MEANMH",MEAN_MH,"VARMH",VAR_MH,"STDMH",STD_MH,"real_a",a_real,"Y",Y)
##Plots VAR and MEAN heatmaps
pHeatmapMeanA = Plots.heatmap(collect(1:N_sim+1),collect(1:n_guess^2),MEAN_measure[1:n_guess^2,:],fill=true,color = cgrad(:lighttest),xlabel = "iteration (from smallest to largest mask ->) ",ylabel = "EF coefficient number",title = "deviation of estimate of EF coefficients 'a' to its real value",size = (500,300),titlefont = font(12))
pHeatmapStdA = Plots.heatmap(collect(1:N_sim+1),collect(1:n_guess^2),STD_MH[1:n_guess^2,:],fill=true,color = cgrad(:lighttest),xlabel = "iteration (from smallest to largest mask ->) ",ylabel = "EF coefficient number",title = "standard deviation of estimate of EF coefficients 'a' ",size = (500,300),titlefont = font(12))

savefig(pHeatmapStdALL,"HeatmapStdALL.png")

pHeatmapMeanGamma = Plots.heatmap(collect(1:N_sim+1),collect(1:n_guess^2),MEAN_measure[n_guess^2+1:end-1,:],fill=true,color = cgrad(:lighttest),xlabel = "iteration (from smallest to largest mask ->) ",ylabel = "COV parameter number",title = "deviation of estimate of COV parameters 'gamma' to its real value",size = (500,300),titlefont = font(11))
pHeatmapStdGamma = Plots.heatmap(collect(1:N_sim+1),collect(1:n_guess^2),STD_MH[n_guess^2+1:end-1,:],fill=true,color = cgrad(:lighttest),xlabel = "iteration (from smallest to largest mask ->) ",ylabel = "COV parameter number",title = "standard deviation of estimate of COV parameters 'gamma' ",size = (500,300),titlefont = font(11))

pnu2Mean=Plots.plot(collect(1:N_sim+1),MEAN_MH[end,:].-ν2_real,xlabel = "iteration (from smallest to largest mask ->)",ylabel = "deviation from mean",title = "deviation of estimated 'nu^2' to its real value",size = (400,250),legend = false)
pnu2Std=Plots.plot(collect(1:N_sim+1),STD_MH[end,:],xlabel = "iteration (from smallest to largest mask ->)",ylabel = "standard deviation",title = "standard deviation of estimated 'nu^2' ",size = (400,250),legend = false)

pHeatmapMeanALL = Plots.heatmap(collect(1:N_sim+1),collect(1:d),MEAN_measure,fill=true,color = cgrad(:lighttest),xlabel = "iteration (from smallest to largest mask ->) ",ylabel = "parameter number",title = "deviation of estimate of parameter to its real value",size = (500,300),titlefont = font(11))
pHeatmapStdALL = Plots.heatmap(collect(1:N_sim+1),collect(1:d),STD_MH,fill=true,color = cgrad(:lighttest),xlabel = "iteration (from smallest to largest mask ->) ",ylabel = "parameter number",title = "standard deviation of estimate of parameter to its real value",size = (500,300),titlefont = font(11))

##
#p1=Plots.plot(Num_Pixels_arr,[MEAN_MH[end,:] MEAN_MH[end,:]-STD_MH[end,:] MEAN_MH[end,:]+STD_MH[end,:] ν2_real*ones(N_sim+1)],xlabel = "Number of data points",ylabel = "mean",title = "Mean and real value of the coefficient \n given #data points",label = ["estimate" "" "" "real"],legendfont = font(10))


p3 = plot(p1,p2,size = (600,300))
#savefig(p3,"examplePerformanceMaskOnMH.png")

ν2pathsmatrix = Path_arr[:,end,:]

plotdensν2 = density(ν2pathsmatrix[:,[1,10,14,end]],label = ["iter. 1 (no mask)" "iter. 10"  "iter. 14"  "iter. 16 (last)"],title = "Empirical pdf for nu^2 for different mask sizes",legendfont = font(11))
plot!(ones(2),[0,12],label = "real value nu^2",color = "red")

savefig(plotdensν2,"EmpPDFMultipleMasksnu2.png")
