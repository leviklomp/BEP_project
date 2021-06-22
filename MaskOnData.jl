using Random
using Distributions
include("SimulateRF.jl")
include("CreateCovarianceMatrices.jl")
include("AuxiliaryFunctions.jl")

function CreateIsolatedMaskM1D(xas,p = 0.5)
    """Creates mask for a 1D field (line) where eaxh pixel has a Ber(p) distribution.
    If the value is 1 the pixel is not eliminated. If 0 the pixel is eliminated. Also returns a coordinate set
    of which coordinates remain in the data."""
    N = length(xas)
    m = rand(Bernoulli(p),N)
    num_ones = sum(m)
    M = zeros(num_ones,N)
    CoordSet = []
    row_num = 1
    for i = 1:N
        if m[i] == 1
            vec = zeros(N)
            vec[i] = 1
            M[row_num,:] = vec
            row_num += 1
            CoordSet = append!(CoordSet,xas[i])
        end
    end
    return M,CoordSet
end


function CreateIsolatedMaskM2D(xas,yas,p = 0.5)
    """Creates mask for a 2D field (grid with xas and yas) where eaxh pixel has a Ber(p) distribution.
    If the value is 1 the pixel is not eliminated. If 0 the pixel is eliminated. Also returns a coordinate set
    of which coordinates remain in the data."""
    Nx = length(xas)
    Ny = length(yas)
    N = Nx*Ny
    m = rand(Bernoulli(p),N)
    num_ones = sum(m)
    M = zeros(num_ones,N)
    CoordSet = []
    row_num = 1
    for row = 1:Ny
        for col = 1:Nx
            i = Nx*(row-1)+col
            if m[i] == 1
                vec = zeros(N)
                vec[i] = 1
                M[row_num,:] = vec
                row_num += 1
                CoordSet = append!(CoordSet,[(xas[col],yas[row])])
            end
        end
    end
    return M,CoordSet
end

function CreateRectangularMask2D(xas,yas,LL,UR)
    """Creates rectangular mask for a 2D field (grid with xas and yas) where the bottom left corner is a tuple LL
    and the upper right corner is a tuple UR. All pixels inside this rectangle are elimnated. Also returns a coordinate set
    of which coordinates remain in the data."""
    Nx = length(xas)
    Ny = length(yas)
    N = Nx*Ny
    M = zeros(N,N)
    CoordSet = []
    row_num = 1
    for row = 1:Ny
        for col = 1:Nx
            i = Nx*(row-1)+col
            x = xas[col]
            y = yas[row]
            if !(LL[1]<=x<=UR[1] && LL[2]<=y<=UR[2]) #if not inside the rectangle
                vec = zeros(N)
                vec[i] = 1
                M[row_num,:] = vec
                row_num += 1
                CoordSet = append!(CoordSet,[(xas[col],yas[row])])
            end
        end
    end
    m = length(CoordSet)
    M = M[1:m,:]
    return M,CoordSet
end


##
#=
L = 1
M = L
dx = 0.1
dy = dx
xas = collect(-L:dx:L)
yas = collect(-M:dy:M)

LL = (-.8,.2)
UR = (.6,.5)
(Mask,CoordSet) = CreateRectangularMask2D(xas,yas,LL,UR)

N = length(CoordSet)
function ρ_X(t1,t2) #Correlation function (in interval [-1,1])

    A,B,C = 1,1,0.1
    return A*cos(B*norm(t1.-t2))*exp(-C*norm(t1.-t2))
end

R = SimGaussian2D(L,M,dx,dy,ρ_X)
Rvec = MatrixToVector(R)
Plots.contour(xas,yas,R,fill = true)

Rnew = Mask*Rvec

newxas = []
newyas = []
for i = 1:N
    newxas = append!(newxas,CoordSet[i][1])
    newyas = append!(newyas,CoordSet[i][2])
end
Maskxas = collect(LL[1]:dx:UR[1])
Maskyas = collect(LL[2]:dy:UR[2])
Maskval = ones(length(Maskyas),length(Maskxas))
Plots.heatmap(xas,yas,R,fill = true)
Plots.heatmap!(Maskxas,Maskyas,Maskval,fill = true)

Σ = CreateKernelMatrixCoordset2D(ρ_X,CoordSet)

Plots.heatmap(collect(1:N),collect(1:N),Σ,fill = true)
=#
