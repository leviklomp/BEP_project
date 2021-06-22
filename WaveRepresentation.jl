using Random
using Statistics
using Distributions
using StatsBase
using Plots



function Fourier_R1(a_array,t,L = 1)
    """Calculates a Fourier Series with coefficients a_l in a_array at a point 'x'
     where the first element is coefficient a_{-l_max} and the last element is
     coefficient a_{l_max} (2l_max+1 elements)"""
     x = t
    n = length(a_array)
    arr  = collect(1:n)
    e_power = exp.(-im*pi*arr*x/L)
    return real(sum(a_array.*e_power))
end

function Fourier_R2(A_matrix,t,L = 1,M = 1)
    """Calculates a Fourier Series with coefficients a_lm in A_matrix at a point 'x'
     where the first element is coefficient a_{-l_max,-l_max} and the last element is
     coefficient a_{l_max,l_max} ( (2l_max+1)^2 elements)"""
     x = t[1]
     y = t[2]

    n = size(A_matrix)[1]
    arr = collect(1:n)
    e_powerx = exp.(-im*pi*arr*x/L)
    e_powery = exp.(-im*pi*arr*y/M)
    e_power = e_powery*transpose(e_powerx)
    return real(sum(A_matrix.*e_power))
end

function Fourier_RF_2D(A,xas,yas,L = 1,M = 1)
    #l_max = Int((length(A)-1)/2)
    R(t) = Fourier_R2(A,t,L,M)
    Nx = length(xas)
    Ny = length(yas)
    RF = zeros(Ny,Nx)
    for i = 1:Nx
        for j = 1:Ny
            t = (xas[i],yas[j])
            RF[j,i] = R(t)
        end
    end
    return RF
end

#=
#############################################################################
A = rand(Normal(0,0.1),21,21)
xas = collect(0:0.01:1)
yas = xas

Plots.contour(xas,yas,Fourier_RF_2D(A,xas,yas),fill = true)
#############################################################################
=#
