using LinearAlgebra
using Distributions
using PyCall
@pyimport pylab as plt
@pyimport seaborn as sns

include("./model.jl")

const J = -1.0

const Nx = 10
const Ny = 10
const Ns = Nx*Ny
const no_link = 0

total_mcs = 2000

Nd = 10
β_start = 0.5
β_end = 1.0

#Sqでブリルアンゾーン何個見るか
Nb = 1

check_mcmcs = false

function metropolis(β, spin, link_mat, link_list)
    energy = zeros(total_mcs)
    vec_e = [float(i) for i in 1:total_mcs]

    #spinの初期化
    for i in 1: Ns
        spin[i] = get_random_spin()
    end

    @time for m in 1:total_mcs
        #random_vec = rand(1:Ns, total_mcs)
        for i in 1:Ns
            i_spin = i#random_vec[i]
            #ΔE/J = (-σⱼ)Σσᵢ - (σⱼΣσᵢ) = 2(-σⱼ*Σσᵢ) 
            #反転したスピンについてcalcすればΔEを与える
            ΔE =  2*calc_local_energy(-spin[i_spin], link_list[i_spin], spin)
            #ΔE < 0 では必ずexp(-βΔE) > 1となり条件は満たされる
            if rand() < exp(-β*ΔE)
                spin[i_spin] *= -1
            end
        end
        
        if check_mcmcs
            energy[m] = calc_total_energy(link_mat, spin)/Ns
        end
    end
    
    for i in 1:Ny
        for j in 1:Nx
            if spin[(i-1)*Ny + j] == 1
                print("+ ")
            else
                print("o ")
            end
        end
        println()
    end
    println()

    if check_mcmcs
        plt.plot(vec_e, energy)
        plt.show()
    end
end

function get_random_spin()
    return rand(-1:2:1)
end

function calc_local_energy(spin_i, links, spin)
    E = 0.0
    for j in 1:length(links)
        E += -J*spin_i*spin[links[j]]
    end

    return E
end

function calc_total_energy(link_mat, spin)
    E = 0.0
    for j in 1:Ns
        for i in 1:j-1
            if link_mat[i, j] == 1
                E += -J*spin[i]*spin[j]
            end
        end
    end
    return E
end

function test_calc_local_energy()
    #Sz = 1のスピンが4っつのising相互作用している
    #E = -(1+1-1-1)=0
    spin = [1,1,-1,-1,-1]
    links = [2,3,4,5]
    E = calc_local_energy(spin[1], links, spin)
    println(E)
end

function calc_magnetization(spin)
    mag = 0
    for i in 1:Ns
        mag += spin[i]
    end
    return abs(mag/Ns)
end

function calc_spin_spin_correlation(spin, sscz)
    for i in 1:Ns
        for j in 1:Ns
            sscz[i, j] = spin[i]*spin[j]/4.0
        end
    end
end

function calc_Sq(sqz, sscz, unit_vec, pos, link_mat, spin)
    V = dot(unit_vec[1], cross(unit_vec[2], unit_vec[3]))
    g1 = 2*pi*cross(unit_vec[2], unit_vec[3])/V
    g2 = 2*pi*cross(unit_vec[3], unit_vec[1])/V
    g3 = 2*pi*cross(unit_vec[1], unit_vec[2])/V
    Lx = sqrt(dot(unit_vec[1], unit_vec[1]))*Nx
    Ly = sqrt(dot(unit_vec[2], unit_vec[2]))*Ny
    
    #qx = []
    #qy = []
    #@time for n1 in -Nb*Nx:Nb*Nx
    #    for n2 in -Nb*Ny:Nb*Ny
    #        tqx = n1/Nx*g1[1] + n2/Ny*g2[1]
    #        tqy = n1/Nx*g1[2] + n2/Ny*g2[2]
    #        push!(qx, tqx)
    #        push!(qy, tqy)
    #    end
    #end

    println("calculate SSF :")
    for n in -Nb*Nx:Nb*Nx
        for m in -Nb*Ny:Nb*Ny
            tq = m/Lx*g1 + n/Ly*g2
            tqx = tq[1]
            tqy = tq[2]

            tmp_sqz = 0.0
            for j in 1:Ns
                for i in 1:j - 1
                    Δrx = pos[i][1] - pos[j][1]
                    Δry = pos[i][2] - pos[j][2]
                    tmp_sqz += 2.0*sscz[i, j]*cos.(tqx*Δrx + tqy*Δry)
                end
                tmp_sqz += sscz[j, j]
            end
            sqz[m + Nb*Nx + 1, n + Nb*Ny + 1] = tmp_sqz/Ns
        end
    end
    println()

    sns.heatmap(sqz, cmap = "Blues")
    plt.show()
end

function main()
    link_mat, link_list, pos = Model.square_lattice(Nx, Ny)
    unit_vec = Model.get_square_lattice_unit_vec()
    #Model.show_links(link_mat)

    β = range(β_start, stop=β_end, length=Nd)
    temp = 1.0/β
    mag = zeros(Nd)
    total_E = zeros(Nd)

    for i in 1:Nd
        spin = Array{Int64}(undef, Ns) 
        sscz = zeros(Ns, Ns)
        sqz = zeros(2*Nb*Nx + 1, 2*Nb*Ny + 1)

        @time metropolis(β[i], spin, link_mat, link_list)

        @time calc_spin_spin_correlation(spin, sscz)
        @time calc_Sq(sqz, sscz, unit_vec, pos, link_mat, spin)
        mag[i] = calc_magnetization(spin)
        #total_E[i] = calc_total_energy(link_mat, spin)/Ns
    end

    plt.plot(temp, mag)
    #plt.plot(temp, total_E)
    plt.show()
end

main()