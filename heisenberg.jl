using LinearAlgebra
using Distributions
using PyCall
@pyimport pylab as plt
@pyimport seaborn as sns

include("./model.jl")

const J = -1.0
const β = 1.0

const Nx = 10
const Ny = 10
const Ns = Nx*Ny
const no_link = 0

total_mcs = 4000
Nd = 10
β_start = 0.1
β_end = 1.0
Nb = 1

check_mcmcs = false

#model = Uniform(0.0, 1.0)
function get_rand_angle()
    θ = 2.0*pi*rand()
    ϕ = acos(1.0 - 2.0*rand())
    return θ, ϕ
end

function set_initial_configuration(spin_vec)
    for i in 1:Ns
        θ, ϕ = get_rand_angle()
        calc_spin_vec(spin_vec[i], θ, ϕ)
    end
end

function calc_spin_vec(spin_vec::Array{Float64}, θ::Float64, ϕ::Float64)
    sin_θ = sin(θ)
    spin_vec[1] = sin_θ*cos(ϕ)
    spin_vec[2] = sin_θ*sin(ϕ)
    spin_vec[3] = cos(θ)
    #println("in calc_spin_vec: spin_vec=",spin_vec)
end

function calc_local_energy(i_spin_vec::Array{Float64}, links , spin_vec)
    E = 0.0
    for j in 1:length(links)
        j_spin = links[j]
        E += -J*dot(i_spin_vec, spin_vec[j_spin])
        #println(i," : ",i_spin_vec," : ",spin_vec[i,:])
    end

    return E
end

function calc_total_energy(link_mat, spin_vec)
    E = 0.0
    for i in 1:Ns
        for j in 1:i - 1
            if link_mat[i, j] != no_link
                E += -J*dot(spin_vec[i], spin_vec[j])
            end
        end
    end

    return E
end

function Metropolis(β, spin_vec, link_mat, link_list)
    energy = zeros(total_mcs)
    vec_e = [float(i) for i in 1:total_mcs]

    #set_initial_configuration(spin_vec)
    #print(spin_vec)

    set_initial_configuration(spin_vec)

    @time for m in 1:total_mcs
        for i in 1:Ns
            tmp_spin_vec = zeros(3)
            θ, ϕ = get_rand_angle()
            calc_spin_vec(tmp_spin_vec, θ, ϕ)

            E1 = calc_local_energy(spin_vec[i], link_list[i], spin_vec)
            E2 = calc_local_energy(tmp_spin_vec, link_list[i], spin_vec)
            ΔE =  E2 - E1

            if rand() < exp(-β*ΔE)
                spin_vec[i] = tmp_spin_vec
            end
        end
        
        if check_mcmcs
            energy[m] = calc_total_energy(link_mat, spin_vec)/Ns
        end
    end

    if check_mcmcs
        plt.plot(vec_e, energy)
        plt.show()
    end
end

function calc_magnetization(spin_vec)
    s_tot = zeros(3)
    for i in 1:Ns
        s_tot += spin_vec[i]
    end
    return norm(s_tot)/Ns
end

function calc_spin_spin_correlation(spin, ssc)
    for i in 1:Ns
        for j in 1:Ns
            ssc[i, j, 1] = spin[i][1]*spin[j][1]/4.0
            ssc[i, j, 2] = spin[i][2]*spin[j][2]/4.0
            ssc[i, j, 3] = spin[i][3]*spin[j][3]/4.0
        end
    end
end

function calc_Sq(sqz, ssc, unit_vec, pos, link_mat, spin)
    V = dot(unit_vec[1], cross(unit_vec[2], unit_vec[3]))
    g1 = 2*pi*cross(unit_vec[2], unit_vec[3])/V
    g2 = 2*pi*cross(unit_vec[3], unit_vec[1])/V
    g3 = 2*pi*cross(unit_vec[1], unit_vec[2])/V
    Lx = norm(unit_vec[1])*Nx
    Ly = norm(unit_vec[2])*Ny
    
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
    @time for m in -Nb*Nx:Nb*Nx
        for n in -Nb*Ny:Nb*Ny
            tq = m/Lx*g1 + n/Ly*g2
            tqx = tq[1]
            tqy = tq[2]

            tmp_sqx = 0.0
            tmp_sqy = 0.0
            tmp_sqz = 0.0
            for i in 1:Ns
                for j in 1:i - 1
                    Δrx = pos[i][1] - pos[j][1]
                    Δry = pos[i][2] - pos[j][2]
                    cos_qr = cos.(tqx*Δrx + tqy*Δry)
                    tmp_sqx += 2.0*ssc[i, j, 1]*cos_qr
                    tmp_sqy += 2.0*ssc[i, j, 2]*cos_qr
                    tmp_sqz += 2.0*ssc[i, j, 3]*cos_qr
                end
                tmp_sqx += ssc[i, i, 1]
                tmp_sqy += ssc[i, i, 2]
                tmp_sqz += ssc[i, i, 3]
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

    #Model.show_links(link_mat, Ns)

    β = range(β_start, stop=β_end, length=Nd)
    temp = 1.0/β
    mag = zeros(Nd)
    total_E = zeros(Nd)
    for i in 1:Nd
        spin = [zeros(3) for i in 1:Ns]
        ssc = zeros(Ns, Ns, 3)
        sq = zeros(2*Nb*Nx + 1, 2*Nb*Ny + 1)

        Metropolis(β[i], spin, link_mat, link_list)

        calc_spin_spin_correlation(spin, ssc)
        calc_Sq(sq, ssc, unit_vec, pos, link_mat, spin)
        mag[i] = calc_magnetization(spin)
        #total_E[i] = calc_total_energy(link_mat, spin)/Ns
    end
    plt.plot(temp, mag)
    #plt.plot(temp, total_E)
    plt.show()
end

main()