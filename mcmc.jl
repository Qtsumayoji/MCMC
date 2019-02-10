using Distributions

#強磁性ハイゼンベルグ模型を想定
const J = 1.0
const beta = 1.0

const width = 3
const height = 3
const n_sites = 2*width*height

const open_boundary = 1
const periodic_boundary = 2
boundary_condition = periodic_boundary

function  metroporis(links, px, py)
    model = Uniform(0.0, 1.0)

    #スピンベクトルの極座標表示
    #もちろんスピンの長さは変わらないのでspin_rは使わない
    spin_r = 1.0
    spin_vec = Array{Float64}[zeros(3) for i in 1:n_sites]

    #set_initial_configuration(spin_vec)
    #print(spin_vec)

    n_mcmc_steps = 10000
    t_spin_vec = zeros(3)

    set_initial_configuration(spin_vec)
    for i in 1:n_mcmc_steps
        for j in 1:n_sites
            E1 = calc_energy(j, spin_vec[j], spin_vec, links)
            θ, ϕ = get_rand_polar_coordinates()  
            calc_spin_vec(t_spin_vec, θ, ϕ)
            #println("t_spin_vec = ",t_spin_vec)
            E2 = calc_energy(j, t_spin_vec, spin_vec, links)
            dE = E2 - E1
            #println("E1,E2 = ",E1,E2)
            if rand(model) < exp(-beta*dE)
                spin_vec[j] = t_spin_vec
            end
        end
        println(calc_total_energy(spin_vec, links))
    end

end

function set_initial_configuration(spin_vec)
    for i in 1:n_sites
        θ, ϕ = get_rand_polar_coordinates()
        calc_spin_vec(spin_vec[i], θ, ϕ)
    end
end

model = Uniform(0.0, 1.0)
function get_rand_polar_coordinates()
    θ = rand(model)
    ϕ = acos(1.0 - 2.0*rand(model))
    return θ, ϕ
end

function calc_spin_vec(spin_vec, θ::Float64, ϕ::Float64)
    sin_θ = sin(θ)
    cos_ϕ = cos(ϕ)
    spin_vec[1] = sin_θ*cos_ϕ
    spin_vec[2] = sin_θ*sin(ϕ)
    spin_vec[3] = cos_ϕ
    #println("in calc_spin_vec: spin_vec=",spin_vec)
end

function calc_energy(i_spin::Int, i_spin_vec::Array{Float64}, spin_vec, link_list)
    E = 0.0
    for i in 1:length(link_list[i_spin])
        j_spin = link_list[i_spin][i]
        E += -J*dot(i_spin_vec, spin_vec[j_spin])
        #println(i," : ",i_spin_vec," : ",spin_vec[i,:])
    end
    return E
end

function calc_total_energy(spin_vec, link_list)
    E = 0.0
    for i in 1:n_sites
        for j in 1:length(link_list[i])
            k = link_list[i][j]
            E += -J*dot(spin_vec[i], spin_vec[k])
        end
    end

    return E
end

function main()
    #links[i,j] = 1だとサイト間に相互作用がある
    links = Array{Int32}(n_sites, n_sites)
    for i in 1:n_sites
        for j in 1:n_sites
            links[i, j] = -1
        end
    end


    px = Float64[]
    py = Float64[]

    make_honeycomb(links, px, py)

    link_list = [[] for i in 1:n_sites]
    for i in 1:n_sites
        for j in 1:n_sites
            if links[i, j] != -1
                push!(link_list[i], j)
            end
        end
    end
    #println(link_list[2][2])

    #show_model(links, px, py)
    metroporis(link_list, px, py)
    #diagonalization(links, px, py)
end

main()