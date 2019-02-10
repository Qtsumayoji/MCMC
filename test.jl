using Distributions
using StatsBase

using PyCall
@pyimport pylab as plt
@pyimport seaborn as sns

# 確率
#相対度数？らしい
ary=[1,1,1,1,2,2,2,3,3,4]
print(proportions(ary,minimum(ary):maximum(ary)))
  # =>
  # 4-element Array{Float64,1}:
  # 0.4
  # 0.3
  # 0.2
  # 0.1

# 乱数
n = 3
rand()   # 1個の乱数（一様分布）、 0 <= rand() < 1
rand(n)  # ｎ個の乱数（一様分布）
randn()  # 1個の乱数（正規分布）、 0 <= randn() < 1
randn(n) # ｎ個の乱数（正規分布）

# 切り上げ・切り捨て
ceil(rand(),n)  # 小数第n位に切り上げ
floor(rand(),n) # 小数第n位に切り捨て

# 階乗(factorial)
factorial(n)   # n!

# 組み合わせ(combination)数
n=5; r=2;
binomial(n, r) # 組み合わせ数、nCr

# 順列(permutation)数
n=5; r=2;
binomial(n, r) * factorial(r) # 順列数、nPr

model = Uniform(-1.5, 1.0)
rand(model, 5)

n = 5
θ = zeros(n)
ϕ = zeros(n)
model = Uniform(0.0, 1.0)
θ = rand(model, n)

for i in 1:n
    p = rand(model)
    ϕ[i] = acos(1.0 - 2.0*p)
end

a = Array{Float64}[zeros(3) for i in 1:5]
b = zeros(3)
println(dot(a[2],b))
println(a[3])

sqz = rand(10,10)
sns.heatmap(sqz, cmap = "Blues")
plt.show()