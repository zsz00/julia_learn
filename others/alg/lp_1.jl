
# LP, Linear Programming, 线性规划.  对标lingo,matlab软件 
using JuMP
using Clp


#=
z = min(x_11*12 + x_12*24 + x_13*8 + x_21*30 + x_22*12 + x_23*24)

x_11 + x_12 + x_13 <= 4
x_21 + x_22 + x_23 <= 8
x_11 + x_21 >= 2
x_12 + x_22 >= 4
x_13 + x_23 >= 5
x_11,x_12,x_13,x_21,x_22,x_23>=0
=#

# 利用JuMP求解原问题
function CarpenterPrimal(c,A,b)
    # 定义Model对象, OutputFlag = 0指不输出log
    Primal = Model(solver = GurobiSolver(OutputFlag = 0))
    # 定义变量，宏的调用也是Julia&JuMP高效编译/元编程的重要技巧
    @variable(Primal, x[1:2]>=0)
    # 定义不等式约束
    constr = @constraint(Primal, A*x.<=b)
    # 定义目标函数
    @objective(Primal, Max, dot(c,x))
    # 求解
    solve(Primal)
    # 返回最优目标函数值，最优解（原问题），最优解（对偶问题）
    return getobjectivevalue(Primal), getvalue(x), getdual(constr)
end


# 利用JuMP求解线性规划(LP)问题
function lp_1()
    # book\数据建模\0课件集合\3数学规划\数学规划建模.ppx  例1 运输问题
    model = Model(Clp.Optimizer)   # Clp GLPK
    @variable(model, x[1:2, 1:3] >= 0)  # 变量
    @objective(model, Min, x[1,1]*12 + x[1,2]*24 + x[1,3]*8 + x[2,1]*30 + x[2,2]*12 + x[2,3]*24)  # 目标函数
    @constraint(model, c1, x[1,1] + x[1,2] + x[1,3] <= 4)  # 约束条件
    @constraint(model, c2, x[2,1] + x[2,2] + x[2,3] <= 8)
    @constraint(model, c3, x[1,1] + x[2,1]>= 2)
    @constraint(model, c4, x[1,2] + x[2,2]>= 4)
    @constraint(model, c5, x[1,3] + x[2,3]>= 5)

    print(model)
    optimize!(model)
    @show termination_status(model) # 终止状态
    @show objective_value(model)   # 最终优化的结果
    @show value.(x)       # 模型的参数值, 解决方案
    @show shadow_price(c1)  # 约束c1的影子价格
    @show shadow_price(c2)
end


lp_1()

