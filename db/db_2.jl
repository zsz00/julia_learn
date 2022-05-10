using Octo
using Octo.Adapters.PostgreSQL
using DataFrames
using StatsPlots
using TableTransforms
using Plots, PairPlots
using Distributions

println(Repo.debug_sql())

function get_data()
    # 连接远程的postgresql数据库
    Repo.connect(
        adapter = Octo.Adapters.PostgreSQL,
        host = "192.168.3.199",
        port = "5432",
        dbname = "dataminer_test",
        user = "postgres",
        password = "123456"
    )

    # a = Repo.query(Raw("""select * from task;"""))   # query execute
    b = Repo.query(Raw("""select obj_info.obj_id,obj_info.blur,task_object.reason from face as obj_info, task_object 
    where obj_info.obj_id=task_object.obj_id and obj_info.dataset_name='af_nj_344w_2730' and task_object.obj_type=5 
    and task_object.person_id='' and task_object.task_id='b500627e-58c4-4ead-aa72-53f7d404bf48' limit 10000;"""))

    bb = DataFrame(b)  # 把结果数据(Vector{<:NamedTuple}) 格式转换为 df. 
    # println(bb)
    Pretty.set(colsize = 10)  # 显示设置
    c = Pretty.table(b)    # 把结果数据(Vector{<:NamedTuple})格式化为显示数据(string)
    println(c)

    @df bb histogram(:blur)
end

function test_1()
    N = 1_000_000
    a = [2randn(N÷2) .+ 6; randn(N÷2)]
    b = [3randn(N÷2); 2randn(N÷2)]
    c = randn(N)
    d = c .+ 0.6randn(N)
    table = (; a, b, c, d)

    table |> PCA() |> corner
end

# get_data()


#=
2021.5.15  work good
export JULIA_NUM_THREADS=4


可以多数据库连接,操作:
https://github.com/wookay/Octo.jl/issues/27  
sqc = Repo.connect(adapter=Octo.Adapters.SQLite, dbfile = ":memory:", multiple=true)
myc = Repo.connect(adapter=Octo.Adapters.MySQL, hostname="sx1", port=3306, username="root", password=pass, db="templatedb", multiple=true)
Repo.insert!(Price, (name = "Jessica", price = 70000.50); db=myc)
pgc = Repo.connect(adapter=Octo.Adapters.PostgreSQL, host="0.0.0.0", port=5432, user="template", password=pass, dbname="templatedb", multiple=true)
Repo.insert!(Price, (name = "Jessica", price = 70000.50); db=pgc)
Repo.execute("update prices set create_dt = '$(now())'  where id> 10"; db=pgc)
Repo.query("select * from prices where id >1"; db=sqc)
Repo.disconnect(db=pgc)

能多数据库联合查询吗?  应该可以



=#

