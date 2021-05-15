using Octo
using Octo.Adapters.PostgreSQL
using DataFrames
using StatsPlots



println(Repo.debug_sql())

# 连接远程的postgresql数据库
Repo.connect(
           adapter = Octo.Adapters.PostgreSQL,
           host = "192.168.3.199",
           port = "5432",
           dbname = "dataminer_test",
           user = "postgres",
           password = "123456"
       )

a = Repo.query(Raw("""select * from task;"""))   # query execute
b = Repo.query(Raw("""select obj_info.obj_id,obj_info.blur,task_object.reason from face as obj_info, task_object 
where obj_info.obj_id=task_object.obj_id and obj_info.dataset_name='af_nj_344w_2730' and task_object.obj_type=5 
and task_object.person_id='' and task_object.task_id='b500627e-58c4-4ead-aa72-53f7d404bf48' limit 1000000;"""))
bb = DataFrame(b)
# println(bb)

@df bb histogram(:blur)


#=
2021.5.15  work good

=#

