using OmniSci
# https://omnisci.github.io/OmniSci.jl/latest/
# 连接OmniSci/mapd 数据库

conn = connect("localhost", 9091, "mapd", "HyperInteractive", "mapd")

status = get_status(conn)  # 获取数据库连接信息
tbl = get_tables_meta(conn)   # 获取所有表的元数据

query = """SELECT origin_city AS "Origin", dest_city AS "Destination", AVG(airtime) AS "Average Airtime" 
FROM flights_2008_10k WHERE distance < 175 GROUP BY origin_city, dest_city;"""

aa = sql_execute(conn, query::String)   # 查询， 返回数据是dataframes 格式

sql_execute(conn::OmniSciConnection, query::String; first_n::Int = -1, at_most_n::Int = -1, as_df::Bool = true)


