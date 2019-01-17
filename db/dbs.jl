using SQLite, MySQL, DataFrames


function sqlite_1(file_path::AbstractString)
    db = SQLite.DB(file_path)
    q = DataFrame(SQLite.query(db, "SELECT * FROM star  limit 10"))
    print(q.star_name) 
end


function mysql_1()
    conn = MySQL.connect("39.106.167.208", "root", "000000"; db="mysql", port=3306, opts = Dict())
    q = DataFrame(MySQL.query(conn, "select User from user limit 2"))
    print(q) 
    MySQL.disconnect(conn)
end


# file_path = "db/example.db"
# sqlite_1(file_path)
mysql_1()
