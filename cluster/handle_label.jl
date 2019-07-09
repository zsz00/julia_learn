using PyCall
np = pyimport("numpy")


function handle_labels(labels,out_dir)
    ids_1 = Set(labels)
    ids_1 = collect(ids_1)
    println("id_sum:", length(ids_1), " labels:", length(labels))   # 25w img
    new_labels = []
    for label in labels   # 太慢   Threads.@threads 
        for i in range(1, length(ids_1), step=1)
            if label == ids_1[i]
                # println(i)
                # new_labels.append(i)
                push!(new_labels, i-1)
            end
        end
    end

    println("new_labels:", maximum(new_labels, dims=1)) 
    new_labels = np.array(new_labels)

    np.savetxt(joinpath(out_dir, "labels_test_0_2.txt"), new_labels, fmt="%d")
    return new_labels
end


city = "disk_2"
out_dir = "/data/yongzhang/cluster/data_3/clean/out_$(city)_6/"
label_file = joinpath(out_dir, "labels_test_0.txt")  #    new_labels
labels = readlines(label_file)

@time handle_labels(labels,out_dir)


# export JULIA_NUM_THREADS=10
# julia handle_label.jl  # 114.612838 seconds
# np.savetxt("labels_test_0_1.txt", new_labels, fmt='%d')
