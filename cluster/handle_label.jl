using PyCall
np = pyimport("numpy")


function handle_labels(labels, new_labels_file)
    ids_1 = Set(labels)
    ids_1 = sort!(collect(ids_1))
    println("=======", join("_", [ids_1[1],ids_1[2], ids_1[3]]))
    println("id_sum:", length(ids_1), " labels:", length(labels), " ", typeof(labels))   # 25w img
    
    new_labels = zeros(Int,length(labels))
    # for (i, label) in enumerate(labels)    # Threads.@threads 
    for i in range(1, length(labels), step=1)
        label = labels[i]
        if label == -1
            new_labels[i] = -1
            continue
        end
        for id in range(2, length(ids_1), step=1)
            if label == ids_1[id]
                # println(i)
                # push!(new_labels, id-1)
                new_labels[i] = id-2
                
            end
        end
    end

    println("new_labels:", maximum(new_labels, dims=1)) 
    new_labels = np.array(new_labels)

    np.savetxt(new_labels_file, new_labels, fmt="%d")
    return new_labels
end


city = "disk_2"
out_dir = "/data/yongzhang/cluster/data_3/clean/out_$(city)_11/"
label_file = joinpath(out_dir, "labels_test_1_1.txt")  #    new_labels
new_labels_file = joinpath(out_dir, "labels_test_1_1_0.txt")
labels = readlines(label_file)
labels = [parse(Int,a) for a in labels]
# labels = hcat(labels)

@time handle_labels(labels, new_labels_file)


# export JULIA_NUM_THREADS=10
# julia handle_label.jl  # 114.612838 seconds  2min
# 8.309718 seconds
#  
