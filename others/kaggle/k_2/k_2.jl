using MLJ, Statistics
using CSV, DataFrames, PrettyTables, DataFramesMeta
using StableRNGs
using StatsModels
using Images
using Flux, MLJFlux

# RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree
# # DTC = @load DecisionTreeClassifier pkg=DecisionTree
# # AdaBoostStumpClassifier    DecisionTreeClassifier  RandomForestClassifier
# XGBC = @load XGBoostClassifier pkg=XGBoost   # LGBMRegressor
# LGBMC = @load LGBMClassifier pkg=LightGBM
# NeuralNetworkClassifier = @load NeuralNetworkClassifier


mutable struct MyBuilder2 <: MLJFlux.Builder
    n1 :: Int
    n2 :: Int
    n3 :: Int
    n4 :: Int
end

function MLJFlux.build(nn::MyBuilder2, rng, n_in, n_out)
    #init = Flux.glorot_uniform(rng)
    init = Flux.kaiming_uniform(rng)
    σ =NNlib.leakyrelu  # relu leakyrelu
    return Flux.Chain(Dense(n_in, nn.n1,  init=init),
                            BatchNorm(nn.n1, relu),
                            Dense(nn.n1, nn.n2, init=init),
                            BatchNorm(nn.n2, relu),
                            Dense(nn.n2, nn.n3, init=init),
                            BatchNorm(nn.n3, relu),
                            Dense(nn.n3, nn.n4, init=init),
                            BatchNorm(nn.n4, relu),
                            # Dropout(0.5),
                            Dense(nn.n4, n_out, init=init))
end

mutable struct MyBuilder3 <: MLJFlux.Builder
    k::Int
    n1::Int
    n2::Int
    n3::Int
    n4::Int
end

function MLJFlux.build(nn::MyBuilder3, rng, n_in, n_out)
    #init = Flux.glorot_uniform(rng)
    init = Flux.kaiming_uniform(rng)
    σ =NNlib.leakyrelu  # relu leakyrelu
    k, n1, n2, n3, n4 = nn.k, nn.n1, nn.n2, nn.n3, nn.n4 
    p = div(k - 1, 2)
    return Flux.Chain(
                    Conv((k, k), 3 => n2, pad=(p, p), init=init),
                    MaxPool((2, 2)),
                    Conv((k, k), n2 => n3, pad=(p, p), init=init),
                    BatchNorm(nn.n1, relu),
                    Conv((k, k), n3 => n4, pad=(p, p), init=init),
                    BatchNorm(nn.n4, relu),
                    Dense(nn.n4, n_out, init=init))
end

function flatten(x::AbstractArray)
	return reshape(x, :, size(x)[end])
end

import MLJFlux
mutable struct MyConvBuilder
	filter_size::Int
	channels1::Int
	channels2::Int
	channels3::Int
end

function MLJFlux.build(b::MyConvBuilder, rng, n_in, n_out, n_channels)
	k, c1, c2, c3 = b.filter_size, b.channels1, b.channels2, b.channels3
	mod(k, 2) == 1 || error("`filter_size` must be odd. ")

	# padding to preserve image size on convolution:
	p = div(k - 1, 2)

	front = Chain(
			   Conv((k, k), n_channels => c1, pad=(p, p), relu),
			   MaxPool((2, 2)),
			   Conv((k, k), c1 => c2, pad=(p, p), relu),
			   MaxPool((2, 2)),
			   Conv((k, k), c2 => c3, pad=(p, p), relu),
			   MaxPool((2 ,2)),
			   flatten)
	d = Flux.outputsize(front, (n_in..., n_channels, 1)) |> first
	return Chain(front, Dense(d, n_out))
end

# typeData could be either "train" or "test.
# labelsInfo should contain the IDs of each image to be read
# The images in the trainResized and testResized data files
# are 20x20 pixels, so imageSize is set to 400.
# path should be set to the location of the data files.
function read_data(typeData, labelsInfo, imageSize, path)
    x = zeros(size(labelsInfo, 1), imageSize...)
    for (index, idImage) in enumerate(labelsInfo.ID)     # we want to index it with a symbol instead of a string i.e. lablesInfoTrain[:ID]   
        nameFile = "$(path)/$(typeData)Resized/$(idImage).Bmp"   # Read image file 
        img = Images.load(nameFile)        # The replacement for imread() is load(). Depending on your platform, load() will use ImageMagick.jl or QuartzImageIO.jl (on a Mac) behind the scenes.
        temp=Gray.(img)  # reshape(img, 1, imageSize...)  # Gray.(img)            # float32sc was deprecated so we have to convert the images dirctly to gray scale which is made easy in Julia
        temp = Float64.(temp)
        # println("$(size(img)), $(size(temp)), $(typeof(img))")
        #Transform image matrix to a vector and store it in data matrix 
        x[index, :, :] = reshape(temp, 1, imageSize...)
    end 
    return x
end


function process_1()
    imageSize = 3, 20, 20  # 20 * 20 # pixels
    path = "/home/zhangyong/codes/julia_learn/others/kaggle/k_2/street_view_data"  #add your path here

    # Read information about training data , IDs.
    labelsInfoTrain = CSV.read("$(path)/trainLabels.csv", DataFrame)  # read_table has been deprecated 
    xTrain = read_data("train", labelsInfoTrain, imageSize, path)
    
    # Read information about test data ( IDs ).
    labelsInfoTest = CSV.read("$(path)/sampleSubmission.csv", DataFrame)
    xTest = read_data("test", labelsInfoTest, imageSize, path)

    # Get only first character of string (convert from string to character).
    # Apply the function to each element of the column "Class"
    yTrain = map(x -> x[1], labelsInfoTrain[:, "Class"])

    xTrain = MLJ.table(xTrain);   # 需要table类型的输入
    xTest = MLJ.table(xTest);
    rng = StableRNG(566)
    yTrain = categorical(yTrain)  # Convert from character to class integer
    println("$(yTrain[1:10])")
    # model = OneHotEncoder() |> DTC()
    # model = XGBC()  # XGBC()  LGBMC()  DTC() RandomForestRegressor
    
    # builder=MyBuilder2(32, 64, 128, 64)
    builder=MyBuilder3(3, 32, 64, 128, 64)
    MLJFlux.build(builder, rng, 3, size(yTrain, 1))
    model = NeuralNetworkClassifier(builder=builder,loss=Flux.Losses.crossentropy) 
    model.batch_size=1000
    model.epochs=10
    # model.acceleration=CUDALibs()
    # iterated_model = IteratedModel(model=model)

    mach = machine(model, xTrain, yTrain)
    fit!(mach, verbosity=2)
    fp = fitted_params(mach)
    # @show fp.fitresult[1]
    # @show fp.fitresult[1].importance()

    # println("$(length(xTrain)), $(size(yTrain))")
    pred_dcrm = predict(mach, xTest)
    pred_dcrm_cls = predict_mode(mach, xTest)
    @show pred_dcrm_cls[1:4]
    # @show pred_dcrm[1:4]
    # out_df = DataFrame(PassengerId=892:1309, Survived=convert(Array{Int64},pred_dcrm_cls))
    labelsInfoTest[!, :Class] = pred_dcrm_cls
    # out_path = "/home/zhangyong/codes/julia_learn/others/kaggle/k_1/out_1.csv"
    CSV.write("$(path)/juliaSubmission_4.csv", labelsInfoTest, writeheader=true)
    # aa = mean(LogLoss(tol=1e-4)(pred_dcrm, yTrain))  # log_loss   0.16995662487369775   0.24579491188658323
    # bb = misclassification_rate(pred_dcrm, yTrain)  # 误分类率     1.0  # 结果不对
    # println("aa:$(aa), bb:$(bb)")   
end


function process_2()
    imageSize = 20,20  # 20 * 20 # pixels
    path = "/home/zhangyong/codes/julia_learn/others/kaggle/k_2/street_view_data"  #add your path here

    labelsInfoTrain = CSV.read("$(path)/trainLabels.csv", DataFrame)  # read_table has been deprecated 
    train_x = read_data("train", labelsInfoTrain, imageSize, path)
    train_y = map(x -> x[1], labelsInfoTrain[:, "Class"])

    labelsInfoTest = CSV.read("$(path)/sampleSubmission.csv", DataFrame)
    xTest = read_data("test", labelsInfoTest, imageSize, path)
    
    @show size(train_x)  # (6283, 20, 20)
    @show size(train_y)
    images = train_x    # 28, 28, 60000
    labels = train_y
    # images = MLJ.table(images);   # 需要table类型的输入
    # images = coerce(images, GrayImage)
    labels = coerce(labels, Multiclass)
    
    ImageClassifier = @load ImageClassifier

    clf = ImageClassifier(builder=MyConvBuilder(3, 16, 32, 32),
					  epochs=100,
                      optimiser=ADAM(0.001),
					  loss=Flux.crossentropy,
                      batch_size=128,
                      acceleration=CUDALibs(),)

    mach = machine(clf, images, labels)

    @time eval = evaluate!(
        mach;
        resampling=Holdout(fraction_train=0.7, shuffle=true, rng=123),
        operation=predict_mode,
        measure=[accuracy, #=cross_entropy, =#misclassification_rate],
        verbosity = 3,
    )
    @show eval   # 0.986, 0.0141
    
end

process_2()


#=
kagggle google街景字符识别   2022.3.16
参考:
https://www.kaggle.com/c/street-view-getting-started-with-julia/data

export JULIA_NUM_THREADS=40
ENV["CUDA_VISIBLE_DEVICES"]=5 
julia others/kaggle/k_2/k_2.jl


问题:
1. dataframe还是不够熟练, 高层api不够友好. 
2. MLJ文档不够好, 层次结构不清晰. 
3. 很多三方包不够成熟. 集成不够好, 文档不够. 
4. 怎么获取特征重要程度,特征相关性? mlj现在支持不行
5. 当x是高维数据时,怎么做table类型的输入? xTrain = Tables.table(xTrain);  xTrain是3dim的这么办?
6. Matrix{RGB{N0f8}} 怎么转为Matrix{Float64}? Float64.(data)
7. 加dataloader

找一个处理自定义图片数据的示例



=#
