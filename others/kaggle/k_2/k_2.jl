using MLJ, Statistics
using CSV, DataFrames, PrettyTables, DataFramesMeta
using StableRNGs
using StatsModels
using Images
using Flux, MLJFlux, Metalhead


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

mutable struct MyConvBuilder
	filter_size::Int
	channels1::Int
	channels2::Int
	channels3::Int
end

#=
function MLJFlux.build(b::MyConvBuilder, rng, n_in, n_out, n_channels)
    init = Flux.kaiming_uniform(rng)
    σ =NNlib.relu  # relu leakyrelu
    k, n1, n2, n3 = b.filter_size, b.channels1, b.channels2, b.channels3
    p = div(k - 1, 2)
    front = Flux.Chain(
                    Conv((k, k), n_channels => n1, pad=SamePad(), init=init),
                    Conv((k, k), n1 => n2, σ, pad=SamePad(), init=init),
                    Conv((k, k), n2 => n2, σ, pad=SamePad(), stride=1, init=init),
                    Dropout(0.25),
                    BatchNorm(n2),
                    Conv((k, k), n2 => n3, σ, pad=SamePad(), stride=1, init=init),
                    Conv((k, k), n3 => n3, σ, pad=SamePad(), stride=2, init=init),
                    BatchNorm(n3),
                    Conv((k, k), n3 => n3, σ, pad=SamePad(), stride=1, init=init),
                    Conv((k, k), n3 => n3, σ, pad=SamePad(), stride=1, init=init),
                    BatchNorm(n3),
                    flatten)
    d = Flux.outputsize(front, (n_in..., n_channels, 1)) |> first
	return Flux.Chain(front, Dense(d, 512, σ), Dropout(0.25), Dense(512, n_out))
end
=#

function MLJFlux.build(b::MyConvBuilder, rng, n_in, n_out, n_channels)
    model = ResNet18(pretrain=false, nclasses=n_out)
    return model
end

# typeData could be either "train" or "test.
# labelsInfo should contain the IDs of each image to be read
# The images in the trainResized and testResized data files
# are 20x20 pixels, so imageSize is set to 400.
# path should be set to the location of the data files.
function read_data(typeData, labelsInfo, imageSize, path)
    # x = zeros(size(labelsInfo, 1), imageSize...)   # RGB{N0f8},
    x = Array{Matrix{RGB{N0f8}},1}()
    for (index, idImage) in enumerate(labelsInfo.ID) 
        nameFile = "$(path)/$(typeData)Resized/$(idImage).Bmp"   
        img = Images.load(nameFile)  # Read image file 
        # println("$(size(img)), $(typeof(img))")   # (20, 20), Matrix{RGB{N0f8}}
        push!(x, img)
        # temp=Gray.(img)  # reshape(img, 1, imageSize...)  # Gray.(img)           
        # temp = convert(Matrix{ColorTypes.RGB{Float64}}, img)
        # println("$(size(img)), $(size(temp)), $(typeof(img)), $(typeof(temp))")
        # x[index, :, :] = reshape(temp, 1, imageSize...)
    end 
    # x = reshape(x, imageSize..., 3, size(labelsInfo, 1))
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
    @time train_x = read_data("train", labelsInfoTrain, imageSize, path)  # 把数据全部读出到内存
    train_y = map(x -> x[1], labelsInfoTrain[:, "Class"])

    labelsInfoTest = CSV.read("$(path)/sampleSubmission.csv", DataFrame)
    test_x = read_data("test", labelsInfoTest, imageSize, path)
    
    @show size(train_x)  # (6283, 20, 20)
    @show size(train_y)
    images = train_x    # 28, 28, 60000
    labels = train_y
    println("$(size(train_x)), $(size(images)), $(typeof(images))")  # AbstractVector{<:ColorImage} 

    # images = MLJ.table(images);   # 需要table类型的输入
    # images = coerce(images, ColorImage)  # 把Array{T, 20,20,3,n} 转换为 AbstractVector{<:ColorImage}
    labels = categorical(labels)
    labels = coerce(labels, Multiclass)

    println("$(size(train_x)), $(size(images))")

    ImageClassifier = @load ImageClassifier

    clf = ImageClassifier(builder=MyConvBuilder(3, 32, 64, 128),
					  epochs=1000,
                      optimiser=ADAM(0.001),
					  loss=Flux.crossentropy,
                      batch_size=512,
                      acceleration=CUDALibs(),)

    mach = machine(clf, images, labels)  # images:Vector{<:Image}
    # fit!(mach, verbosity=3)
    # fp = fitted_params(mach)

    @time eval = evaluate!(
        mach;
        resampling=Holdout(fraction_train=0.7, shuffle=true, rng=123),
        operation=predict_mode,
        measure=[accuracy, #= cross_entropy,=# misclassification_rate],
        verbosity=4,
    )
    @show eval   # (0.501, 0.499) (0.47, 0.53) (0.438, 0.562) (0.486, 0.514)
    # 179.574995  (0.478, 0.522) (0.565, 0.435) (0.607, 0.393)  (0.649, 0.351) (0.647, 0.353) 
    # (0.656, 0.344) (0.675, 0.325) (0.663, 0.337)  (0.682, 0.318)
    pred_dcrm_cls = predict_mode(mach, test_x)
    @show pred_dcrm_cls[1:10]

    labelsInfoTest[!, :Class] = pred_dcrm_cls
    CSV.write("$(path)/juliaSubmission_11.csv", labelsInfoTest, writeheader=true)
end

process_2()


#=
kagggle google街景字符识别   2022.3.16--3.20
参考:
https://www.kaggle.com/c/street-view-getting-started-with-julia/data

export JULIA_NUM_THREADS=40
export CUDA_VISIBLE_DEVICES=5
ENV["CUDA_VISIBLE_DEVICES"]=5 
julia others/kaggle/k_2/k_2.jl


问题:
1. dataframe还是不够熟练, 高层api不够友好. 
2. MLJ文档不够好, 层次结构不清晰. 
3. 很多三方包不够成熟. 集成不够好, 文档不够. 
4. 怎么获取特征重要程度,特征相关性? mlj现在支持不行
5. 当x是高维数据时,怎么做table类型的输入? xTrain = Tables.table(xTrain);
6. Matrix{RGB{N0f8}} 怎么转为Matrix{Float64}? 
7. 加dataloader,数据增强
8. 加resnet18  OK

using Images, Colors, ImageCore, ImageView
img = load("Albert Einstein-rare-pics56.jpg")
img_RGB = float(Array(channelview(img)))
R_pixel = img_RGB[1,:,:];

=#
