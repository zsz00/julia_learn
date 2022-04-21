using MLJ, MLJFlux, Flux
using MLDatasets


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


function mian()
    train_x, train_y = MNIST.traindata()
    @show size(train_y)
    images = train_x    # 28, 28, 60000
    labels = train_y
    images = coerce(images, GrayImage)
    labels = coerce(labels, Multiclass)
    println("$(size(train_x)), $(typeof(train_x)), $(size(images))")
	# (28, 28, 60000), Base.ReinterpretArray{FixedPointNumbers.N0f8, 3, UInt8, Array{UInt8, 3}, false}, (60000,)
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


@time mian()


