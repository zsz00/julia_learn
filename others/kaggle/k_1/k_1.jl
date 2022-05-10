using MLJ, Statistics
using CSV, DataFrames, PrettyTables, DataFramesMeta
using StableRNGs
using StatsModels
using Strs

# LinearRegressor = @load LinearRegressor pkg=MLJLinearModels  # verbosity=0  GLM  MLJLinearModels
RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree
DTC = @load DecisionTreeClassifier pkg=DecisionTree
# AdaBoostStumpClassifier    DecisionTreeClassifier  RandomForestClassifier
XGBC = @load XGBoostClassifier pkg=XGBoost
# LGBMRegressor = @load LGBMRegressor
LGBMC = @load LGBMClassifier pkg=LightGBM


function set_missing_ages_model(df_train)
    age_df = @select(df_train, :Age,:Fare,:Parch,:SibSp,:Pclass)

    known_age = @subset(age_df, @byrow !ismissing(:Age))  # ok
    unknown_age = @subset(age_df, @byrow ismissing(:Age)) # ok
    y, X = unpack(known_age, ==(:Age))

    rfr = RandomForestRegressor()
    model = machine(rfr, X, y)
    fit!(model)
    fp = fitted_params(model)
    @show fp
    
    return model
end

function process_missing_ages(model, df_train)
    df_train[ismissing.(df_train.Fare), :Fare] .= 8.0
    disallowmissing!(df_train, :Fare)
    age_df = @select(df_train, :Age,:Fare,:Parch,:SibSp,:Pclass)
    unknown_age = @subset(age_df, @byrow ismissing(:Age))
    y_test, X_test = unpack(unknown_age, ==(:Age))
    
    pred_y = predict(model, X_test)
    # @show pred_y
    df_train[ismissing.(df_train.Age), :Age] = pred_y
    # df_train = @subset(df_train, @byrow !ismissing(:Age))  # 取出age正常的
    return df_train
end

function process_1(df_train; step="train")
    if step == "train"
        y, df_train_1 = unpack(df_train, ==(:Survived));
        y = categorical(y)
    else
        df_train_1 = df_train
        y = []
    end
    df_train_2 = @select(df_train_1, :Age,:Fare,:Parch,:SibSp,:Pclass,:Sex)
    df_train_2 = coerce(df_train_2, :Pclass=>Continuous, :Sex=>Multiclass);
    # println(size(df_train_2))
    # @pt MLJ.schema(df_train_2)
    
    encoder = ContinuousEncoder()
    mach = machine(encoder, df_train_2) |> fit!
    X_encoded = MLJ.transform(mach, df_train_2)
    
    return X_encoded, y
end


function predict_survived()
    path = "/home/zhangyong/codes/julia_learn/others/kaggle/k_1/train.csv"
    df_train = DataFrame(CSV.File(path))
    
    age_model = set_missing_ages_model(df_train)
    df_train = process_missing_ages(age_model, df_train)
    X_encoded, y = process_1(df_train)

    # model = OneHotEncoder() |> DTC()
    model = XGBC()  # XGBC()  LGBMC()  DTC()
    dcrm = machine(model, X_encoded, y)
    fit!(dcrm)
    fp = fitted_params(dcrm)
    @show fp.fitresult[1]
    # @show fp.fitresult[1].importance()
    # path = "/home/zhangyong/codes/julia_learn/others/kaggle/k_1/test.csv"
    # df_train = DataFrame(CSV.File(path))
    # df_train = process_missing_ages(age_model, df_train)
    # X_encoded, y = process_1(df_train; step="test")

    println(f"\(size(X_encoded)), \(size(y))")
    pred_dcrm = predict(dcrm, X_encoded)
    pred_dcrm_cls = predict_mode(dcrm, X_encoded)
    @show pred_dcrm_cls[20:40]
    @show pred_dcrm[1:4]
    # out_df = DataFrame(PassengerId=892:1309, Survived=convert(Array{Int64},pred_dcrm_cls))
    # out_path = "/home/zhangyong/codes/julia_learn/others/kaggle/k_1/out_1.csv"
    # CSV.write(out_path, out_df)
    aa = mean(LogLoss(tol=1e-4)(pred_dcrm, y))  # log_loss   0.16995662487369775   0.24579491188658323
    bb = misclassification_rate(pred_dcrm, y)  # 误分类率     1.0  # 结果不对
    println(f"aa:\(aa), bb:\(bb)")   
end


predict_survived()


#=
kagggle 泰坦尼克号 幸存者预测   2022.1.25
参考:
https://www.kaggle.com/c/titanic/data  task page
https://zhuanlan.zhihu.com/p/267141619
https://blog.csdn.net/han_xiaoyang/article/details/49797143
https://juliaai.github.io/DataScienceTutorials.jl/isl/lab-8/index.html

RandomForestRegressor, DecisionTreeClassifier
数据预处理,回归,分类

问题:
1. dataframe还是不够熟练, 高层api不够友好. 
2. MLJ文档不够好, 层次结构不清晰. 
3. 很多三方包不够成熟. 集成不够好, 文档不够. 
4. 怎么获取特征重要程度,特征相关性?

RandomForestRegressor    0.24579491188658323
DecisionTreeClassifier   0.16995662487369775  0.1714180243807161
XGBoostClassifier        0.25722778   0.26236624
LGBMClassifier           0.4550175524945603   0.43918027580360663

=#
