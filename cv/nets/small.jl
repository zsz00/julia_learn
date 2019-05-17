using Flux

small_1 = Chain(
  Conv((3,3), 3=>16, relu),  # 卷积层， relu激活函数
  x -> maxpool(x, (2,2)),    # 匿名函数， 最大池化
  Conv((3,3), 16=>32, relu), 
  Conv((3,3), 32=>64, relu),
  x -> maxpool(x, (2,2)), 
  Conv((1,1), 64=>128, relu),
  Conv((3,3), 128=>256, relu),
  x -> maxpool(x, (2,2)),   
  x -> reshape(x, :, size(x, 4)),
  Dense(256, 10),    # 全连接层， 组合应用特征，完成 分类任务。288
  softmax)

