using MLJ
using MLJModelInterface
using DecisionTree
@load DecisionTreeRegressor pkg=DecisionTree verbosity=0  # add=true

# load some data:
# task = load_reduced_ames()
# X, y = task()
X, y = @load_ames

# one-hot encode the inputs, X:
hot_model = OneHotEncoder()
hot = machine(hot_model, X)
MLJ.fit!(hot)
Xt = transform(hot, X)

# fit a decision tree to the transformed data:
tree_model = DecisionTreeRegressor()
tree = machine(tree_model, Xt, y)
DecisionTree.fit!(tree, rows = 1:1300)

