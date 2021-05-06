using Transformers
using Transformers.Basic
using Transformers.Pretrain
using CUDA
include("/home/zhangyong/cluster/julia_learn/cluster/transformer/example/AttentionIsAllYouNeed/1-model.jl")


function test_1()
    ENV["DATADEPS_ALWAYS_ACCEPT"] = true

    bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"
    vocab = Vocabulary(wordpiece)

    text1 = "Peter Piper picked a peck of pickled peppers" |> tokenizer |> wordpiece
    text2 = "Fuzzy Wuzzy was a bear" |> tokenizer |> wordpiece

    text = ["[CLS]"; text1; "[SEP]"; text2; "[SEP]"]
    @assert text == [
        "[CLS]", "peter", "piper", "picked", "a", "peck", "of", "pick", "##led", "peppers", "[SEP]", 
        "fuzzy", "wu", "##zzy",  "was", "a", "bear", "[SEP]"
    ]

    token_indices = vocab(text)
    segment_indices = [fill(1, length(text1)+2); fill(2, length(text2)+1)]

    sample = (tok = token_indices, segment = segment_indices)

    bert_embedding = sample |> bert_model.embed
    feature_tensors = bert_embedding |> bert_model.transformers
    println(size(feature_tensors))   # (768, 18)
end

function test_2()
    # example/AttentionIsAllYouNeed/1-model.jl
    println(length(CUDA.devices()))
    train!()

end



test_2()


#=
2021.1.30尝试, 2.10走通了 test_1()
1. 此时用的CUDA 1.0, 还不能升级.
2. 模型文件在外网,需要手动下载并放到指定位置. 

2021.3.5   test_2()  ok
Batch = 4
export NVIDIA_VISIBLE_DEVICES='2 3'
$ julia-1.5 transformer_1.jl -g wmt14
CPU:1.8G,100%, GPU:11G,15%
只能julia@1.5, cuda@1.3

怎么做多GPU的数据并行


=#

