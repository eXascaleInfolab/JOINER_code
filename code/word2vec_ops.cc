/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("SkipgramWord2vec")
    .Output("vocab_word: string")
    .Output("vocab_freq: int32")
    .Output("words_per_epoch: int64")
    .Output("current_epoch: int32")
    .Output("total_words_processed: int64")
    .Output("examples: int32")
    .Output("labels: int32")
    .Output("heads: int32")
    .Output("relations: int32")
    .Output("tails: int32")
    .Output("vocab_entity: string")
    .Output("vocab_relation: string")
    .Output("ent_freq: int32")
    .Output("rel_freq: int32")
    .Output("facts_per_epoch: int64")
    .Output("total_facts_processed: int64")
    .SetIsStateful()
    .Attr("filename: string")
    .Attr("batch_size: int")
    .Attr("window_size: int = 5")
    .Attr("min_count: int = 5")
    .Attr("subsample: float = 1e-3")
    .Attr("filename_kg: string")
    .Doc(R"doc(
Parses a text file and creates a batch of examples.

vocab_word: A vector of words in the corpus.
vocab_freq: Frequencies of words. Sorted in the non-ascending order.
words_per_epoch: Number of words per epoch in the data file.
current_epoch: The current epoch number.
total_words_processed: The total number of words processed so far.
examples: A vector of word ids.
labels: A vector of word ids.
filename: The corpus's text file name.
window_size: The number of words to predict to the left and right of the target.
min_count: The minimum number of word occurrences for it to be included in the
    vocabulary.
subsample: Threshold for word occurrence. Words that appear with higher
    frequency will be randomly down-sampled. Set to 0 to disable.
)doc");

REGISTER_OP("NegTrainWord2vec")
    .Input("w_in: Ref(float)")
    .Input("examples: int32")
    .Input("labels: int32")
    .Input("lr: float")
    .Input("w_in_rel: Ref(float)")
    .Input("heads: int32")
    .Input("relations: int32")
    .Input("tails: int32")
    .Input("w_in_ent: Ref(float)")
    .Input("w_out: Ref(float)")
    .SetIsStateful()
    .Attr("vocab_count: list(int)")
    .Attr("num_negative_samples: int")
    .Attr("ent_count: list(int)")
    .Attr("rel_count: list(int)")
    .Doc(R"doc(
Training via negative sampling.

w_in: input word embedding.
examples: A vector of word ids.
labels: A vector of word ids.
vocab_count: Count of words in the vocabulary.
num_negative_samples: Number of negative samples per example.
)doc");

}  // end namespace tensorflow
