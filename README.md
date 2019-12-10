# JOINER

JOINER is a joint text and Knowledge Graph embedding method using regularization. JOINER not only preserves co-occurrence between words in a text corpus and relations between entities in a Knowledge Graph, it also provides the flexibility to control the amount of information shared between the two data sources via regularization. This method does not generate additional learning samples, which makes it computationally efficient.


Dataset link: https://drive.google.com/open?id=1QRB2lIRfNwphs1I6pAqEC3sT4XoXrLqb


How to compile the code [Tensorflow]

```
gcc compute-accuracy_euclidean.c -o compute-accuracy_euclidean

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )

TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc -o word2vec_ops.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0

python word2vec_optimized.py 0.025
```

In `word2vec_kernels.cc`, you can set the KG training data path, margin and the regularization parameter (they are are coded). E.g.:
```
string train_data_kg_tmp = "../data/freebase_ids_label_without_punctuation_without_duplicates_train_top200kEntities";

float ptranse_margin = 8.f; //pTransE margin

float regulariz_param = 0.002f; //regulariz_param
```

Parameter tuning:

In `word2vec_optimized.py`, you can set:

-where to save the models: `save_path`

-path where to read the text corpus: `train_data`

-size of text and kg embeddings: `embedding_size`

-number of epochs: `epochs_to_train`

-number of negative samples: `num_neg_samples`

-`batch_size size`:500

-number of threads: `concurrent_steps`

-windows text size: `window_size`

-number of min words count: `min_count`

-subsample from w2v: `subsample`

The learning rate is set as input parameter of the python script:
-`learning_rate`: this values must be pass as input parameter of word2vec_optimized.py (E.g.: `python word2vec_optimized.py 0.025`)

NOTE: if you change the text or KG corpuses or min_count, then you have to recreate the anchors!
To recreate the anchors, go to `word2vec_kernels.cc`, and:

-UNCOMMENT:
```
// std::cout << "build_anchors_kg" << "\n";
// anchor_kg = build_anchors_kg();
// std::cout << "build_anchors_text" << "\n";
// anchor_text = build_anchors_text();
// std::cout << "\nANCHORS BUILT\n" << std::flush;
```

-COMMENT:
```
std::cout << "read_anchors_text" << "\n";
anchor_text = read_anchors_text();
std::cout << "read_anchors_kg" << "\n";
anchor_kg = read_anchors_kg();
std::cout << "\nANCHORS READ\n" << std::flush;
```

The new anchors will be saved into the folders with all the scripts. Then move the anchors files to `../data`

# Evaluation tasks

-Analogy: `./compute-accuracy emb_text.bin < questions-words_lower.txt` and `./compute-accuracy emb_text.bin < questions-phrases_lower.txt`

-Link prediction: `./Test_TransE_euclidean bern <knowledge_graph_embedding_path>`

# Python lib versions
Python 2.7.12

matplotlib==2.1.2

scipy==1.0.0

six==1.11.0

numpy==1.14.0

tensorflow==1.5.0

tensorflow-tensorboard==1.5.1

nltk==3.2.5

g++ (Ubuntu 5.4.0-6ubuntu1~16.04.10) 5.4.0 20160609

gcc (Ubuntu 5.4.0-6ubuntu1~16.04.10) 5.4.0 20160609


# Citation

If you found this code or this dataset, please cite:

```
@inproceedings{
  title={Revisiting Text and Knowledge Graph Joint Embeddings: The Amount of Shared Information Matters!},
  author={Paolo Rosso, Dingqi Yang, and Philippe Cudré-Mauroux},
  booktitle={2019 IEEE International Conference on Big Data (Big Data)},
  year={2019},
  organization={IEEE}
}
```
