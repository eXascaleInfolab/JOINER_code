#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include <vector>
#include <fstream>
#include <regex>
#include <map>
#include <iostream>
#include <math.h>

#include <stdlib.h>

namespace tensorflow {

const int kPrecalc = 3000;
const int kSentenceSize = 1000;

namespace {

  std::map<string, int32> word_id_tmp;
  std::map<string, int32> entity_id_tmp;
  std::map<string, int32> relation_id_tmp;
  std::map<int32, string> id_word_tmp;
  std::map<int32, string> id_entity_tmp;
  std::map<int32, string> id_relation_tmp;
  int * anchor_text;
  int * anchor_kg;
  float ptranse_margin = 8.f; //pTransE margin
  float regulariz_param = 0.002f; //regulariz_param
  std::map<int,std::map<int,int> > left_entity,right_entity;
  std::map<int,double> left_num,right_num;

  string train_data_kg_tmp = "../data/freebase_ids_label_without_punctuation_without_duplicates_train_top200kEntities";

  //Exponential table
  #define EXP_TABLE_SIZE 1000
  #define MAX_EXP 6
  #define RAND_MULTIPLIER 25214903917
  #define RAND_INCREMENT 11
  float *expTable;

  float sigmoid(float f) {
    if (f >= MAX_EXP) return 1;
    else if (f <= -MAX_EXP) return 0;
    else return expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2 ))];
  }

  //random number generator fast
  unsigned long getNextRand(unsigned long next_random){
    unsigned long next_random_return = next_random * (unsigned long) RAND_MULTIPLIER + RAND_INCREMENT;
    return next_random_return;
  }

  bool ScanWord(StringPiece* input, string* word) {
    str_util::RemoveLeadingWhitespace(input);
    StringPiece tmp;
    if (str_util::ConsumeNonWhitespace(input, &tmp)) {
      word->assign(tmp.data(), tmp.size());
      return true;
    } else {
      return false;
    }
  }

  void compute_norm (Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer> Tw_in, std::map<int32, string> my_map){
    Tensor tensor_buf(DT_FLOAT, TensorShape({}));
    auto tmp_norm = tensor_buf.scalar<float>();
    for (int iii=1; iii<=my_map.size(); iii++){
      auto tmp_vec = Tw_in.chip<0>(iii);
      auto euc_tmp = ((tmp_vec) * (tmp_vec)).sum();
      tmp_norm += euc_tmp;
    }
    std::cout << "tmp_norm: " << tmp_norm << "\n";
  }

  std::vector<string> split(string s, string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    std::vector<string> res;
    while ((pos_end = s.find(delimiter, pos_start)) != string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }
    res.push_back(s.substr(pos_start));
    return res;
  }

  string entityname_to_entity(string entity_name) {
    return "<http://dbpedia.org/resource/" + entity_name + ">";
  }

  string entity_to_entityname(string entity) {
      //if entity==<http://dbpedia.org/resource/anarchism>, then match[1]==anarchism
      std::regex rgx("<http://dbpedia.org/resource/(\\w+[a-zA-Z0-9]+)(.*)>");
      std::smatch match;
      regex_search (entity, match, rgx);
      return match[1];
  }

  int get_value_int (std::map<string, int32> my_map, string item) {
    std::map<string, int32>::iterator it_my_map;
    it_my_map = my_map.find(item);
    if (it_my_map != my_map.end()) {
      return it_my_map->second;
    }
    return -1; //key not found
  }

  string get_value_string (std::map<int32, string> my_map, int item) {
    std::map<int32, string>::iterator it_my_map;
    it_my_map = my_map.find(item);
    if (it_my_map != my_map.end()) {
      return it_my_map->second;
    }
    return ""; //key not found
  }

  int* build_anchors_text () {
    std::ofstream outFile("./anchors_text.txt");
    int *c = new int[word_id_tmp.size()+1];
    c[0]=-1;
    outFile << c[0];
    int i = 1;
    for (auto const& el : id_word_tmp){
      if (fmod(i,1000) == 0) {
        std::cout << "build_anchors_text: " << i << "/" << word_id_tmp.size()+1 << "\n" << std::flush;
      }
      c[i] = get_value_int (entity_id_tmp, el.second); //search if a text word is an entity
      outFile << "\n" << c[i];
      i = i+1;
    }
    return c;
  }

  int* build_anchors_kg () {
    std::ofstream outFile("./anchors_kg.txt");
    int *c = new int[entity_id_tmp.size()];
    int i = 0;
    for (auto const& el : id_entity_tmp){
      if (fmod(i,1000) == 0) {
        std::cout << "build_anchors_kg: " << i << "/" << entity_id_tmp.size() << "\n" << std::flush;
      }
      c[i] = get_value_int (word_id_tmp, el.second); //search if an entity exist in the text
      if (i==entity_id_tmp.size()-1){
        outFile << c[i];
      }
      else{
        outFile << c[i] << "\n";
      }
      i = i+1;
    }
    return c;
  }

  int* read_anchors_text () {
    //text anchor
    int *c = new int[word_id_tmp.size()+1];
    int i = 0;
    std::ifstream fs("../data/anchors_text.txt");
    int my_iterator;
    while (fs >> my_iterator){
      c[i] = my_iterator;
      i++;
    }
    fs.close();
    return c;
  }

  int* read_anchors_kg () {
    //text anchor
    int *c = new int[entity_id_tmp.size()];
    int i = 0;
    std::ifstream fs("../data/anchors_kg.txt");
    int my_iterator;
    while (fs >> my_iterator){
      c[i] = my_iterator;
      i++;
    }
    fs.close();
    return c;
  }

}

class SkipgramWord2vecOp : public OpKernel {
 public:
  explicit SkipgramWord2vecOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), rng_(&philox_) {
    string filename;
    string filename_kg;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("window_size", &window_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("min_count", &min_count_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("subsample", &subsample_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("filename_kg", &filename_kg));
    OP_REQUIRES_OK(ctx, Init(ctx->env(), filename));
    OP_REQUIRES_OK(ctx, InitKg(ctx->env(), filename_kg));
    // std::cout << "build_anchors_kg" << "\n";
    // anchor_kg = build_anchors_kg();
    // std::cout << "build_anchors_text" << "\n";
    // anchor_text = build_anchors_text();
    // std::cout << "\nANCHORS BUILT\n" << std::flush;

    std::cout << "read_anchors_text" << "\n";
    anchor_text = read_anchors_text();
    std::cout << "read_anchors_kg" << "\n";
    anchor_kg = read_anchors_kg();
    std::cout << "\nANCHORS READ\n" << std::flush;

    //exponential table
    expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));
    for (int i=0; i<EXP_TABLE_SIZE; i++) {
      expTable[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); //Precompute the exp() table
      expTable[i] = expTable[i] / (expTable[i] + 1); //Precompute f(x) = x / (x + 1)
    }

    mutex_lock l(mu_);
    example_pos_ = corpus_size_;
    label_pos_ = corpus_size_;
    label_limit_ = corpus_size_;
    sentence_index_ = kSentenceSize;
    sentence_index_kg_ = kSentenceSize;
    for (int i = 0; i < kPrecalc; ++i) {
      NextExample(&precalc_examples_[i].input, &precalc_examples_[i].label);
      NextExampleKg(&precalc_examples_kg_[i].head, &precalc_examples_kg_[i].relation, &precalc_examples_kg_[i].tail);
    }
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor words_per_epoch(DT_INT64, TensorShape({}));
    Tensor facts_per_epoch(DT_INT64, TensorShape({}));
    Tensor current_epoch(DT_INT32, TensorShape({}));
    Tensor total_words_processed(DT_INT64, TensorShape({}));
    Tensor total_facts_processed(DT_INT64, TensorShape({}));
    Tensor examples(DT_INT32, TensorShape({batch_size_}));
    auto Texamples = examples.flat<int32>();
    Tensor labels(DT_INT32, TensorShape({batch_size_}));
    auto Tlabels = labels.flat<int32>();
    Tensor heads(DT_INT32, TensorShape({batch_size_}));
    auto Theads = heads.flat<int32>();
    Tensor relations(DT_INT32, TensorShape({batch_size_}));
    auto Trelations = relations.flat<int32>();
    Tensor tails(DT_INT32, TensorShape({batch_size_}));
    auto Ttails = tails.flat<int32>();

    {
      mutex_lock l(mu_);
      for (int i = 0; i < batch_size_; ++i) {
        Texamples(i) = precalc_examples_[precalc_index_].input;
        Tlabels(i) = precalc_examples_[precalc_index_].label;
        precalc_index_++;
        if (precalc_index_ >= kPrecalc) {
          precalc_index_ = 0;
          for (int j = 0; j < kPrecalc; ++j) {
            NextExample(&precalc_examples_[j].input, &precalc_examples_[j].label);
          }
        }
      }
      for (int i = 0; i < batch_size_; ++i) {
        Theads(i) = precalc_examples_kg_[precalc_index_kg_].head;
        Trelations(i) = precalc_examples_kg_[precalc_index_kg_].relation;
        Ttails(i) = precalc_examples_kg_[precalc_index_kg_].tail;
        precalc_index_kg_++;
        if (precalc_index_kg_ >= kPrecalc) {
          precalc_index_kg_ = 0;
          for (int j = 0; j < kPrecalc; ++j) {
            NextExampleKg(&precalc_examples_kg_[j].head, &precalc_examples_kg_[j].relation, &precalc_examples_kg_[j].tail);
          }
        }
      }
      words_per_epoch.scalar<int64>()() = corpus_size_;
      current_epoch.scalar<int32>()() = current_epoch_;
      total_words_processed.scalar<int64>()() = total_words_processed_;
      total_facts_processed.scalar<int64>()() = total_facts_processed_;
      facts_per_epoch.scalar<int64>()() = relation_size_;
    }
    ctx->set_output(0, word_);
    ctx->set_output(1, freq_);
    ctx->set_output(2, words_per_epoch);
    ctx->set_output(3, current_epoch);
    ctx->set_output(4, total_words_processed);
    ctx->set_output(5, examples);
    ctx->set_output(6, labels);
    ctx->set_output(7, heads);
    ctx->set_output(8, relations);
    ctx->set_output(9, tails);
    ctx->set_output(10, entity_);
    ctx->set_output(11, relation_);
    ctx->set_output(12, freq_entity_);
    ctx->set_output(13, freq_relation_);
    ctx->set_output(14, facts_per_epoch);
    ctx->set_output(15, total_facts_processed);
  }

 private:
  struct Example {
    int32 input;
    int32 label;
  };

  struct ExampleKg {
    int32 head;
    int32 relation;
    int32 tail;
  };

  int32 batch_size_ = 0;
  int32 window_size_ = 5;
  float subsample_ = 1e-3;
  int min_count_ = 5;
  int32 vocab_size_ = 0;
  int32 vocab_entities_size_ = 0;
  int32 vocab_relations_size_ = 0;
  Tensor word_;
  Tensor freq_;
  Tensor entity_;
  Tensor freq_entity_;
  Tensor relation_;
  Tensor freq_relation_;
  int64 corpus_size_ = 0;
  int64 entity_size_ = 0;
  int64 relation_size_ = 0;
  std::vector<int32> corpus_;
  std::vector<int32> corpus_head_;
  std::vector<int32> corpus_relation_;
  std::vector<int32> corpus_tail_;
  std::vector<Example> precalc_examples_;
  std::vector<ExampleKg> precalc_examples_kg_;
  int precalc_index_ = 0;
  int precalc_index_kg_ = 0;
  std::vector<int32> sentence_;
  std::vector<int32> sentence_kg_head_;
  std::vector<int32> sentence_kg_relation_;
  std::vector<int32> sentence_kg_tail_;
  int sentence_index_ = 0;
  int sentence_index_kg_ = 0;

  mutex mu_;
  random::PhiloxRandom philox_ GUARDED_BY(mu_);
  random::SimplePhilox rng_ GUARDED_BY(mu_);
  int32 current_epoch_ GUARDED_BY(mu_) = -1;
  int64 total_words_processed_ GUARDED_BY(mu_) = 0;
  int64 total_facts_processed_ GUARDED_BY(mu_) = 0;
  int64 example_pos_ GUARDED_BY(mu_);
  int64 line_kg_pos_ GUARDED_BY(mu_);
  int32 label_pos_ GUARDED_BY(mu_);
  int32 label_limit_ GUARDED_BY(mu_);

  void NextExample(int32* example, int32* label) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    while (true) {
      if (label_pos_ >= label_limit_) {
        ++total_words_processed_;
        ++sentence_index_;
        if (sentence_index_ >= kSentenceSize) {
          sentence_index_ = 0;
          for (int i = 0; i < kSentenceSize; ++i, ++example_pos_) {
            if (example_pos_ >= corpus_size_) {
              ++current_epoch_;
              example_pos_ = 0;
            }
            if (subsample_ > 0) {
              int32 word_freq = freq_.flat<int32>()(corpus_[example_pos_]);
              float keep_prob =
                  (std::sqrt(word_freq / (subsample_ * corpus_size_)) + 1) *
                  (subsample_ * corpus_size_) / word_freq;
              if (rng_.RandFloat() > keep_prob) {
                i--;
                continue;
              }
            }
            sentence_[i] = corpus_[example_pos_];
          }
        }
        const int32 skip = 1 + rng_.Uniform(window_size_);
        label_pos_ = std::max<int32>(0, sentence_index_ - skip);
        label_limit_ =
            std::min<int32>(kSentenceSize, sentence_index_ + skip + 1);
      }
      if (sentence_index_ != label_pos_) {
        break;
      }
      ++label_pos_;
    }
    *example = sentence_[sentence_index_];
    *label = sentence_[label_pos_++];
  }

  void NextExampleKg(int32* head, int32* relation, int32* tail) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        ++total_facts_processed_;
        ++sentence_index_kg_;
        if (sentence_index_kg_ >= kSentenceSize) {
          sentence_index_kg_ = 0;
          for (int i = 0; i < kSentenceSize; ++i) {
            if (line_kg_pos_ >= relation_size_) {
              // ++current_epoch_;
              line_kg_pos_ = 0;
            }
            sentence_kg_head_[i] = corpus_head_[line_kg_pos_];
            sentence_kg_relation_[i] = corpus_relation_[line_kg_pos_];
            sentence_kg_tail_[i] = corpus_tail_[line_kg_pos_];
            line_kg_pos_++;
          }
        }
    *head = sentence_kg_head_[sentence_index_kg_];
    *relation = sentence_kg_relation_[sentence_index_kg_];
    *tail = sentence_kg_tail_[sentence_index_kg_];
  }

  Status Init(Env* env, const string& filename) {
    string data;
    TF_RETURN_IF_ERROR(ReadFileToString(env, filename, &data));
    StringPiece input = data;
    string w;
    corpus_size_ = 0;
    std::unordered_map<string, int32> word_freq;
    while (ScanWord(&input, &w)) {
      ++(word_freq[w]);
      ++corpus_size_;
    }
    typedef std::pair<string, int32> WordFreq;
    std::vector<WordFreq> ordered;
    for (const auto& p : word_freq) {
      if (p.second >= min_count_) ordered.push_back(p);
    }
    LOG(INFO) << "Data file: " << filename << " contains " << data.size()
              << " bytes, " << corpus_size_ << " words, " << word_freq.size()
              << " unique words, " << ordered.size()
              << " unique frequent words.";
    // word_freq.clear();
    std::sort(ordered.begin(), ordered.end(),
              [](const WordFreq& x, const WordFreq& y) {
                return x.second > y.second;
              });
    vocab_size_ = static_cast<int32>(1 + ordered.size());
    Tensor word(DT_STRING, TensorShape({vocab_size_}));
    Tensor freq(DT_INT32, TensorShape({vocab_size_}));
    word.flat<string>()(0) = "UNK";
    static const int32 kUnkId = 0;
    std::unordered_map<string, int32> word_id;
    int64 total_counted = 0;
    for (std::size_t i = 0; i < ordered.size(); ++i) {
      const auto& w = ordered[i].first;
      auto id = i + 1;
      word.flat<string>()(id) = w;
      auto word_count = ordered[i].second;
      freq.flat<int32>()(id) = word_count;
      total_counted += word_count;
      word_id[w] = id;
      id_word_tmp[id] = w;
      word_id_tmp[w] = id;
    }
    freq.flat<int32>()(kUnkId) = corpus_size_ - total_counted;
    word_ = word;
    freq_ = freq;
    corpus_.reserve(corpus_size_);
    input = data;
    while (ScanWord(&input, &w)) {
      corpus_.push_back(gtl::FindWithDefault(word_id, w, kUnkId));
    }
    precalc_examples_.resize(kPrecalc);
    sentence_.resize(kSentenceSize);

    int tmp_count = 0;
    input = data;
    while (ScanWord(&input, &w)) {
      if (word_freq[w] >= min_count_) {
        tmp_count++;
      }
    }
    word_freq.clear();
    std::cout << "Actual number of words after cleaning: " << tmp_count << "\n";

    return Status::OK();
  }

  Status InitKg(Env* env, const string& filename) {
    string data;
    TF_RETURN_IF_ERROR(ReadFileToString(env, filename, &data));
    StringPiece input = data;
    entity_size_ = 0;
    relation_size_ = 0;
    std::unordered_map<string, int32> entity_freq;
    std::unordered_map<string, int32> relation_freq;

    std::ifstream new_file(train_data_kg_tmp);
    string str_line;
    string delimiter = " ";
    while (std::getline(new_file, str_line)) {
      std::vector<string> splitted_line = split(str_line, delimiter);
      ++(entity_freq[splitted_line.at(0)]);
      ++entity_size_;
      ++(relation_freq[splitted_line.at(1)]);
      ++relation_size_;
      ++(entity_freq[splitted_line.at(2)]);
      ++entity_size_;
    }

    typedef std::pair<string, int32> EntityFreq;
    typedef std::pair<string, int32> RelationFreq;
    std::vector<EntityFreq> ordered_entity;
    std::vector<RelationFreq> ordered_relation;
    for (const auto& p : entity_freq) {
      ordered_entity.push_back(p);
    }
    for (const auto& p : relation_freq) {
      ordered_relation.push_back(p);
    }
    LOG(INFO) << "Data file: " << filename << " contains " << data.size()
              << " bytes, " << entity_size_ << " entities, " << relation_size_ << " relations, " << entity_freq.size()
              << " unique entities, " << relation_freq.size() << " unique relations, " << ordered_entity.size()
              << " unique frequent entities, " << ordered_relation.size() << " unique frequent relations.";
    entity_freq.clear();
    relation_freq.clear();
    std::sort(ordered_entity.begin(), ordered_entity.end(),
              [](const EntityFreq& x, const EntityFreq& y) {
                return x.second > y.second;
              });
    std::sort(ordered_relation.begin(), ordered_relation.end(),
              [](const RelationFreq& x, const RelationFreq& y) {
                return x.second > y.second;
              });
    vocab_entities_size_ = static_cast<int32>(ordered_entity.size());
    vocab_relations_size_ = static_cast<int32>(ordered_relation.size());
    Tensor entity(DT_STRING, TensorShape({vocab_entities_size_}));
    Tensor freq_entity(DT_INT32, TensorShape({vocab_entities_size_}));
    Tensor relation(DT_STRING, TensorShape({vocab_relations_size_}));
    Tensor freq_relation(DT_INT32, TensorShape({vocab_relations_size_}));

    std::unordered_map<string, int32> entity_id;
    std::unordered_map<string, int32> relation_id;
    int64 total_counted_entities = 0;
    int64 total_counted_relations = 0;
    for (std::size_t i = 0; i < ordered_entity.size(); ++i) {
      const auto& w = ordered_entity[i].first;
      auto id = i;
      entity.flat<string>()(id) = w;
      auto entity_count = ordered_entity[i].second;
      freq_entity.flat<int32>()(id) = entity_count;
      total_counted_entities += entity_count;
      entity_id[w] = id;
      entity_id_tmp[w] = id;
      id_entity_tmp[id] = w;
    }
    for (std::size_t i = 0; i < ordered_relation.size(); ++i) {
      const auto& w = ordered_relation[i].first;
      auto id = i;
      relation.flat<string>()(id) = w;
      auto relation_count = ordered_relation[i].second;
      freq_relation.flat<int32>()(id) = relation_count;
      total_counted_relations += relation_count;
      relation_id[w] = id;
      relation_id_tmp[w] = id;
      id_relation_tmp[id] = w;
    }
    entity_ = entity;
    freq_entity_ = freq_entity;
    relation_ = relation;
    freq_relation_ = freq_relation;
    corpus_head_.reserve(entity_size_/2);
    corpus_relation_.reserve(relation_size_);
    corpus_tail_.reserve(entity_size_/2);

    input = data;
    std::ifstream new_file_2(train_data_kg_tmp);
    while (std::getline(new_file_2, str_line)) {
      std::vector<string> splitted_line = split(str_line, delimiter);
      corpus_head_.push_back(gtl::FindWithDefault(entity_id, splitted_line.at(0), -1));
      corpus_relation_.push_back(gtl::FindWithDefault(relation_id, splitted_line.at(1), -1));
      corpus_tail_.push_back(gtl::FindWithDefault(entity_id, splitted_line.at(2), -1));
    }

    precalc_examples_kg_.resize(kPrecalc);
    sentence_kg_head_.resize(kSentenceSize);
    sentence_kg_relation_.resize(kSentenceSize);
    sentence_kg_tail_.resize(kSentenceSize);

    //reducing false negative labels technique
    std::ifstream new_file_3(train_data_kg_tmp);
    while (std::getline(new_file_3, str_line)) {
      std::vector<string> splitted_line = split(str_line, delimiter);
      left_entity[relation_id_tmp[splitted_line.at(1)]][entity_id_tmp[splitted_line.at(0)]]++;
      right_entity[relation_id_tmp[splitted_line.at(1)]][entity_id_tmp[splitted_line.at(2)]]++;
    }

    for (int i=0; i<ordered_relation.size(); i++) {
      double sum1=0,sum2=0;
      for (std::map<int,int>::iterator it = left_entity[i].begin(); it!=left_entity[i].end(); it++) {
        sum1++;
      	sum2+=it->second;
      }
      left_num[i]=sum2/sum1;
    }

    for (int i=0; i<ordered_relation.size(); i++) {
      double sum1=0,sum2=0;
      for (std::map<int,int>::iterator it = right_entity[i].begin(); it!=right_entity[i].end(); it++) {
        sum1++;
      	sum2+=it->second;
      }
      right_num[i]=sum2/sum1;
    }

    return Status::OK();
  }

};

REGISTER_KERNEL_BUILDER(Name("SkipgramWord2vec").Device(DEVICE_CPU), SkipgramWord2vecOp);


class NegTrainWord2vecOp : public OpKernel {
 public:
  explicit NegTrainWord2vecOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    base_.Init(0, 0);

    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_negative_samples", &num_samples_));

    std::vector<int32> vocab_count;
    std::vector<int32> ent_count;
    std::vector<int32> rel_count;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_count", &vocab_count));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ent_count", &ent_count));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rel_count", &rel_count));

    std::vector<float> vocab_weights;
    std::vector<float> vocab_weights_ent;
    std::vector<float> vocab_weights_rel;
    vocab_weights.reserve(vocab_count.size());
    vocab_weights_ent.reserve(ent_count.size());
    vocab_weights_rel.reserve(rel_count.size());
    for (const auto& f : vocab_count) {
      float r = std::pow(static_cast<float>(f), 0.75f);
      vocab_weights.push_back(r);
    }
    for (const auto& f : ent_count) {
      float r = std::pow(static_cast<float>(f), 0.75f);
      vocab_weights_ent.push_back(r);
    }
    for (const auto& f : rel_count) {
      float r = std::pow(static_cast<float>(f), 0.75f);
      vocab_weights_rel.push_back(r);
    }

  }

  void update_embedding_text_eucl(Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer>* Tw_e, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer>* Tw_l, const int32 example, const int32 label, float* lr, int num_samples_, int emb_size, unsigned long sample, Eigen::TensorMap<Eigen::TensorFixedSize<float, Eigen::Sizes<>, 1, long int>, 16, Eigen::MakePointer>& euc, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long int>, 16, Eigen::MakePointer>& gradient, unsigned long *next_random, Eigen::TensorMap<Eigen::TensorFixedSize<float, Eigen::Sizes<>, 1, long int>, 16, Eigen::MakePointer>& prob, int label_avoid) {

    //label_avoid==0 w/o anchors because 0==UNK
    //label_avoid==-1 w anchors because 0 is a kg element

    auto v_in = Tw_e->chip<0>(example); //w, example
    auto v_out = Tw_l->chip<0>(label); //v, label

    //positive
    auto tmp_sub_pos =  v_in - v_out; //w-v
    euc = ((tmp_sub_pos) * (tmp_sub_pos)).sum() * 0.5f;
    prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
    gradient = -(1.f-prob()) * tmp_sub_pos; //derivative of log(sig(z))
    v_in += gradient * (*lr); //w
    v_out -= gradient * (*lr); //v

    //negative
    for (int j = 0; j < num_samples_; ++j) {
      *next_random = getNextRand(*next_random);
      sample = fmod((*next_random)>>16, emb_size); //emb_size == vocab_size
      if (sample != label && sample != label_avoid) { //0 is 'UNK'
        auto v_sample = Tw_l->chip<0>(sample); //negative from v
        auto tmp_sub_neg =  v_in - v_sample; //w - neg
        euc = ((tmp_sub_neg) * (tmp_sub_neg)).sum() * 0.5f;
        prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
        gradient = -(0.f-prob()) * tmp_sub_neg;
        v_in += gradient * (*lr); //w
        v_sample -= gradient * (*lr); //neg
      }
    }
  }

  void update_embedding_text_eucl_with_reg(Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer>* Tw_e, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer>* Tw_l, const int32 example, const int32 label, float* lr, int num_samples_, int emb_size, unsigned long sample, Eigen::TensorMap<Eigen::TensorFixedSize<float, Eigen::Sizes<>, 1, long int>, 16, Eigen::MakePointer>& euc, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long int>, 16, Eigen::MakePointer>& gradient, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long int>, 16, Eigen::MakePointer>& gradient_reg, unsigned long *next_random, Eigen::TensorMap<Eigen::TensorFixedSize<float, Eigen::Sizes<>, 1, long int>, 16, Eigen::MakePointer>& prob, int label_avoid, const int32 anchor, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer>* Tw_anchor, int bool_w_anchor, int bool_v_anchor) {

    //label_avoid==0 w/o anchors because 0==UNK
    //label_avoid==-1 w anchors because 0 is a kg element

    // ||w-v|| + regulariz_param||w-w_anchor||
    // ||w-v|| + regulariz_param||v_anchor-v||

    auto v_in = Tw_e->chip<0>(example); //w
    auto v_out = Tw_l->chip<0>(label); //v
    auto v_anchor = Tw_anchor->chip<0>(anchor); //vector of the anchor

    //positive
    auto tmp_sub_pos =  v_in - v_out; //w-v
    auto tmp_sub_pos_reg = v_anchor; //just to initialize

    if (bool_w_anchor==1 and bool_v_anchor==0) { //anchor for only w/target/example
     auto tmp_sub_pos_reg = v_in - v_anchor; //w - anchor
    }

    else if (bool_w_anchor==0 and bool_v_anchor==1) { //anchor for only v/context/label
      auto tmp_sub_pos_reg = v_anchor - v_out; //anchor - v
    }

    else{
      std::cout << "ERROR, wrong anchor\n";
    }



    euc = (((tmp_sub_pos) * (tmp_sub_pos)).sum() * 0.5f) + ((((tmp_sub_pos_reg) * (tmp_sub_pos_reg)).sum() * 0.5f) * regulariz_param);
    prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
    gradient = -(1.f-prob()) * tmp_sub_pos; //derivative of log(sig(z))


    if (bool_w_anchor==1 and bool_v_anchor==0) { //anchor for only w/target/example/v_in
      // ||w-v|| + regulariz_param||w-w_anchor||
      gradient_reg = -(1.f-prob()) * (tmp_sub_pos + (tmp_sub_pos_reg * regulariz_param)); //derivative of log(sig(z)). Gradient with positive sign.
      v_in += gradient_reg * (*lr); //w
      v_out -= gradient * (*lr); //v

      //negative: ||neg_sample-v|| + regulariz_param||neg_sample-w_anchor|| (neg_sample is a text word)
      for (int j = 0; j < num_samples_; ++j) {
        *next_random = getNextRand(*next_random);
        sample = fmod((*next_random)>>16, emb_size); //emb_size == vocab_size
        if (sample != label && sample != label_avoid) { //avoid 0 because it is 'UNK'
          auto w_sample = Tw_e->chip<0>(sample); //negative sample from text (examples/w)
          auto tmp_sub_neg = w_sample - v_out; //neg_sample - v
          auto tmp_sub_neg_reg = w_sample - v_anchor; //neg_sample - w_anchor
          euc = (((tmp_sub_neg) * (tmp_sub_neg)).sum() * 0.5f) + ((((tmp_sub_neg_reg) * (tmp_sub_neg_reg)).sum() * 0.5f) * regulariz_param);
          prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
          gradient = -(0.f-prob()) * tmp_sub_neg; //derivative of log(sig(z))
          gradient_reg = -(0.f-prob()) * (tmp_sub_neg + (tmp_sub_neg_reg * regulariz_param)); //derivative of log(sig(z)). Gradient with positive sign.
          w_sample += gradient_reg * (*lr); //neg_sample
          v_out -= gradient * (*lr); //v
        }
      }
    }


    else if (bool_w_anchor==0 and bool_v_anchor==1) { //anchor for only v/context/label/v_out
      // ||w-v|| + regulariz_param||v_anchor-v||
      gradient_reg = -(1.f-prob()) * (tmp_sub_pos + (tmp_sub_pos_reg * regulariz_param)); //derivative of log(sig(z)). Gradient with negative sign.
      v_in += gradient * (*lr); //w
      v_out -= gradient_reg * (*lr); //v

      //negative: ||w-neg_sample|| + regulariz_param||v_anchor-neg_sample|| (neg_sample is a text word)
      for (int j = 0; j < num_samples_; ++j) {
        *next_random = getNextRand(*next_random);
        sample = fmod((*next_random)>>16, emb_size); //emb_size == vocab_size
        if (sample != label && sample != label_avoid) { //avoid 0 because it is 'UNK'
          auto v_sample = Tw_l->chip<0>(sample); //negative sample from text (labels/v)
          auto tmp_sub_neg = v_in - v_sample; //w - neg_sample
          auto tmp_sub_neg_reg = v_anchor - v_sample; //v_anchor - neg_sample
          euc = (((tmp_sub_neg) * (tmp_sub_neg)).sum() * 0.5f) + ((((tmp_sub_neg_reg) * (tmp_sub_neg_reg)).sum() * 0.5f) * regulariz_param);
          prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
          gradient = -(0.f-prob()) * tmp_sub_neg; //derivative of log(sig(z))
          gradient_reg = -(0.f-prob()) * (tmp_sub_neg + (tmp_sub_neg_reg * regulariz_param)); //derivative of log(sig(z)). Gradient with negative sign.
          v_in += gradient * (*lr); //w
          v_sample -= gradient_reg * (*lr); //neg_sample
        }
      }
    }

    // std::cout << get_value_string (id_word_tmp, example) << ", " << get_value_string (id_word_tmp, label) << ": label anchor-> " << get_value_string (id_entity_tmp, anchor) << "\n";
  }


  void update_embedding_text_eucl_with_reg_with_w_and_v(Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer>* Tw_e, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer>* Tw_l, const int32 example, const int32 label, float* lr, int num_samples_, int emb_size, unsigned long sample, Eigen::TensorMap<Eigen::TensorFixedSize<float, Eigen::Sizes<>, 1, long int>, 16, Eigen::MakePointer>& euc, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long int>, 16, Eigen::MakePointer>& gradient_reg_w, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long int>, 16, Eigen::MakePointer>& gradient_reg_v, unsigned long *next_random, Eigen::TensorMap<Eigen::TensorFixedSize<float, Eigen::Sizes<>, 1, long int>, 16, Eigen::MakePointer>& prob, int label_avoid, const int32 w_anchor, const int32 v_anchor, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer>* Tw_anchor) {

    //label_avoid==0 w/o anchors because 0==UNK
    //label_avoid==-1 w anchors because 0 is a kg element

    // ||w-v|| + regulariz_param||w-w_anchor|| + regulariz_param||v_anchor-v||

    auto v_in = Tw_e->chip<0>(example); //w
    auto v_out = Tw_l->chip<0>(label); //v
    auto v_w_anchor = Tw_anchor->chip<0>(w_anchor); //w_anchor
    auto v_v_anchor = Tw_anchor->chip<0>(v_anchor); //v_anchor

    //positive
    auto tmp_sub_pos =  v_in - v_out; //w-v
    auto tmp_sub_pos_reg_w = v_in - v_w_anchor; //w-w_anchor
    auto tmp_sub_pos_reg_v = v_v_anchor - v_out; //v_anchor-v

    euc = (((tmp_sub_pos) * (tmp_sub_pos)).sum() * 0.5f) + ((((tmp_sub_pos_reg_w) * (tmp_sub_pos_reg_w)).sum() * 0.5f) * regulariz_param)  + ((((tmp_sub_pos_reg_v) * (tmp_sub_pos_reg_v)).sum() * 0.5f) * regulariz_param);
    prob = ((euc-ptranse_margin).exp() + 1.f).inverse();

    gradient_reg_w = -(1.f-prob()) * (tmp_sub_pos + (tmp_sub_pos_reg_w * regulariz_param)); //positive sign.
    gradient_reg_v = -(1.f-prob()) * (tmp_sub_pos + (tmp_sub_pos_reg_v * regulariz_param)); //negative sign.

    v_in += gradient_reg_w * (*lr); //w
    v_out -= gradient_reg_v * (*lr); //v

    // ||neg_w-neg_v|| + regulariz_param||neg_w-w_anchor|| + regulariz_param||v_anchor-neg_v||
    int sample_w=label_avoid;
    int sample_v=label_avoid;
    for (int j = 0; j < num_samples_; ++j) {
      *next_random = getNextRand(*next_random);
      sample_w = fmod((*next_random)>>16, emb_size);
      *next_random = getNextRand(*next_random);
      sample_v = fmod((*next_random)>>16, emb_size);
      if (sample_w!=label && sample_w!=label_avoid && sample_v!=label && sample_v!=label_avoid) { //avoid 0 because it is 'UNK'
        auto neg_w = Tw_e->chip<0>(sample_w); //negative sample from text examples/w
        auto neg_v = Tw_l->chip<0>(sample_v); //negative sample from text labels/v
        auto tmp_sub_neg = neg_w - neg_v; //neg_w-neg_v
        auto tmp_sub_neg_reg_w = neg_w-v_w_anchor; //neg_w-w_anchor
        auto tmp_sub_neg_reg_v = v_v_anchor-neg_v; //v_anchor-neg_v
        euc = (((tmp_sub_neg) * (tmp_sub_neg)).sum() * 0.5f) + ((((tmp_sub_neg_reg_w) * (tmp_sub_neg_reg_w)).sum() * 0.5f) * regulariz_param)  + ((((tmp_sub_neg_reg_v) * (tmp_sub_neg_reg_v)).sum() * 0.5f) * regulariz_param);
        prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
        gradient_reg_w = -(1.f-prob()) * (tmp_sub_neg + (tmp_sub_neg_reg_w * regulariz_param)); //positive sign.
        gradient_reg_v = -(1.f-prob()) * (tmp_sub_neg + (tmp_sub_neg_reg_v * regulariz_param)); //negative sign.
        neg_w += gradient_reg_w * (*lr); //neg_w
        neg_v -= gradient_reg_v * (*lr); //neg_v
      }

    }

  }


  void update_embedding_kg_eucl(Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer>* Tw_in_h, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer>* Tw_in_r, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer>* Tw_in_t, const int32 head, const int32 relation, const int32 tail, float* lr, int head_emb_size, int rel_emb_size, int tail_emb_size, unsigned long *next_random, int dims, Eigen::TensorMap<Eigen::TensorFixedSize<float, Eigen::Sizes<>, 1, long int>, 16, Eigen::MakePointer>& euc, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long int>, 16, Eigen::MakePointer>& gradient, Eigen::TensorMap<Eigen::TensorFixedSize<float, Eigen::Sizes<>, 1, long int>, 16, Eigen::MakePointer>& prob, int head_avoid, int tail_avoid) {

    //*_avoid == 0 is for text because 0==UNK (w/ anchors)
    //*_avoid == -1 is for kg because 0 is an element (w/o anchors)


    //positive
    auto v_head = Tw_in_h->chip<0>(head);
    auto v_relation = Tw_in_r->chip<0>(relation);
    auto v_tail = Tw_in_t->chip<0>(tail);

    //positive
    auto tmp_sub_pos =  v_head+v_relation-v_tail;
    euc = ((tmp_sub_pos) * (tmp_sub_pos)).sum() * 0.5f;
    prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
    gradient = -(1.f-prob()) * tmp_sub_pos; //derivative of log(sig(z))
    v_head += gradient * (*lr);
    v_relation += gradient * (*lr);
    v_tail -= gradient * (*lr);

    //negative head
    int sample_head=head_avoid; //head_avoid == -1
    for (int j = 0; j < num_samples_; ++j) {
      *next_random = getNextRand(*next_random);
      sample_head = fmod((*next_random)>>16, head_emb_size); //head_emb_size == entity_size
      if (sample_head != head && sample_head != head_avoid) { //head_avoid == -1
        auto neg_head = Tw_in_h->chip<0>(sample_head);
        auto tmp_sub_neg =  neg_head+v_relation-v_tail;
        euc = ((tmp_sub_neg) * (tmp_sub_neg)).sum() * 0.5f;
        prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
        gradient = -(0.f-prob()) * tmp_sub_neg;
        neg_head += gradient * (*lr);
        v_relation += gradient * (*lr);
        v_tail -= gradient * (*lr);
      }
    }


    //positive
    auto tmp_sub_pos_2 =  v_head+v_relation-v_tail;
    euc = ((tmp_sub_pos_2) * (tmp_sub_pos_2)).sum() * 0.5f;
    prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
    gradient = -(1.f-prob()) * tmp_sub_pos_2;
    v_head += gradient * (*lr);
    v_relation += gradient * (*lr);
    v_tail -= gradient * (*lr);

    //negative relation
    int sample_relation = -1; //to avoid -1 index during the kg sampling
    for (int j = 0; j < num_samples_; ++j) {
      *next_random = getNextRand(*next_random);
      sample_relation = fmod((*next_random)>>16, rel_emb_size); //rel_emb_size == relation_size
      if (sample_relation != relation && sample_relation != -1) { //to avoid -1 index during the kg sampling
        auto neg_relation = Tw_in_r->chip<0>(sample_relation);
        auto tmp_sub_neg =  v_head+neg_relation-v_tail;
        euc = ((tmp_sub_neg) * (tmp_sub_neg)).sum() * 0.5f;
        prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
        gradient = -(0.f-prob()) * tmp_sub_neg;
        v_head += gradient * (*lr);
        neg_relation += gradient * (*lr);
        v_tail -= gradient * (*lr);
      }
    }


    //positive
    auto tmp_sub_pos_3 =  v_head+v_relation-v_tail;
    euc = ((tmp_sub_pos_3) * (tmp_sub_pos_3)).sum() * 0.5f;
    prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
    gradient = -(1.f-prob()) * tmp_sub_pos_3;
    v_head += gradient * (*lr);
    v_relation += gradient * (*lr);
    v_tail -= gradient * (*lr);

    //negative tail
    int sample_tail = tail_avoid; //tail_avoid == -1
    for (int j = 0; j < num_samples_; ++j) {
      *next_random = getNextRand(*next_random);
      sample_tail = fmod((*next_random)>>16, tail_emb_size); //tail_emb_size == entity_size
      if (sample_tail != tail && sample_tail != tail_avoid){
        auto neg_tail = Tw_in_t->chip<0>(sample_tail);
        auto tmp_sub_neg =  v_head+v_relation-neg_tail;
        euc = ((tmp_sub_neg) * (tmp_sub_neg)).sum() * 0.5f;
        prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
        gradient = -(0.f-prob()) * tmp_sub_neg;
        v_head += gradient * (*lr);
        v_relation += gradient * (*lr);
        neg_tail -= gradient * (*lr);
      }
    }

  }


  void update_embedding_kg_eucl_with_reg(Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer>* Tw_in_h, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer>* Tw_in_r, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer>* Tw_in_t, const int32 head, const int32 relation, const int32 tail, float* lr, int kg_emb_size, unsigned long *next_random, int dims, Eigen::TensorMap<Eigen::TensorFixedSize<float, Eigen::Sizes<>, 1, long int>, 16, Eigen::MakePointer>& euc, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long int>, 16, Eigen::MakePointer>& gradient, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long int>, 16, Eigen::MakePointer>& gradient_reg, Eigen::TensorMap<Eigen::TensorFixedSize<float, Eigen::Sizes<>, 1, long int>, 16, Eigen::MakePointer>& prob, int head_avoid, int tail_avoid, const int32 anchor, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer>* Tw_anchor, int rel_emb_size) {

    //*_avoid == 0 is for text because 0==UNK (w/ anchors)
    //*_avoid == -1 is for kg because 0 is an element (w/o anchors)

    // ||h+r-t|| + regulariz_param||h-h_anchor||
    // ||h+r-t|| + regulariz_param||t_anchor-t||

    auto v_head = Tw_in_h->chip<0>(head); //head
    auto v_relation = Tw_in_r->chip<0>(relation); //relation
    auto v_tail = Tw_in_t->chip<0>(tail); //tail
    auto v_anchor = Tw_anchor->chip<0>(anchor); //anchor

    auto tmp_sub_pos = v_head+v_relation-v_tail; //h+r-t
    auto tmp_sub_pos_reg = v_head; //just to initialize

    if (head_avoid==0 && tail_avoid==-1) {//head anchor
      // ||h+r-t|| + regulariz_param||h-h_anchor||
      auto tmp_sub_pos_reg = v_head-v_anchor; //h-h_anchor
    }

    else if (head_avoid==-1 && tail_avoid==0) {//tail anchor
      // ||h+r-t|| + regulariz_param||t_anchor-t||
      auto tmp_sub_pos_reg = v_anchor-v_tail; //t_anchor-t

      // std::cout << get_value_string (id_entity_tmp, head) << ", " << get_value_string (id_relation_tmp, relation) << ", " << get_value_string (id_entity_tmp, tail) << ": head anchor-> " << get_value_string (id_word_tmp, anchor) << "\n";
    }

    else{
      std::cout << "ERROR, wrong anchor\n";
    }

    euc = (((tmp_sub_pos) * (tmp_sub_pos)).sum() * 0.5f) + ((((tmp_sub_pos_reg) * (tmp_sub_pos_reg)).sum() * 0.5f) * regulariz_param);
    prob = ((euc-ptranse_margin).exp() + 1.f).inverse();

    gradient = -(1.f-prob()) * tmp_sub_pos; //derivative of log(sig(z))


    if (head_avoid==0 && tail_avoid==-1) {//head anchor
      // ||h+r-t|| + regulariz_param||h-h_anchor||
      gradient_reg = -(1.f-prob()) * (tmp_sub_pos + (tmp_sub_pos_reg * regulariz_param)); //gradient_reg with positive sign (head is positive)
      v_head += gradient_reg * (*lr); //h
      v_relation += gradient * (*lr); //r
      v_tail -= gradient * (*lr); //t

      //negative: ||neg_sample+r-t|| + regulariz_param||neg_sample-h_anchor|| (neg_sample is a kg entity)
      int sample_head=tail_avoid; //we want to avoid -1
      for (int j = 0; j < num_samples_; ++j) {
        *next_random = getNextRand(*next_random);
        sample_head = fmod((*next_random)>>16, kg_emb_size); //kg_emb_size == entity_size
        if (sample_head != head && sample_head != tail_avoid) { //tail_avoid == -1 because we sample from kg
          auto neg_head = Tw_in_h->chip<0>(sample_head); //neg_head is a kg entity
          auto tmp_sub_neg = neg_head+v_relation-v_tail; //neg_sample+r-t
          auto tmp_sub_neg_reg = neg_head-v_anchor; //neg_sample-h_anchor
          euc = (((tmp_sub_neg) * (tmp_sub_neg)).sum() * 0.5f) + ((((tmp_sub_neg_reg) * (tmp_sub_neg_reg)).sum() * 0.5f) * regulariz_param);
          prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
          gradient = -(0.f-prob()) * tmp_sub_neg;
          gradient_reg = -(0.f-prob()) * (tmp_sub_neg + (tmp_sub_neg_reg * regulariz_param)); //derivative of log(sig(z)). gradient_reg with positive sign
          neg_head += gradient_reg * (*lr); //neg_sample
          v_relation += gradient * (*lr); //r
          v_tail -= gradient * (*lr); //t
        }
      }

      //positive
      auto tmp_sub_pos_2 = v_head+v_relation-v_tail; //h+r-t
      auto tmp_sub_pos_reg_2 = v_head-v_anchor; //h-h_anchor
      euc = (((tmp_sub_pos_2) * (tmp_sub_pos_2)).sum() * 0.5f) + ((((tmp_sub_pos_reg_2) * (tmp_sub_pos_reg_2)).sum() * 0.5f) * regulariz_param);
      prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
      gradient = -(1.f-prob()) * tmp_sub_pos_2; //derivative of log(sig(z))
      gradient_reg = -(1.f-prob()) * (tmp_sub_pos_2 + (tmp_sub_pos_reg_2 * regulariz_param)); //gradient_reg with positive sign (head is positive)
      v_head += gradient_reg * (*lr); //h
      v_relation += gradient * (*lr); //r
      v_tail -= gradient * (*lr); //t
      //negative relation: ||h+neg_sample-t|| + regulariz_param||h-h_anchor|| (neg_sample is a relation)
      int sample_relation=-1; //we want to avoid -1
      for (int j = 0; j < num_samples_; ++j) {
        *next_random = getNextRand(*next_random);
        sample_relation = fmod((*next_random)>>16, rel_emb_size); //rel_emb_size == relation_size
        if (sample_relation != relation && sample_relation != -1) { //-1 because we sample from relation
          auto neg_relation = Tw_in_r->chip<0>(sample_relation); //neg_relation is a relation
          auto tmp_sub_neg_2 = v_head+neg_relation-v_tail; //h+neg_sample-t
          auto tmp_sub_neg_reg_2 = v_head-v_anchor; //h-h_anchor
          euc = (((tmp_sub_neg_2) * (tmp_sub_neg_2)).sum() * 0.5f) + ((((tmp_sub_neg_reg_2) * (tmp_sub_neg_reg_2)).sum() * 0.5f) * regulariz_param);
          prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
          gradient = -(0.f-prob()) * tmp_sub_neg_2;
          gradient_reg = -(0.f-prob()) * (tmp_sub_neg_2 + (tmp_sub_neg_reg_2 * regulariz_param)); //derivative of log(sig(z)). gradient_reg with positive sign
          v_head += gradient_reg * (*lr); //h
          neg_relation += gradient * (*lr); //neg_sample
          v_tail -= gradient * (*lr); //t
        }
      }


      //positive
      auto tmp_sub_pos_3 = v_head+v_relation-v_tail; //h+r-t
      auto tmp_sub_pos_reg_3 = v_head-v_anchor; //h-h_anchor
      euc = (((tmp_sub_pos_3) * (tmp_sub_pos_3)).sum() * 0.5f) + ((((tmp_sub_pos_reg_3) * (tmp_sub_pos_reg_3)).sum() * 0.5f) * regulariz_param);
      prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
      gradient = -(1.f-prob()) * tmp_sub_pos_3; //derivative of log(sig(z))
      gradient_reg = -(1.f-prob()) * (tmp_sub_pos_3 + (tmp_sub_pos_reg_3 * regulariz_param)); //gradient_reg with positive sign (head is positive)
      v_head += gradient_reg * (*lr); //h
      v_relation += gradient * (*lr); //r
      v_tail -= gradient * (*lr); //t
      //negative tail: ||h+r-neg_sample|| + regulariz_param||h-h_anchor|| (neg_sample is a kg entity)
      int sample_tail=-1; //we want to avoid -1
      for (int j = 0; j < num_samples_; ++j) {
        *next_random = getNextRand(*next_random);
        sample_tail = fmod((*next_random)>>16, kg_emb_size); //kg_emb_size == entity_size
        if (sample_tail != tail && sample_tail != -1) { //-1 because we sample from kg
          auto neg_tail = Tw_in_t->chip<0>(sample_tail); //neg_tail is a kg entity
          auto tmp_sub_neg_3 = v_head+v_relation-neg_tail; //h+r-neg_sample
          auto tmp_sub_neg_reg_3 = v_head-v_anchor; //h-h_anchor
          euc = (((tmp_sub_neg_3) * (tmp_sub_neg_3)).sum() * 0.5f) + ((((tmp_sub_neg_reg_3) * (tmp_sub_neg_reg_3)).sum() * 0.5f) * regulariz_param);
          prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
          gradient = -(0.f-prob()) * tmp_sub_neg_3;
          gradient_reg = -(0.f-prob()) * (tmp_sub_neg_3 + (tmp_sub_neg_reg_3 * regulariz_param)); //derivative of log(sig(z)). gradient_reg with positive sign
          v_head += gradient_reg * (*lr); //h
          v_relation += gradient * (*lr); //r
          neg_tail -= gradient * (*lr); //neg_sample
        }
      }




    }

    else if (head_avoid==-1 && tail_avoid==0) {//tail anchor
      //positive: ||h+r-t|| + regulariz_param||t_anchor-t||
      gradient_reg = -(1.f-prob()) * (tmp_sub_pos + (tmp_sub_pos_reg * regulariz_param)); //derivative of log(sig(z)). gradient_reg with negative sign
      v_head += gradient * (*lr); //h
      v_relation += gradient * (*lr); //r
      v_tail -= gradient_reg * (*lr); //t
      // std::cout << get_value_string (id_entity_tmp, head) << ", " << get_value_string (id_relation_tmp, relation) << ", " << get_value_string (id_entity_tmp, tail) << ": head anchor-> " << get_value_string (id_word_tmp, anchor) << "\n";


      //negative tail: ||h+r-neg_sample|| + regulariz_param||t_anchor-neg_sample||
      int sample_tail = -1; //we want to avoid -1
      for (int j = 0; j < num_samples_; ++j) {
        *next_random = getNextRand(*next_random);
        sample_tail = fmod((*next_random)>>16, kg_emb_size); //kg_emb_size == entity_size
        if (sample_tail != tail && sample_tail != -1){ //avoid -1
          auto neg_tail = Tw_in_t->chip<0>(sample_tail); //neg_tail is a kg entity
          auto tmp_sub_neg = v_head+v_relation-neg_tail; //h+r-neg_sample
          auto tmp_sub_neg_reg = v_anchor-neg_tail; //t_anchor-neg_sample
          euc = (((tmp_sub_neg) * (tmp_sub_neg)).sum() * 0.5f) + ((((tmp_sub_neg_reg) * (tmp_sub_neg_reg)).sum() * 0.5f) * regulariz_param);
          prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
          gradient = -(0.f-prob()) * tmp_sub_neg;
          gradient_reg = -(0.f-prob()) * (tmp_sub_neg + (tmp_sub_neg_reg * regulariz_param)); //derivative of log(sig(z)). gradient_reg with negative sign
          v_head += gradient * (*lr); //h
          v_relation += gradient * (*lr); //r
          neg_tail -= gradient_reg * (*lr); //neg_sample
        }
      }


      //positive: ||h+r-t|| + regulariz_param||t_anchor-t||
      auto tmp_sub_pos_2 = v_head+v_relation-v_tail; //h+r-t
      auto tmp_sub_pos_reg_2 = v_anchor-v_tail; //t_anchor-t
      euc = (((tmp_sub_pos_2) * (tmp_sub_pos_2)).sum() * 0.5f) + ((((tmp_sub_pos_reg_2) * (tmp_sub_pos_reg_2)).sum() * 0.5f) * regulariz_param);
      prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
      gradient = -(1.f-prob()) * tmp_sub_pos_2; //derivative of log(sig(z))
      gradient_reg = -(1.f-prob()) * (tmp_sub_pos_2 + (tmp_sub_pos_reg_2 * regulariz_param)); //derivative of log(sig(z)). gradient_reg with negative sign
      v_head += gradient * (*lr); //h
      v_relation += gradient * (*lr); //r
      v_tail -= gradient_reg * (*lr); //t
      //negative ralation: ||h+neg_sample-t|| + regulariz_param||t_anchor-t||
      int sample_relation = -1; //we want to avoid -1
      for (int j = 0; j < num_samples_; ++j) {
        *next_random = getNextRand(*next_random);
        sample_relation = fmod((*next_random)>>16, rel_emb_size); //rel_emb_size == relation_size
        if (sample_relation != relation && sample_relation != -1){ //sample_relation == -1
          auto neg_relation = Tw_in_r->chip<0>(sample_relation); //neg_relation is a kg entity
          auto tmp_sub_neg_2 = v_head+neg_relation-v_tail; //h+neg_sample-t
          auto tmp_sub_neg_reg_2 = v_anchor-v_tail; //t_anchor-t
          euc = (((tmp_sub_neg_2) * (tmp_sub_neg_2)).sum() * 0.5f) + ((((tmp_sub_neg_reg_2) * (tmp_sub_neg_reg_2)).sum() * 0.5f) * regulariz_param);
          prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
          gradient = -(0.f-prob()) * tmp_sub_neg_2;
          gradient_reg = -(0.f-prob()) * (tmp_sub_neg_2 + (tmp_sub_neg_reg_2 * regulariz_param)); //derivative of log(sig(z)). gradient_reg with negative sign
          v_head += gradient * (*lr); //h
          neg_relation += gradient * (*lr); //neg_sample
          v_tail -= gradient_reg * (*lr); //t
        }
      }


      //positive: ||h+r-t|| + regulariz_param||t_anchor-t||
      auto tmp_sub_pos_3 = v_head+v_relation-v_tail; //h+r-t
      auto tmp_sub_pos_reg_3 = v_anchor-v_tail; //t_anchor-t
      euc = (((tmp_sub_pos_3) * (tmp_sub_pos_3)).sum() * 0.5f) + ((((tmp_sub_pos_reg_3) * (tmp_sub_pos_reg_3)).sum() * 0.5f) * regulariz_param);
      prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
      gradient = -(1.f-prob()) * tmp_sub_pos_3; //derivative of log(sig(z))
      gradient_reg = -(1.f-prob()) * (tmp_sub_pos_3 + (tmp_sub_pos_reg_3 * regulariz_param)); //derivative of log(sig(z)). gradient_reg with negative sign
      v_head += gradient * (*lr); //h
      v_relation += gradient * (*lr); //r
      v_tail -= gradient_reg * (*lr); //t
      //negative head: ||neg_sample+r-t|| + regulariz_param||t_anchor-t||
      int sample_head = -1; //we want to avoid -1
      for (int j = 0; j < num_samples_; ++j) {
        *next_random = getNextRand(*next_random);
        sample_head = fmod((*next_random)>>16, kg_emb_size); //kg_emb_size == entity_size
        if (sample_head != head && sample_head != -1){ //sample_head == -1
          auto neg_head = Tw_in_h->chip<0>(sample_head); //neg_head is a kg entity
          auto tmp_sub_neg_3 = neg_head+v_relation-v_tail; //neg_sample+r-t
          auto tmp_sub_neg_reg_3 = v_anchor-v_tail; //t_anchor-t
          euc = (((tmp_sub_neg_3) * (tmp_sub_neg_3)).sum() * 0.5f) + ((((tmp_sub_neg_reg_3) * (tmp_sub_neg_reg_3)).sum() * 0.5f) * regulariz_param);
          prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
          gradient = -(0.f-prob()) * tmp_sub_neg_3;
          gradient_reg = -(0.f-prob()) * (tmp_sub_neg_3 + (tmp_sub_neg_reg_3 * regulariz_param)); //derivative of log(sig(z)). gradient_reg with negative sign
          neg_head += gradient * (*lr); //neg_sample
          v_relation += gradient * (*lr); //r
          v_tail -= gradient_reg * (*lr); //t
        }
      }



    }

  }

  void update_embedding_kg_eucl_with_reg_h_and_t(Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer>* Tw_in_h, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer>* Tw_in_r, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer>* Tw_in_t, const int32 head, const int32 relation, const int32 tail, float* lr, int kg_emb_size, unsigned long *next_random, int dims, Eigen::TensorMap<Eigen::TensorFixedSize<float, Eigen::Sizes<>, 1, long int>, 16, Eigen::MakePointer>& euc, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long int>, 16, Eigen::MakePointer>& gradient, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long int>, 16, Eigen::MakePointer>& gradient_reg_h, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long int>, 16, Eigen::MakePointer>& gradient_reg_t, Eigen::TensorMap<Eigen::TensorFixedSize<float, Eigen::Sizes<>, 1, long int>, 16, Eigen::MakePointer>& prob, int sample_avoid, const int32 h_anchor, const int32 t_anchor, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer>* Tw_anchor, int rel_emb_size) {

    //*_avoid == 0 is for text because 0==UNK (w/ anchors)
    //*_avoid == -1 is for kg because 0 is an element (w/o anchors)

    //positive
    // ||h+r-t|| + regulariz_param||h-h_anchor|| + regulariz_param||t_anchor-t||

    auto v_head = Tw_in_h->chip<0>(head); //h
    auto v_relation = Tw_in_r->chip<0>(relation); //r
    auto v_tail = Tw_in_t->chip<0>(tail); //t
    auto v_h_anchor = Tw_anchor->chip<0>(h_anchor); //h_anchor
    auto v_t_anchor = Tw_anchor->chip<0>(t_anchor); //t_anchor

    auto tmp_sub_pos = v_head+v_relation-v_tail; //h+r-t
    auto tmp_sub_pos_reg_h = v_head-v_h_anchor; //h-h_anchor
    auto tmp_sub_pos_reg_t = v_t_anchor-v_tail; //t_anchor-t


    // std::cout << get_value_string (id_entity_tmp, head) << ", " << get_value_string (id_relation_tmp, relation) << ", " << get_value_string (id_entity_tmp, tail) << ": head anchor-> " << get_value_string (id_word_tmp, anchor) << "\n";


    euc = (((tmp_sub_pos) * (tmp_sub_pos)).sum() * 0.5f) + ((((tmp_sub_pos_reg_h) * (tmp_sub_pos_reg_h)).sum() * 0.5f) * regulariz_param) + ((((tmp_sub_pos_reg_t) * (tmp_sub_pos_reg_t)).sum() * 0.5f) * regulariz_param);
    prob = ((euc-ptranse_margin).exp() + 1.f).inverse();

    gradient = -(1.f-prob()) * tmp_sub_pos;

    gradient_reg_h = -(1.f-prob()) * (tmp_sub_pos + (tmp_sub_pos_reg_h * regulariz_param)); //derivative of log(sig(z)). positive gradienr
    gradient_reg_t = -(1.f-prob()) * (tmp_sub_pos + (tmp_sub_pos_reg_t * regulariz_param)); //derivative of log(sig(z)). negative gradient

    v_head += gradient_reg_h * (*lr); //h
    v_relation += gradient * (*lr); //r
    v_tail -= gradient_reg_t * (*lr); //t

    //negative head: ||neg_h+r-t|| + regulariz_param||neg_h-h_anchor|| + regulariz_param||t_anchor-t||
    int sample_head=-1; //-1 because we sample from the kg
    for (int j = 0; j < num_samples_; ++j) {
      *next_random = getNextRand(*next_random);
      sample_head = fmod((*next_random)>>16, kg_emb_size); //kg_emb_size == entity_size
      if (sample_head != head && sample_head != -1) {
        auto neg_head = Tw_in_h->chip<0>(sample_head); //neg_head is a kg entity
        auto tmp_sub_neg = neg_head+v_relation-v_tail; //neg_h+r-t
        auto tmp_sub_neg_reg_h = neg_head-v_h_anchor; //neg_h-h_anchor
        auto tmp_sub_neg_reg_t = v_t_anchor-v_tail; //t_anchor-t
        euc = (((tmp_sub_neg) * (tmp_sub_neg)).sum() * 0.5f) + ((((tmp_sub_neg_reg_h) * (tmp_sub_neg_reg_h)).sum() * 0.5f) * regulariz_param) + ((((tmp_sub_neg_reg_t) * (tmp_sub_neg_reg_t)).sum() * 0.5f) * regulariz_param);
        prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
        gradient = -(0.f-prob()) * tmp_sub_neg;
        gradient_reg_h = -(0.f-prob()) * (tmp_sub_neg + (tmp_sub_neg_reg_h * regulariz_param)); //derivative of log(sig(z)). positive sign.
        gradient_reg_t = -(0.f-prob()) * (tmp_sub_neg + (tmp_sub_neg_reg_t * regulariz_param)); //derivative of log(sig(z)). negative sign.
        neg_head += gradient_reg_h * (*lr); //neg_h
        v_relation += gradient * (*lr); //r
        v_tail -= gradient_reg_t * (*lr); //t
      }
    }

    //positive
    // ||h+r-t|| + regulariz_param||h-h_anchor|| + regulariz_param||t_anchor-t||
    auto tmp_sub_pos_2 = v_head+v_relation-v_tail; //h+r-t
    auto tmp_sub_pos_reg_h_2 = v_head-v_h_anchor; //h-h_anchor
    auto tmp_sub_pos_reg_t_2 = v_t_anchor-v_tail; //t_anchor-t
    euc = (((tmp_sub_pos_2) * (tmp_sub_pos_2)).sum() * 0.5f) + ((((tmp_sub_pos_reg_h_2) * (tmp_sub_pos_reg_h_2)).sum() * 0.5f) * regulariz_param) + ((((tmp_sub_pos_reg_t_2) * (tmp_sub_pos_reg_t_2)).sum() * 0.5f) * regulariz_param);
    prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
    gradient = -(1.f-prob()) * tmp_sub_pos_2;
    gradient_reg_h = -(1.f-prob()) * (tmp_sub_pos_2 + (tmp_sub_pos_reg_h_2 * regulariz_param)); //derivative of log(sig(z)). positive gradienr
    gradient_reg_t = -(1.f-prob()) * (tmp_sub_pos_2 + (tmp_sub_pos_reg_t_2 * regulariz_param)); //derivative of log(sig(z)). negative gradient
    v_head += gradient_reg_h * (*lr); //h
    v_relation += gradient * (*lr); //r
    v_tail -= gradient_reg_t * (*lr); //t
    //negative relation: ||h+neg_r-t|| + regulariz_param||h-h_anchor|| + regulariz_param||t_anchor-t||
    int sample_relation=-1; //-1 because we sample from the relation
    for (int j = 0; j < num_samples_; ++j) {
      *next_random = getNextRand(*next_random);
      sample_relation = fmod((*next_random)>>16, rel_emb_size); //rel_emb_size == relation_size
      if (sample_relation != relation && sample_relation != -1) {
        auto neg_relation = Tw_in_r->chip<0>(sample_relation); //neg_relation is a relation
        auto tmp_sub_neg_2 = v_head+neg_relation-v_tail; //h+neg_r-t
        auto tmp_sub_neg_reg_h_2 = v_head-v_h_anchor; //h-h_anchor
        auto tmp_sub_neg_reg_t_2 = v_t_anchor-v_tail; //t_anchor-t
        euc = (((tmp_sub_neg_2) * (tmp_sub_neg_2)).sum() * 0.5f) + ((((tmp_sub_neg_reg_h_2) * (tmp_sub_neg_reg_h_2)).sum() * 0.5f) * regulariz_param) + ((((tmp_sub_neg_reg_t_2) * (tmp_sub_neg_reg_t_2)).sum() * 0.5f) * regulariz_param);
        prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
        gradient = -(0.f-prob()) * tmp_sub_neg_2;
        gradient_reg_h = -(0.f-prob()) * (tmp_sub_neg_2 + (tmp_sub_neg_reg_h_2 * regulariz_param)); //derivative of log(sig(z)). positive sign.
        gradient_reg_t = -(0.f-prob()) * (tmp_sub_neg_2 + (tmp_sub_neg_reg_t_2 * regulariz_param)); //derivative of log(sig(z)). negative sign.
        v_head += gradient_reg_h * (*lr); //h
        neg_relation += gradient * (*lr); //neg_r
        v_tail -= gradient_reg_t * (*lr); //t
      }
    }


    //positive
    // ||h+r-t|| + regulariz_param||h-h_anchor|| + regulariz_param||t_anchor-t||
    auto tmp_sub_pos_3 = v_head+v_relation-v_tail; //h+r-t
    auto tmp_sub_pos_reg_h_3 = v_head-v_h_anchor; //h-h_anchor
    auto tmp_sub_pos_reg_t_3 = v_t_anchor-v_tail; //t_anchor-t
    euc = (((tmp_sub_pos_3) * (tmp_sub_pos_3)).sum() * 0.5f) + ((((tmp_sub_pos_reg_h_3) * (tmp_sub_pos_reg_h_3)).sum() * 0.5f) * regulariz_param) + ((((tmp_sub_pos_reg_t_3) * (tmp_sub_pos_reg_t_3)).sum() * 0.5f) * regulariz_param);
    prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
    gradient = -(1.f-prob()) * tmp_sub_pos_3;
    gradient_reg_h = -(1.f-prob()) * (tmp_sub_pos_3 + (tmp_sub_pos_reg_h_3 * regulariz_param)); //derivative of log(sig(z)). positive gradienr
    gradient_reg_t = -(1.f-prob()) * (tmp_sub_pos_3 + (tmp_sub_pos_reg_t_3 * regulariz_param)); //derivative of log(sig(z)). negative gradient
    v_head += gradient_reg_h * (*lr); //h
    v_relation += gradient * (*lr); //r
    v_tail -= gradient_reg_t * (*lr); //t
    //negative tail: ||h+r-neg_t|| + regulariz_param||h-h_anchor|| + regulariz_param||t_anchor-neg_t||
    int sample_tail=-1; //-1 because we sample from the kg
    for (int j = 0; j < num_samples_; ++j) {
      *next_random = getNextRand(*next_random);
      sample_tail = fmod((*next_random)>>16, kg_emb_size); //kg_emb_size == entity_size
      if (sample_tail != tail && sample_tail != -1) {
        auto neg_tail = Tw_in_t->chip<0>(sample_tail); //neg_tail is a kg entity
        auto tmp_sub_neg_3 = v_head+v_relation-neg_tail; //h+r-neg_t
        auto tmp_sub_neg_reg_h_3 = v_head-v_h_anchor; //h-h_anchor
        auto tmp_sub_neg_reg_t_3 = v_t_anchor-neg_tail; //t_anchor-neg_t
        euc = (((tmp_sub_neg_3) * (tmp_sub_neg_3)).sum() * 0.5f) + ((((tmp_sub_neg_reg_h_3) * (tmp_sub_neg_reg_h_3)).sum() * 0.5f) * regulariz_param) + ((((tmp_sub_neg_reg_t_3) * (tmp_sub_neg_reg_t_3)).sum() * 0.5f) * regulariz_param);
        prob = ((euc-ptranse_margin).exp() + 1.f).inverse();
        gradient = -(0.f-prob()) * tmp_sub_neg_3;
        gradient_reg_h = -(0.f-prob()) * (tmp_sub_neg_3 + (tmp_sub_neg_reg_h_3 * regulariz_param)); //derivative of log(sig(z)). positive sign.
        gradient_reg_t = -(0.f-prob()) * (tmp_sub_neg_3 + (tmp_sub_neg_reg_t_3 * regulariz_param)); //derivative of log(sig(z)). negative sign.
        v_head += gradient_reg_h * (*lr); //h
        v_relation += gradient * (*lr); //neg_r
        neg_tail -= gradient_reg_t * (*lr); //t
      }
    }

  }


  void Compute(OpKernelContext* ctx) override {
    Tensor w_in = ctx->mutable_input(0, false);
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(w_in.shape()), errors::InvalidArgument("Must be a matrix"));

    Tensor w_out = ctx->mutable_input(9, false);
    OP_REQUIRES(ctx, w_in.shape() == w_out.shape(), errors::InvalidArgument("w_in.shape == w_out.shape"));

    const Tensor& examples = ctx->input(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(examples.shape()), errors::InvalidArgument("Must be a vector"));

    const Tensor& labels = ctx->input(2);

    const Tensor& learning_rate = ctx->input(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(learning_rate.shape()), errors::InvalidArgument("Must be a scalar"));

    auto Tw_in = w_in.matrix<float>();
    auto Tw_out = w_out.matrix<float>();
    auto Texamples = examples.flat<int32>();
    auto Tlabels = labels.flat<int32>();
    auto lr = learning_rate.scalar<float>()();
    const int64 vocab_size = w_in.dim_size(0);
    const int64 dims = w_in.dim_size(1);
    const int64 batch_size_ = examples.dim_size(0);

    unsigned long next_random = (long) rand();
    unsigned long sample = (long) rand();
    unsigned long sample_head = (long) rand();
    unsigned long sample_relation = (long) rand();
    unsigned long sample_tail = (long) rand();

    //KG
    Tensor w_in_ent = ctx->mutable_input(8, false);
    Tensor w_in_rel = ctx->mutable_input(4, false);

    int found_head_anchor = 0;
    int found_tail_anchor = 0;

    const Tensor& heads = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(heads.shape()), errors::InvalidArgument("Must be a vector"));
    const Tensor& relations = ctx->input(6);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(relations.shape()), errors::InvalidArgument("Must be a vector"));
    const Tensor& tails = ctx->input(7);
    OP_REQUIRES(ctx, heads.shape() == tails.shape(), errors::InvalidArgument("heads.shape == tails.shape"));

    auto Tw_in_ent = w_in_ent.matrix<float>();
    auto Tw_in_rel = w_in_rel.matrix<float>();
    const int64 entity_size = w_in_ent.dim_size(0);
    const int64 relation_size = w_in_rel.dim_size(0);
    auto Theads = heads.flat<int32>();
    auto Trelations = relations.flat<int32>();
    auto Ttails = tails.flat<int32>();

    Tensor eucl_buf_text(DT_FLOAT, TensorShape({}));
    auto euc_text= eucl_buf_text.scalar<float>();
    Tensor gradient_buf_text_w(DT_FLOAT, TensorShape({dims}));
    auto gradient_text_w = gradient_buf_text_w.flat<float>();
    Tensor gradient_buf_text_v(DT_FLOAT, TensorShape({dims}));
    auto gradient_text_v = gradient_buf_text_v.flat<float>();
    Tensor prob_buf_text(DT_FLOAT, TensorShape({}));
    auto prob_text = prob_buf_text.scalar<float>();

    //Text
    for (int64 i = 0; i < batch_size_; ++i) {
      const int32 example = Texamples(i);
      DCHECK(0 <= example && example < vocab_size) << example;
      const int32 label = Tlabels(i);
      DCHECK(0 <= label && label < vocab_size) << label;

      //Context anchor (target_anchor and context_anchor are kg indexes)
      // int target_anchor = anchor_text[example];
      int context_anchor = anchor_text[label];

      if (context_anchor==-1) { //no anchors
        update_embedding_text_eucl(&Tw_in, &Tw_out, example, label, &lr, num_samples_, vocab_size, sample, euc_text, gradient_text_w, &next_random, prob_text, 0);
      }
      else { //if we only find the anchor for context (v)
        update_embedding_text_eucl_with_reg(&Tw_in, &Tw_out, example, label, &lr, num_samples_, vocab_size, sample, euc_text, gradient_text_w, gradient_text_v, &next_random, prob_text, 0, context_anchor, &Tw_in_ent, 0, 1);
        // std::cout << get_value_string (id_word_tmp, example) << ", " << get_value_string (id_word_tmp, label) << ": label anchor-> " << get_value_string (id_entity_tmp, context_anchor) << "\n\n";
      }

      // if (target_anchor!=-1 and context_anchor==-1) { //if we only find the anchor for target (w)
      //   update_embedding_text_eucl_with_reg(&Tw_in, &Tw_out, example, label, &lr, num_samples_, vocab_size, sample, euc_text, gradient_text_w, gradient_text_v, &next_random, prob_text, 0, target_anchor, &Tw_in_ent, 1, 0); //with anchor
      //   // std::cout << get_value_string (id_word_tmp, example) << ", " << get_value_string (id_word_tmp, label) << ": label anchor-> " << get_value_string (id_entity_tmp, context_anchor) << "\n\n";
      // }
      //
      // if (target_anchor==-1 and context_anchor!=-1) { //if we only find the anchor for context (v)
      //   update_embedding_text_eucl_with_reg(&Tw_in, &Tw_out, example, label, &lr, num_samples_, vocab_size, sample, euc_text, gradient_text_w, gradient_text_v, &next_random, prob_text, 0, context_anchor, &Tw_in_ent, 0, 1);
      //   // std::cout << get_value_string (id_word_tmp, example) << ", " << get_value_string (id_word_tmp, label) << ": label anchor-> " << get_value_string (id_entity_tmp, context_anchor) << "\n\n";
      // }
      //
      // if (context_anchor!=-1 and target_anchor!=-1) { //if we find anchors for both taget and context (w and v)
      //   update_embedding_text_eucl_with_reg_with_w_and_v(&Tw_in, &Tw_out, example, label, &lr, num_samples_, vocab_size, sample, euc_text, gradient_text_w, gradient_text_v, &next_random, prob_text, 0, target_anchor, context_anchor, &Tw_in_ent);
      // }

    }


    //KG
    int head_anchor;
    int tail_anchor;

    Tensor eucl_buf_kg(DT_FLOAT, TensorShape({}));
    auto euc_kg= eucl_buf_kg.scalar<float>();
    Tensor gradient_buf_kg(DT_FLOAT, TensorShape({dims}));
    auto gradient_kg = gradient_buf_kg.flat<float>();
    Tensor gradient_buf_kg_reg_h(DT_FLOAT, TensorShape({dims}));
    auto gradient_kg_reg_h = gradient_buf_kg_reg_h.flat<float>();
    Tensor gradient_buf_kg_reg_t(DT_FLOAT, TensorShape({dims}));
    auto gradient_kg_reg_t = gradient_buf_kg_reg_t.flat<float>();
    Tensor prob_buf_kg(DT_FLOAT, TensorShape({}));
    auto prob_kg = prob_buf_kg.scalar<float>();

    for (int64 i = 0; i < batch_size_; ++i) {
      const int32 head = Theads(i);
      const int32 relation = Trelations(i);
      const int32 tail = Ttails(i);


      //Head anchor (head_anchor and tail_anchor are text indexes)
      head_anchor = anchor_kg[head];
      tail_anchor = anchor_kg[tail];

      if (head_anchor==-1 && tail_anchor==-1) { //no anchors
        update_embedding_kg_eucl(&Tw_in_ent, &Tw_in_rel, &Tw_in_ent, head, relation, tail, &lr, entity_size, relation_size, entity_size, &next_random, dims, euc_kg, gradient_kg, prob_kg, -1, -1);
      }

      else if (head_anchor!=-1 && tail_anchor!=-1) { //both anchors
        update_embedding_kg_eucl_with_reg_h_and_t(&Tw_in_ent, &Tw_in_rel, &Tw_in_ent, head, relation, tail, &lr, entity_size, &next_random, dims, euc_kg, gradient_kg, gradient_kg_reg_h, gradient_kg_reg_t, prob_kg, -1, head_anchor, tail_anchor, &Tw_in, relation_size);
        // std::cout << get_value_string (id_entity_tmp, head) << ", " << get_value_string (id_relation_tmp, relation) << ", " << get_value_string (id_entity_tmp, tail) << ": head and tail anchors-> " << get_value_string (id_word_tmp, head_anchor) << ", " << get_value_string (id_word_tmp, tail_anchor) << "\n";
      }

      else if (head_anchor!=-1 && tail_anchor==-1) { //if head_anchor exist in text dictionary
        update_embedding_kg_eucl_with_reg(&Tw_in_ent, &Tw_in_rel, &Tw_in_ent, head, relation, tail, &lr, entity_size, &next_random, dims, euc_kg, gradient_kg, gradient_kg_reg_h, prob_kg, 0, -1, head_anchor, &Tw_in, relation_size);
        // std::cout << get_value_string (id_entity_tmp, head) << ", " << get_value_string (id_relation_tmp, relation) << ", " << get_value_string (id_entity_tmp, tail) << ": head anchor-> " << get_value_string (id_word_tmp, head_anchor) << "\n\n";
      }

      else if (head_anchor==-1 && tail_anchor!=-1) { //if tail_anchor exist in text dictionary
        update_embedding_kg_eucl_with_reg(&Tw_in_ent, &Tw_in_rel, &Tw_in_ent, head, relation, tail, &lr, entity_size, &next_random, dims, euc_kg, gradient_kg,  gradient_kg_reg_t, prob_kg, -1, 0, tail_anchor, &Tw_in, relation_size);
        // std::cout << get_value_string (id_entity_tmp, head) << ", " << get_value_string (id_relation_tmp, relation) << ", " << get_value_string (id_entity_tmp, tail) << ": tail anchor-> " << get_value_string (id_word_tmp, tail_anchor) << "\n\n";
      }

    }

  }

 private:
  int32 num_samples_ = 0;
  GuardedPhiloxRandom base_;
};




REGISTER_KERNEL_BUILDER(Name("NegTrainWord2vec").Device(DEVICE_CPU), NegTrainWord2vecOp);

}
