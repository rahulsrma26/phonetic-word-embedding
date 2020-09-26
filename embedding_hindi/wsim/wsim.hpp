#include <cmath>
#include <algorithm>
#include <bitset>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
// UTF-8
#include <locale>
#include <utility>
#include <codecvt>

#define DEBUG 1

namespace wsim {

using FeatureType = uint32_t;
using Phones = std::vector<FeatureType>;
using Phone2Feature = std::unordered_map<wchar_t, FeatureType>;
using Feature2Phone = std::unordered_map<FeatureType, wchar_t>;

const int BIGRAM = 1;
const int INSERT_BEG_END = 2;
const int VOWEL_BUFF = 4;
const int RHYME_BUFF = 8;

class Dictionary {
  public:
    Dictionary(const std::string &mapping, const std::string &words);
    ~Dictionary();

    std::string getDictionaryPath() const;
    std::string getMappingPath() const;

    int size() const;
    int getIndex(const std::string &word) const;
    std::string getWord(int idx) const;

    float score(int w1, int w2, int flags, float nonDiagonalPenalty) const;

    std::pair<std::pair<std::vector<int>, std::vector<int>>, std::vector<float>>
    randomScores(int n, int flags, float nonDiagonalPenalty) const;

    // float checkEmbeddings(const std::vector<std::vector<float>>&, int flags, float nonDiagonalPenalty) const;

    std::vector<std::pair<float, std::string>>
    topSimilar(const std::string &word, int k, int flags,
               float nonDiagonalPenalty) const;

  protected:
    std::string dictionary_path_m;
    std::string feature_path_m;
    Feature2Phone feature2phone_m;
    Phone2Feature phone2feature_m;
    std::unordered_map<std::wstring, int> words2idx_m;
    std::vector<std::wstring> dictionary_m;
    std::vector<Phones> phones_m;

    float score_unigram(int w1, int w2, int flags,
                        float nonDiagonalPenalty) const;
    float score_bigram(int w1, int w2, int flags,
                       float nonDiagonalPenalty) const;

    std::tuple<wchar_t,std::vector<FeatureType>> check_word(const std::wstring& word) const;
    Phones add_beg_end(const Phones &word) const;
};
} // namespace wsim
