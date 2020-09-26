#include "wsim.hpp"

namespace wsim {

constexpr const char* LOCALE = "en_US.UTF-8";

static const auto SET_INIT = []()
{
    std::wcout.imbue(std::locale(LOCALE));
    return 0;
}();

std::wstring s2ws(const std::string& str)
{
    using convert_typeX = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_typeX, wchar_t> converterX;

    return converterX.from_bytes(str);
}

std::string ws2s(const std::wstring& wstr)
{
    using convert_typeX = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_typeX, wchar_t> converterX;

    return converterX.to_bytes(wstr);
}

const FeatureType START_BIT = 0;
const FeatureType END_BIT = 1;
const FeatureType VOWEL_BIT = 2;
const FeatureType VOWEL_FLAG = 1 << VOWEL_BIT;

// std::map<std::string, std::vector<std::string>> phone_feature_mapping;

Phone2Feature loadFeatures(const std::string& path){
    std::unordered_map<std::wstring, int> features = {
        {L"beg", START_BIT}, {L"end", END_BIT}, {L"vwl", VOWEL_BIT}};
    Phone2Feature feature_map;
    std::wifstream wif(path);
    wif.imbue(std::locale(LOCALE));

    for (std::wstring line; std::getline(wif, line);) {
        if (!line.size() || line[0] == ';')
            continue;
        std::wstringstream iss(line);
        std::wstring first_word;
        iss >> first_word;
        if (first_word.size() != 1){
            std::wcout << L"WARNING!!! Problem with phone '" << first_word
                << "'. Expected size 1 found " << first_word.size() << ". Skipping\n";
            continue;
        }
        auto phone = first_word.front();
        if (feature_map.find(phone) != feature_map.end()){
            std::wcout << L"WARNING!!! More than one entries found for '"
                << phone << L"' Ignoring this entry\n";
            continue;
        }
        FeatureType feature_bits = 0;
        std::wstring word;
        while (iss >> word){
            if(features.find(word) == features.end())
                features.insert({word, (int)features.size()});
            feature_bits |= (1 << features[word]);
        }
        feature_map[phone] = feature_bits;
    }
#ifdef DEBUG
    std::cout << "Total phones = " << feature_map.size() - 2 << " + 2" << '\n';
    std::cout << "Total features = " << features.size() << '\n';
    // for(auto p: feature_map){
    //     std::wcout << p.first << ' ' << std::bitset<32>(p.second) << '\n';
    // }
#endif
    return feature_map;
}

template <typename T, typename S>
auto getRevMap(const std::unordered_map<T, S>& m){
    std::unordered_map<S, T> r;
    for (const auto &pair : m)
        r.insert({pair.second, pair.first});
    return r;
}

inline uint32_t bitCount(const uint32_t v) {
#ifdef _MSC_VER
    return _mm_popcnt_u32(v);
#else
    return __builtin_popcount(v);
#endif
}

inline float getSimilarity(const FeatureType f1, const FeatureType f2) {
    return bitCount(f1 & f2) / (float)(bitCount(f1 | f2));
}

inline float getSimilarity(const FeatureType a1, const FeatureType a2,
                           const FeatureType b1, const FeatureType b2,
                           const bool vowel_buff) {
    const FeatureType w1 = a1 | a2;
    const FeatureType w2 = b1 | b2;
    float score = bitCount(w1 & w2) / (float)(bitCount(w1 | w2));
    if (vowel_buff)
        score = ((a2 & VOWEL_FLAG) && (b2 & VOWEL_FLAG) && a2 == b2)
                    ? sqrt(score)
                    : (score * score);
    return score;
}

Phones Dictionary::add_beg_end(const Phones &word) const {
    Phones list;
    list.push_back(phone2feature_m.at('^'));
    for (const auto phone : word)
        list.push_back(phone);
    // std::wcout << "$" << phone2feature_m.at('$') << std::endl;
    list.push_back(phone2feature_m.at('$'));
    return list;
}

std::tuple<wchar_t, Phones> Dictionary::check_word(const std::wstring &word) const{
    Phones phones;
    for(wchar_t ch: word){
        auto result = phone2feature_m.find(ch);
        if(result == phone2feature_m.end())
            return {ch, {}};
        phones.push_back(result->second);
    }
    return {'\0', phones};
}

Dictionary::Dictionary(const std::string &mapfile, const std::string &dictfile)
    : dictionary_path_m(dictfile), feature_path_m(mapfile) {
#ifdef DEBUG
    std::cout << "Phone feature map path = " << feature_path_m << '\n';
    std::cout << "Dictionary path = " << dictionary_path_m << '\n';
#endif
    phone2feature_m = loadFeatures(feature_path_m);
    feature2phone_m = getRevMap(phone2feature_m);

    std::wifstream fin(dictionary_path_m);
    fin.imbue(std::locale(LOCALE));
    for (std::wstring line; std::getline(fin, line);) {
        if (!line.size() || line[0] == ';')
            continue;
        std::wstringstream iss(line);
        std::wstring word;
        iss >> word;
        auto result = words2idx_m.find(word);
        if (result != words2idx_m.end()) {
            std::wcout << L"WARNING!!! dictionary has repeated word '"
                << word << L"'. Ignoring this entry\n";
        }
        else{
            auto [failed, phones] = check_word(word);
            if(failed != '\0'){
                std::wcout << L"WARNING!!! word '" << word << "' is using character '" <<
                    failed << "' which is not defined in mapping. Skipping.\n";
            }
            else{
                words2idx_m.insert({word, (int)words2idx_m.size()});
                phones_m.emplace_back(phones);
                dictionary_m.push_back(word);
            }
        }
    }
#ifdef DEBUG
    std::cout << "Dictionary size = " << dictionary_m.size() << '\n';
#endif
    // const auto a1 = getFeature("W");
    // std::cout << "W = " << std::bitset<31>(a1) << '\n';
    // const auto a2 = getFeature("AH");
    // std::cout << "AH = " << std::bitset<31>(a2) << '\n';
    // const auto b1 = FEATURES_MAP.at("^");
    // std::cout << "^ = " << std::bitset<31>(b1) << '\n';
    // const auto b2 = getFeature("AH");
    // std::cout << "Similarity(W+AH, ^+AH) = " << getSimilarity(a1, a2, b1, b2,
    // true) << '\n';
    // auto cap = L'^';
    // auto ha = L'ह';
    // auto la = L'ल';
    // auto halant = L"[्"[1];
    // auto cap_f = phone2feature_m.at(cap);
    // auto ha_f = phone2feature_m.at(ha);
    // auto la_f = phone2feature_m.at(la);
    // auto halant_f = phone2feature_m.at(halant);
    // std::wcout << ha << ' ' << la << ' ' << getSimilarity(ha_f, la_f) << std::endl;
    // std::wcout << ha << ' ' << halant << ' ' << getSimilarity(ha_f, halant_f) << std::endl;
    // std::cout << "ह ल" << getSimilarity(a1, a2, b1, b2, true) << std::endl;
}

Dictionary::~Dictionary() {
#ifdef DEBUG
    std::cout << "Dictionary " << dictionary_path_m << " Unloaded \n";
#endif
}

std::string Dictionary::getDictionaryPath() const { return dictionary_path_m; }

std::string Dictionary::getMappingPath() const { return feature_path_m; }

int Dictionary::size() const { return dictionary_m.size(); }

int Dictionary::getIndex(const std::string &word_) const {
    auto word = s2ws(word_);
    auto result = words2idx_m.find(word);
    if (result == words2idx_m.end())
        return -1;
    return result->second;
}

std::string Dictionary::getWord(int idx) const { return ws2s(dictionary_m[idx]); }

float Dictionary::score_unigram(int w1, int w2, int flags,
                                float nonDiagonalPenalty) const {
    auto p1 = (flags & INSERT_BEG_END) ? add_beg_end(phones_m[w1]) : phones_m[w1];
    auto p2 = (flags & INSERT_BEG_END) ? add_beg_end(phones_m[w2]) : phones_m[w2];
    const int n1 = p1.size();
    const int n2 = p2.size();
    std::vector<float> even(n1), odd(n1);
    even[0] = getSimilarity(p2[0], p1[0]);
    for (int i = 1; i < n1; i++)
        even[i] = getSimilarity(p2[0], p1[i]) + even[i - 1];
    // for (auto &v : even)
    //     printf("%1.5f  ", v);
    // std::cout << '\n';
    for (int i = 1; i < n2; i++) {
        auto &cur = (i & 1) ? odd : even;
        auto &prev = (i & 1) ? even : odd;
        // nonDiagonalPenalty is avoided because of no choice
        cur[0] = getSimilarity(p2[i], p1[0]) + prev[0];
        for (int j = 1; j < n1; j++) {
            const auto sim = getSimilarity(p2[i], p1[j]);
            if (p2[i] == p1[j]) {
                cur[j] = sim + prev[j - 1];
            } else {
                if (prev[j] > cur[j - 1])
                    cur[j] = sim * nonDiagonalPenalty + prev[j];
                else
                    cur[j] = sim * nonDiagonalPenalty + cur[j - 1];
            }
        }
        // for (auto &v : cur)
        //     printf("%1.5f  ", v);
        // std::cout << '\n';
    }
    const int base = (flags & INSERT_BEG_END) ? 2 : 0;
    float best = (((n2 & 1) ? even : odd).back() - base);
    return best / (float)(std::max(n1, n2) - base);
}

float Dictionary::score_bigram(int w1, int w2, int flags,
                               float nonDiagonalPenalty) const {
    auto p1 = (flags & INSERT_BEG_END) ? add_beg_end(phones_m[w1]) : phones_m[w1];
    auto p2 = (flags & INSERT_BEG_END) ? add_beg_end(phones_m[w2]) : phones_m[w2];
    const bool vowel_buff = (flags & VOWEL_BUFF);
    const int n1 = p1.size() - 1;
    const int n2 = p2.size() - 1;
    if (n1 < 1 || n2 < 1)
        return 0;

    // for (int i = 0; i <= n1; i++)
    //     std::wcout << "[" << feature2phone_m.at(p1[i]) << "] ";
    // std::wcout << std::endl;
    // for (int i = 0; i <= n2; i++)
    //     std::wcout << "[" << feature2phone_m.at(p2[i]) << "] ";
    // std::wcout << std::endl;
    // for (int i = 0; i < n2; i++){
    //     for (int j = 0; j < n1; j++)
    //         printf("%1.5f  ", getSimilarity(p2[i], p2[i+1], p1[j], p1[j+1], vowel_buff));
    //     std::cout << '\n';
    // }
    // std::cout << "----------------\n";

    std::vector<float> even(n1), odd(n1);
    even[0] = getSimilarity(p2[0], p2[1], p1[0], p1[1], vowel_buff);
    for (int i = 1; i < n1; i++)
        even[i] = getSimilarity(p2[0], p2[1], p1[i], p1[i + 1], vowel_buff) +
                  even[i - 1];
    // for (auto &v : even)
    //     printf("%1.5f  ", v);
    // std::cout << '\n';
    for (int i = 1; i < n2; i++) {
        auto &cur = (i & 1) ? odd : even;
        auto &prev = (i & 1) ? even : odd;
        // nonDiagonalPenalty is avoided because of no choice
        cur[0] =
            getSimilarity(p2[i], p2[i + 1], p1[0], p1[1], vowel_buff) + prev[0];
        for (int j = 1; j < n1; j++) {
            const auto sim =
                getSimilarity(p2[i], p2[i + 1], p1[j], p1[j + 1], vowel_buff);
            if (sim < 1) {
                if (prev[j] > cur[j - 1])
                    cur[j] = sim * nonDiagonalPenalty + prev[j];
                else
                    cur[j] = sim * nonDiagonalPenalty + cur[j - 1];
            } else {
                cur[j] = sim + prev[j - 1];
            }
        }
        // for (auto &v : cur)
        //     printf("%1.5f  ", v);
        // std::cout << '\n';
    }
    float score = ((n2 & 1) ? even : odd).back() / std::max(n1, n2);
    return score;
}

float Dictionary::score(int w1, int w2, int flags,
                        float nonDiagonalPenalty) const {
    if (flags & BIGRAM)
        return score_bigram(w1, w2, flags, nonDiagonalPenalty);
    return score_unigram(w1, w2, flags, nonDiagonalPenalty);
}

// const std::vector<int> dataset = {
//     92239,  106825, 14871,  26831,  128685, 52332,  70035,  34429,  111397,
//     12033,  88873,  49269,  26144,  43173,  12361,  91563,  114005, 92523,
//     113310, 93846,  49401,  133852, 133853, 92237,  133854, 111008, 110591,
//     127833, 103577, 63994,  37558,  118189, 118276, 33874,  118652, 121536,
//     77476,  58064,  41187,  20796,  16762,  70539,  112328, 120747, 120107,
//     79105,  60812,  105141, 108262, 109860, 92006,  55044,  131504, 109872,
//     121539, 70753,  97089,  97405,  133858, 15812,  26092,  130766, 76996,
//     23037,  51871,  16795,  28113,  130408, 16241,  119331, 130885, 129168,
//     131044, 133855, 133856, 128217, 116742, 133857};

std::pair<std::pair<std::vector<int>, std::vector<int>>, std::vector<float>>
Dictionary::randomScores(int n, int flags, float nonDiagonalPenalty) const {
    std::vector<int> i1(n), i2(n);
    std::vector<float> s(n);
    // i1[0] = dataset[rand() % dataset.size()];
    // i2[0] = dataset[rand() % dataset.size()];
    // s[0] = score(i1[0], i2[0], flags, nonDiagonalPenalty);

    for (int i = 0; i < n; i++) {
        i1[i] = rand() % dictionary_m.size();
        i2[i] = rand() % dictionary_m.size();
        s[i] = score(i1[i], i2[i], flags, nonDiagonalPenalty);
    }
    return {{i1, i2}, s};
}

std::vector<std::pair<float, std::string>>
Dictionary::topSimilar(const std::string &word, int k, int flags,
                       float nonDiagonalPenalty) const {
    // std::cout << "flags: " << flags << std::endl;
    const auto idx = getIndex(word);
    std::vector<std::pair<float, std::string>> vals;
    vals.reserve(words2idx_m.size());
    for (auto &wi_pair : words2idx_m)
        vals.push_back({score(idx, wi_pair.second, flags, nonDiagonalPenalty),
                        ws2s(wi_pair.first)});

    std::sort(vals.begin(), vals.end(),
              [](const auto &e1, const auto &e2) {
                  return e1.first > e2.first;
              });

    if (0 < k && k < (int)vals.size())
        vals.resize(k);

    return vals;
}
} // namespace wsim

// int main(){
//     using namespace wsim;
//     // wsim::loadFeatures("../hindi/featuremap.txt");
//     Dictionary d("../hindi/featuremap.txt", "../hindi/hindi_dict.txt");
//     // auto w1 = L"हल्दी";
//     // auto i1 = d.getIndex(w1);
//     // auto w2 = L"हल्की";
//     // auto i2 = d.getIndex(w2);
//     // auto s = d.score(i1, i2, INSERT_BEG_END | BIGRAM | VOWEL_BUFF, 0.5);
//     // std::cout << "Score = " << s << "\n";
//     for(auto res: d.topSimilar(L"सहारा", 10, INSERT_BEG_END | BIGRAM | VOWEL_BUFF, 0.4)){
//         std::wcout << res.first << " | " << res.second << '\n';
//     }
// }
