#if !defined(AMT_MODEL_DECISION_TREE_SPLITTER_HPP)
#define AMT_MODEL_DECISION_TREE_SPLITTER_HPP

#include <dataframe.hpp>

namespace amt::splitter {

struct ID3 {

  constexpr ID3() = default;
  constexpr ID3(ID3 const &other) = default;
  constexpr ID3(ID3 &&other) = default;
  constexpr ID3 &operator=(ID3 const &other) = default;
  constexpr ID3 &operator=(ID3 &&other) = default;
  constexpr ~ID3() = default;

  constexpr ID3(Frame auto const &Y) : m_E(entropy_cal(Y[0])) {}

  constexpr ID3(Series auto const &Y) : m_E(entropy_cal(Y)) {}

  constexpr std::size_t
  operator()(Frame auto const &XY,
             std::unordered_set<std::size_t> const &blocked_ids) const {
    if (XY.empty()) {
      throw std::length_error(ERR_CSTR("amt::splitter::ID3::operator()(Frame "
                                       "auto const&) : frame is empty"));
    }

    double max{};
    std::size_t idx{};
    double total = static_cast<double>(XY.rows());
    for (auto i = 0ul; i < XY.cols() - 1u; ++i) {
      if (!blocked_ids.count(i)) {
        auto f = freq(XY[i]);
        double curr_et = m_E;
        for (auto const &[k, v] : f) {
          auto nf = filter[i](XY, equal(k));
          auto et = entropy_cal(nf.back());
          auto r = static_cast<double>(v) / total;
          curr_et -= r * et;
        }
        if (curr_et > max) {
          idx = i;
          max = curr_et;
        }
      }
    }

    return idx;
  }

  constexpr double total_entropy() const noexcept { return m_E; }

private:
  constexpr double entropy_cal(Series auto const &s) const {
    double ret{};
    auto f = freq(s);
    auto t = static_cast<double>(s.size());
    for (auto const &[_, v] : f) {
      auto r = static_cast<double>(v) / t;
      if (std::isnormal(r))
        ret -= r * std::log2(r);
    }
    return ret;
  }

private:
  double m_E{};
};

} // namespace amt::splitter

#endif // AMT_MODEL_DECISION_TREE_SPLITTER_HPP
