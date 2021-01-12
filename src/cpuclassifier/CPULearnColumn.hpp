//
// Created by mateusz
//

#ifndef NBC4CPU_CPULEARNCOLUMN_HPP
#define NBC4CPU_CPULEARNCOLUMN_HPP
#include <vector>

namespace nbc4cpu {
  template <typename ValueType> class CPULearnColumn {
  public:
    using AvgAndVariance = std::pair<ValueType, ValueType>;

    /**
     * Constructor
     * @param col column of values
     */
    CPULearnColumn(std::vector<ValueType> &col);

    /**
     * Calculates or returns average and standard deviation
     * @return average and standard deviation
     */
    AvgAndVariance operator()();

  private:
    AvgAndVariance learn();

    bool resSet_;
    AvgAndVariance result_;
    std::vector<ValueType> &col_;
  };

  template <typename ValueType>
  CPULearnColumn<ValueType>::CPULearnColumn(std::vector<ValueType> &col)
      : resSet_(false), col_(col) {}

  template <typename ValueType>
  typename CPULearnColumn<ValueType>::AvgAndVariance
  CPULearnColumn<ValueType>::operator()() {
    if (!resSet_) {
      result_ = learn();
      resSet_ = true;
    }
    return result_;
  }

  template <typename ValueType>
  typename CPULearnColumn<ValueType>::AvgAndVariance
  CPULearnColumn<ValueType>::learn() {
    if (col_.empty()) {
      throw(" column summary");
    }
    ValueType avg = 0.0;
    for (auto &n : col_) avg += n;
    avg = avg / static_cast<ValueType>(col_.size());

    ValueType variance = 0.0;
    for (auto &n : col_) variance += ((n - avg) * (n - avg));
    variance = variance * (1 / static_cast<ValueType>(col_.size() - 1));

    return std::make_pair(avg, variance);
  }
} // namespace nbc4cpu

#endif // NBC4CPU_CPULEARNCOLUMN_HPP
