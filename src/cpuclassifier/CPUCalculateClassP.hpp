//
// Created by mateusz
//

#ifndef NBC4CPU_CPUCALCULATECLASSP_HPP
#define NBC4CPU_CPUCALCULATECLASSP_HPP

#include "CPULearnClass.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
namespace nbc4cpu {
  template <typename ValueType> class CPUCalculateClassP {
  public:
    using Statistics = typename CPULearnClass<ValueType>::Statistics;
    using Record     = std::vector<ValueType>;

    /**
     * Constructor
     * @param statistics - statistics of a class values
     * @param classId - class identifer value
     * @param queue - device queue
     */
    CPUCalculateClassP(Statistics statistics, ValueType classId);

    /**
     * Calculates propability of record given in a constructor
     * @return propability
     */
    ValueType operator()(const Record& record);

    /**
     * Getter for classId
     * @return class identifier
     */
    inline const ValueType getClassId() const { return classId_; }

  private:
    ValueType calcPropability(const Record& record);
    ValueType classId_;
    Statistics statistics_;
  };

  template <typename ValueType>
  CPUCalculateClassP<ValueType>::CPUCalculateClassP(
      CPUCalculateClassP::Statistics statistics,
      ValueType classId)
      : classId_(classId), statistics_(statistics) {
    if (statistics_.empty()) {
      throw(" class Propability calculation");
    }
  }

  template <typename ValueType>
  ValueType CPUCalculateClassP<ValueType>::operator()(const Record& record) {
    if (record.size() != statistics_.size()) {
      throw(" record and dataset have diffrent sizes of columns");
    }
    return calcPropability(record);
  }

  template <typename ValueType>
  ValueType
  CPUCalculateClassP<ValueType>::calcPropability(const Record& record) {
    static const ValueType sqr2pi = sqrt(2 * 3.14159265);
    // exponent = exp(-((x-avg)^ 2 / (2 * variance )))
    // base =  (1 / (sqrt(2 * pi) * sqrt(variance)))
    // return e1*b1*e2*b2.... : e in exponent, b in base

    std::vector<ValueType> avgs;
    std::vector<ValueType> variances;
    for (const auto& iter : statistics_) {
      avgs.emplace_back(iter.first);
      variances.emplace_back(iter.second);
    }
    Record computing;
    // 1. Calculate exponent
    for (int i = 0; i < record.size(); i++)
      computing.emplace_back(record[i] - avgs[i]);
    // Note : x - avg
    for (auto& x : computing) x = -1 * (x * x);
    // Note : -(x - avg)^2

    for (int i = 0; i < computing.size(); i++)
      computing[i] = exp(computing[i] / (2.0 * variances[i]));

    for (int i = 0; i < computing.size(); i++)
      computing[i] = computing[i] / (sqrt(variances[i]) * sqr2pi);

    // 3. summarize
    ValueType propability = 1.0;
    for (auto& x : computing) propability = propability * x;

    // 4. return propability
    return propability;
  }
} // namespace nbc4cpu

#endif // NBC4CPU_CPUCALCULATECLASSP_HPP
