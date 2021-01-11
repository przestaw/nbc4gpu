//
// Created by przestaw on 11.12.2020.
//

#ifndef NBC4CPU_CPULEARNCLASS_HPP
#define NBC4CPU_CPULEARNCLASS_HPP

#include "CPULearnColumn.hpp"

namespace nbc4cpu {
  template <typename ValueType> class CPULearnClass {
  public:
    using Learner = nbc4cpu::CPULearnColumn<ValueType>;
    using Statistics =
        std::vector<typename Learner::AvgAndVariance>; //!< row containing
                                                       //!< statistic in each
                                                       //!< column

    using Column  = std::vector<ValueType>; //!< contains value for each row
    using Dataset = std::vector<Column>; //!< contains all rows for given class

    /**
     * Cosntructor
     * @param dataset - set of columns of same length representin records with
     * same class identifier value
     * @param classId - class identifer value
     * @param queue - device queue
     */
    CPULearnClass(Dataset dataset,
                  ValueType classId);

    /**
     * Learns statistics for each column in dataset
     * @return vector of statistics
     */
    Statistics operator()();

    /**
     * Getter for classId
     * @return class identifier
     */
    inline const ValueType getClassId() const { return classId_; }

  private:
    void workerThread();
    size_t getUnlearned();

    Dataset dataset_;
    const ValueType classId_;

    size_t unlearned;

    bool calculated;
    Statistics record_;
  };

  template <typename ValueType>
  CPULearnClass<ValueType>::CPULearnClass(CPULearnClass::Dataset dataset,
                                          ValueType classId)
      : dataset_(std::move(dataset)), classId_(classId),
        unlearned(0), calculated(false), record_(dataset_.size()) {}

  template <typename ValueType>
  typename CPULearnClass<ValueType>::Statistics
  CPULearnClass<ValueType>::operator()() {
      if (!calculated) {
        // todo - worker threads?
        workerThread();
        calculated = true;
      }
    return record_;
  }

  template <typename ValueType> void CPULearnClass<ValueType>::workerThread() {
    size_t pivot = 0;
    while ((pivot = getUnlearned()) < dataset_.size()) {
      Learner learner(dataset_[pivot]);
      record_[pivot] = learner();
    }
  }

  template <typename ValueType>
  size_t CPULearnClass<ValueType>::getUnlearned() {
    return unlearned++;
  }

} // namespace nbc4cpu

#endif // NBC4CPU_CPULEARNCLASS_HPP
