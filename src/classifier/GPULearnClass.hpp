//
// Created by przestaw on 11.12.2020.
//

#ifndef NBC4GPU_GPULEARNCLASS_HPP
#define NBC4GPU_GPULEARNCLASS_HPP

#include <project_defines.h>
#include <classifier/GPULearnColumn.hpp>
#include <mutex>

namespace nbc4gpu {
  template <typename ValueType> class GPULearnClass {
  public:
    using Learner = nbc4gpu::GPULearnColumn<ValueType>;
    using Statistics =
        std::vector<typename Learner::AvgStdDev>; //!< row containing statistic
                                                  //!< in each column

    using Column  = std::vector<ValueType>; //!< contains value for each row
    using Dataset = std::vector<Column>; //!< contains all rows for given class

    /**
     * Cosntructor
     * @param dataset - set of columns of same length representin records with
     * same class identifier value
     * @param classId - class identifer value
     * @param queue - device queue
     */
    GPULearnClass(Dataset dataset,
                  ValueType classId,
                  boost::compute::command_queue &queue);

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
    bool calculated;
    std::mutex calculationGuard;
    Statistics record_;
    const ValueType classId_;
    boost::compute::command_queue &queue_;
    std::mutex guard;
    size_t unlearned;
  };

  template <typename ValueType>
  GPULearnClass<ValueType>::GPULearnClass(GPULearnClass::Dataset dataset,
                                          ValueType classId,
                                          boost::compute::command_queue &queue)
      : dataset_(std::move(dataset)), calculated(false), classId_(classId),
        queue_(queue), unlearned(0) {}

  template <typename ValueType>
  typename GPULearnClass<ValueType>::Statistics
  GPULearnClass<ValueType>::operator()() {
    // use double check pattern
    if (!calculated) {
      std::unique_lock lock(calculationGuard);
      if (!calculated) {
        // todo - worker threads?
        workerThread();
        calculated = true;
      }
    }
    return record_;
  }

  template <typename ValueType> void GPULearnClass<ValueType>::workerThread() {
    size_t pivot = 0;
    while ((pivot = getUnlearned()) < dataset_.size()) {
      Learner learner(dataset_[pivot], queue_);
      record_[pivot] = learner();
    }
  }

  template <typename ValueType>
  size_t GPULearnClass<ValueType>::getUnlearned() {
    std::unique_lock lock(guard);
    return unlearned++;
  }

} // namespace nbc4gpu

#endif // NBC4GPU_GPULEARNCLASS_HPP
