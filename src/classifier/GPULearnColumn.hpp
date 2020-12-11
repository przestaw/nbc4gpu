//
// Created by przestaw on 04.12.2020.
//

#ifndef NBC4GPU_GPULEARNCOLUMN_HPP
#define NBC4GPU_GPULEARNCOLUMN_HPP

#include <project_defines.h>
DIAGNOSTIC_PUSH
#include <boost/compute/algorithm/accumulate.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/core.hpp>
#include <boost/compute/lambda.hpp>
DIAGNOSTIC_POP
#include <mutex>


namespace nbc4gpu {
  template <typename ValueType> class GPULearnColumn {
  public:
    using AvgAndVariance = std::pair<ValueType, ValueType>;

    /**
     * Constructor
     * @param col column of values
     * @param queue device column
     */
    GPULearnColumn(std::vector<ValueType> &col,
                   boost::compute::command_queue &queue);

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
    boost::compute::command_queue &queue_;
    std::mutex calculationGuard;
  };

  template <typename ValueType>
  GPULearnColumn<ValueType>::GPULearnColumn(
      std::vector<ValueType> &col,
      boost::compute::command_queue &queue)
      : resSet_(false), col_(col), queue_(queue) {}

  template <typename ValueType>
  typename GPULearnColumn<ValueType>::AvgAndVariance
  GPULearnColumn<ValueType>::operator()() {
    if (!resSet_) {
      std::unique_lock lock(calculationGuard);
      if (!resSet_) {
        result_ = learn();
        resSet_ = true;
      }
    }
    return result_;
  }

  template <typename ValueType>
  typename GPULearnColumn<ValueType>::AvgAndVariance
  GPULearnColumn<ValueType>::learn() {
    // transfer the values to the device
    boost::compute::vector<double> avgVector(col_.size(), queue_.get_context());
    boost::compute::future<void> fAvg = boost::compute::copy_async(
        col_.begin(), col_.end(), avgVector.begin(), queue_);
    // reduce is destructive, need to copy twice
    boost::compute::vector<double> varianceVector(col_.size(),
                                                queue_.get_context());
    boost::compute::future<void> fStdDev = boost::compute::copy_async(
        col_.begin(), col_.end(), varianceVector.begin(), queue_);

    double avg = 0;
    fAvg.wait(); // Wait for data to be copied

    boost::compute::reduce(avgVector.begin(), avgVector.end(), &avg, queue_);
    avg = avg / static_cast<double>(col_.size());

    fStdDev.wait(); // Wait for data to be copied
    boost::compute::transform(varianceVector.begin(),
                              varianceVector.end(),
                              varianceVector.begin(),
                              boost::compute::lambda::_1 - avg,
                              queue_);

    boost::compute::transform(varianceVector.begin(),
                              varianceVector.end(),
                              varianceVector.begin(),
                              boost::compute::lambda::_1
                                  * boost::compute::lambda::_1,
                              queue_);

    double variance = 0;
    boost::compute::reduce(
        varianceVector.begin(), varianceVector.end(), &variance, queue_);
    variance = variance * (1 / static_cast<double>(col_.size() - 1));

    return std::make_pair(avg, variance);
  }
} // namespace nbc4gpu

#endif // NBC4GPU_GPULEARNCOLUMN_HPP
