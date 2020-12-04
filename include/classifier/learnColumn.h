//
// Created by przestaw on 04.12.2020.
//

#ifndef NBC4GPU_LEARNCOLUMN_H
#define NBC4GPU_LEARNCOLUMN_H

#include <project_defines.h>
DIAGNOSTIC_PUSH
#include <boost/compute/algorithm/accumulate.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/core.hpp>
#include <boost/compute/lambda.hpp>
DIAGNOSTIC_POP

namespace nbc4gpu {
  template <typename ValueType> class GPULearnColumn {
  public:
    using AvgStdDev = std::pair<ValueType, ValueType>;

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
    AvgStdDev operator()();

  private:
    bool resSet_;
    AvgStdDev result_;
    std::vector<ValueType> &col_;
    boost::compute::command_queue &queue_;

    AvgStdDev learn();
  };

  template <typename ValueType>
  GPULearnColumn<ValueType>::GPULearnColumn(
      std::vector<ValueType> &col,
      boost::compute::command_queue &queue)
      : resSet_(false), col_(col), queue_(queue) {}

  template <typename ValueType>
  typename GPULearnColumn<ValueType>::AvgStdDev
  GPULearnColumn<ValueType>::operator()() {
    if (!resSet_) {
      result_ = learn();
      resSet_ = true;
    }
    return result_;
  }

  template <typename ValueType>
  typename GPULearnColumn<ValueType>::AvgStdDev
  GPULearnColumn<ValueType>::learn() {
    // transfer the values to the device
    boost::compute::vector<double> avgVector(col_.size(), queue_.get_context());
    boost::compute::future<void> fAvg = boost::compute::copy_async(
        col_.begin(), col_.end(), avgVector.begin(), queue_);
    // reduce is destructive, need to copy twice
    boost::compute::vector<double> stdDevVector(col_.size(),
                                                queue_.get_context());
    boost::compute::future<void> fStdDev = boost::compute::copy_async(
        col_.begin(), col_.end(), stdDevVector.begin(), queue_);

    double avg = 0;
    fAvg.wait(); // Wait for data to be copied

    boost::compute::reduce(avgVector.begin(), avgVector.end(), &avg, queue_);
    avg = avg / static_cast<double>(col_.size());

    fStdDev.wait(); // Wait for data to be copied
    boost::compute::transform(stdDevVector.begin(),
                              stdDevVector.end(),
                              stdDevVector.begin(),
                              boost::compute::lambda::_1 - avg,
                              queue_);

    boost::compute::transform(stdDevVector.begin(),
                              stdDevVector.end(),
                              stdDevVector.begin(),
                              boost::compute::lambda::_1
                                  * boost::compute::lambda::_1,
                              queue_);

    double stdDev = 0;
    boost::compute::reduce(
        stdDevVector.begin(), stdDevVector.end(), &stdDev, queue_);
    stdDev = sqrt(stdDev * (1 / static_cast<double>(col_.size() - 1)));

    return std::make_pair(avg, stdDev);
  }
} // namespace nbc4gpu

#endif // NBC4GPU_LEARNCOLUMN_H
