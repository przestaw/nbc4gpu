//
// Created by przestaw on 11.12.2020.
//

#ifndef NBC4GPU_GPUCALCULATECLASSP_HPP
#define NBC4GPU_GPUCALCULATECLASSP_HPP

#include <error/exeception.h>
#include <project_defines.h>
DIAGNOSTIC_PUSH
#include <boost/compute/algorithm/accumulate.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/core.hpp>
#include <boost/compute/lambda.hpp>
DIAGNOSTIC_POP
#include <classifier/GPULearnClass.hpp>

namespace nbc4gpu {
  template <typename ValueType, typename PropabilityType>
  class GPUCalculateClassP {
  public:
    using Statistics = typename GPULearnClass<ValueType>::Statistics;
    using Record     = std::vector<ValueType>;

    /**
     *
     * @param record
     * @param statistics
     * @param queue
     */
    GPUCalculateClassP(Record record,
                       Statistics statistics,
                       boost::compute::command_queue &queue);

    /**
     * Calculates propability of record given in a constructor
     * @return propability
     */
    PropabilityType operator()();

  private:
    void calcPropability();

    Record record_;
    Statistics statistics_;
    boost::compute::command_queue &queue_;
    std::mutex calculationGuard;
    bool calculated;
  };

  template <typename ValueType, typename PropabilityType>
  GPUCalculateClassP<ValueType, PropabilityType>::GPUCalculateClassP(
      GPUCalculateClassP::Record record,
      GPUCalculateClassP::Statistics statistics,
      boost::compute::command_queue &queue)
      : record_(record), statistics_(statistics), queue_(queue),
        calculated(false) {
    if (record_.size() != statistics_.size()) {
      throw nbc4gpu::error::MismatchedSize(
          " record and dataset have diffrent sizes of columns");
    }
  }

  template <typename ValueType, typename PropabilityType>
  PropabilityType GPUCalculateClassP<ValueType, PropabilityType>::operator()() {
    // use double check pattern
    if (!calculated) {
      std::unique_lock lock(calculationGuard);
      if (!calculated) {
        calcPropability();
        calculated = true;
      }
    }
    return record_;
  }

  template <typename ValueType, typename PropabilityType>
  void GPUCalculateClassP<ValueType, PropabilityType>::calcPropability() {
    static const ValueType sqr2pi = sqrt(2 * 3.141);
    // exponent = exp(-((x-avg)^ 2 / (2 * variance )))
    // base =  (1 / (sqrt(2 * pi) * variance))
    // return e1*b1*e2*b2.... : e in exponent, b in base

    // 1. transfer record, avg & means to device
    boost::compute::vector<ValueType> recordVector(record_.size(), queue_.get_context());
    boost::compute::future<void> fRec = boost::compute::copy_async(
        record_.begin(), record_.end(), recordVector.begin(), queue_);

    std::vector<ValueType> avgs;
    std::vector<ValueType> variances;
    for (const auto iter : statistics_) {
      avgs.emplace_back(iter.first);
      variances.emplace_back(iter.second);
    }

    boost::compute::vector<ValueType> avgVector(avgs.size(), queue_.get_context());
    boost::compute::future<void> fAvg = boost::compute::copy_async(
        avgs.begin(), avgs.end(), avgVector.begin(), queue_);

    boost::compute::vector<ValueType> varianceVector(variances.size(), queue_.get_context());
    boost::compute::future<void> fVar = boost::compute::copy_async(
        variances.begin(), variances.end(), varianceVector.begin(), queue_);

    // using compute::transform
    // 1.1. calc - (x - avg)^2
    fRec.wait();
    fAvg.wait();
    boost::compute::transform(recordVector.begin(),
                              recordVector.end(),
                              avgVector.begin(),
                              recordVector.begin(),
                              boost::compute::lambda::_1 - boost::compute::lambda::_2,
                              queue_);
    boost::compute::transform(recordVector.begin(),
                              recordVector.end(),
                              recordVector.begin(),
                              1.0 / (boost::compute::lambda::_1 * 2.0),
                              queue_);

    // 1.2. calc  1 / (2*variance)
    // 1.3. calc  1.1*1.2 = exponent
    fVar.wait();
    boost::compute::transform(recordVector.begin(),
                              recordVector.end(),
                              varianceVector.begin(),
                              recordVector.begin(),
                              boost::compute::lambda::_1 / (boost::compute::lambda::_2 * 2.0),
                              queue_);
    // Note: RecordVector has (x-avg)^ 2 / (2 * variance ) values after operation above

    // 2. calc base

    // 2.1 calc variance*sqr2pi
    // 2.2 calc 1/(variance*sqr2pi)

    boost::compute::transform(varianceVector.begin(),
                              varianceVector.end(),
                              varianceVector.begin(),
                              1.0 / (boost::compute::lambda::_1 * sqr2pi),
                              queue_);

    // Note: varianceVector has 1/(variance*sqr2pi) values after operation above

    // 3. summarize
    // using accumulate or on CPU ??
    // 3.1 multiply all base and exponent values

    // 4. save propability
  }
} // namespace nbc4gpu

#endif // NBC4GPU_GPUCALCULATECLASSP_HPP
