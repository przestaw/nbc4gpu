//
// Created by przestaw on 11.12.2020.
//

#ifndef NBC4GPU_GPUCALCULATECLASSP_HPP
#define NBC4GPU_GPUCALCULATECLASSP_HPP

#include <Defines.h>
#include <error/Execeptions.h>
DIAGNOSTIC_PUSH
#include <boost/compute/algorithm/accumulate.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/core.hpp>
#include <boost/compute/lambda.hpp>
DIAGNOSTIC_POP
#include <classifier/GPULearnClass.hpp>

namespace nbc4gpu {
  template <typename ValueType> class GPUCalculateClassP {
  public:
    using Statistics = typename GPULearnClass<ValueType>::Statistics;
    using Record     = std::vector<ValueType>;

    /**
     * Constructor
     * @param statistics - statistics of a class values
     * @param classId - class identifer value
     * @param queue - device queue
     */
    GPUCalculateClassP(Statistics statistics,
                       ValueType classId,
                       boost::compute::command_queue& queue);

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
    boost::compute::command_queue& queue_;
  };

  template <typename ValueType>
  GPUCalculateClassP<ValueType>::GPUCalculateClassP(
      GPUCalculateClassP::Statistics statistics,
      ValueType classId,
      boost::compute::command_queue& queue)
      : classId_(classId), statistics_(statistics), queue_(queue) {
    if (statistics_.empty()) {
      throw nbc4gpu::error::ZeroValuesProvided(
          " class Propability calculation");
    }
  }

  template <typename ValueType>
  ValueType GPUCalculateClassP<ValueType>::operator()(const Record& record) {
    if (record.size() != statistics_.size()) {
      throw nbc4gpu::error::MismatchedSize(
          " record and dataset have diffrent sizes of columns");
    }
    return calcPropability(record);
  }

  template <typename ValueType>
  ValueType
  GPUCalculateClassP<ValueType>::calcPropability(const Record& record) {
    static const ValueType sqr2pi = sqrt(2 * 3.14159265);
    // exponent = exp(-((x-avg)^ 2 / (2 * variance )))
    // base =  (1 / (sqrt(2 * pi) * sqrt(variance)))
    // return e1*b1*e2*b2.... : e in exponent, b in base

    // 1. transfer record, avg & means to device
    boost::compute::vector<ValueType> recordVector(record.size(),
                                                   queue_.get_context());
    boost::compute::future<void> fRec = boost::compute::copy_async(
        record.begin(), record.end(), recordVector.begin(), queue_);

    std::vector<ValueType> avgs;
    std::vector<ValueType> variances;
    for (const auto& iter : statistics_) {
      avgs.emplace_back(iter.first);
      variances.emplace_back(iter.second);
    }

    boost::compute::vector<ValueType> avgVector(avgs.size(),
                                                queue_.get_context());
    boost::compute::future<void> fAvg = boost::compute::copy_async(
        avgs.begin(), avgs.end(), avgVector.begin(), queue_);

    boost::compute::vector<ValueType> varianceVector(variances.size(),
                                                     queue_.get_context());
    boost::compute::future<void> fVar = boost::compute::copy_async(
        variances.begin(), variances.end(), varianceVector.begin(), queue_);

    // 1. Calculate exponent
    fRec.wait();
    fAvg.wait();
    boost::compute::transform(recordVector.begin(),
                              recordVector.end(),
                              avgVector.begin(),
                              recordVector.begin(),
                              boost::compute::lambda::_1
                                  - boost::compute::lambda::_2,
                              queue_);
    // Note : x - avg

    boost::compute::transform(
        recordVector.begin(),
        recordVector.end(),
        recordVector.begin(),
        -1.0 * (boost::compute::lambda::_1 * boost::compute::lambda::_1),
        queue_);
    // Note : -(x - avg)^2

    fVar.wait();
    boost::compute::transform(
        recordVector.begin(),
        recordVector.end(),
        varianceVector.begin(),
        recordVector.begin(),
        exp(boost::compute::lambda::_1 / (2.0 * boost::compute::lambda::_2)),
        queue_);
    // Note : uses builtin function boost::compute::exp<ValueType(ValueType)>()
    //        wrapped using BOOST_COMPUTE_LAMBDA_WRAP_UNARY_FUNCTION_T

    boost::compute::transform(recordVector.begin(),
                              recordVector.end(),
                              varianceVector.begin(),
                              recordVector.begin(),
                              boost::compute::lambda::_1
                                  / (sqrt(boost::compute::lambda::_2) * sqr2pi),
                              queue_);

    // 3. summarize
    ValueType propability = 1.0;
    boost::compute::reduce(recordVector.begin(),
                           recordVector.end(),
                           &propability,
                           //(boost::compute::multiplies<ValueType>()),
                           boost::compute::lambda::_1
                               * boost::compute::lambda::_2,
                           queue_);

    // 4. return propability
    return propability;
  }
} // namespace nbc4gpu

#endif // NBC4GPU_GPUCALCULATECLASSP_HPP
