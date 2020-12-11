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
      : record_(record), statistics_(statistics), queue_(queue), calculated(false) {
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
    // TODO
  }
} // namespace nbc4gpu

#endif // NBC4GPU_GPUCALCULATECLASSP_HPP
