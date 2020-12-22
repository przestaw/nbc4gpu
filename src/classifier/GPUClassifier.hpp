//
// Created by przestaw on 18.12.2020.
//

#ifndef NBC4GPU_GPUCLASSIFIER_H
#define NBC4GPU_GPUCLASSIFIER_H

#include <Defines.h>
#include <error/Execeptions.h>
DIAGNOSTIC_PUSH
#include <boost/compute/algorithm/accumulate.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/core.hpp>
#include <boost/compute/lambda.hpp>
DIAGNOSTIC_POP
#include "GPUCalculateClassP.hpp"
#include <classifier/GPULearnClass.hpp>

namespace nbc4gpu {
  template <typename ValueType> class GPUClassifier {
  public:
    using LearnClass      = nbc4gpu::GPULearnClass<ValueType>;
    using CalculateClassP = nbc4gpu::GPUCalculateClassP<ValueType>;

    using ClassDataset = std::pair<ValueType, typename LearnClass::Dataset>;
    using FullDataset  = std::vector<ClassDataset>;

    using Probability = std::pair<ValueType, ValueType>; // pair<P, class value>
    using Record      = typename CalculateClassP::Record;

    /**
     * Constructor
     * @param fullDataset dataset in form of column vectors divided by classes
     */
    explicit GPUClassifier(FullDataset fullDataset);

    /**
     * Learns classes statistics inf not learned before
     */
    void learnClasses();

    /**
     * Calculates probability for given record and returns vector of class
     * probabilities
     * @param record record to calculate probability
     * @return vector of class probabilities
     */
    std::vector<Probability> calculateProbabilities(Record record);

    /**
     * Calculates probability for given record and returns most probable class
     * @param record record to calculate probability
     * @return most probable class
     */
    Probability calculateMostProbableClass(Record record);

    /**
     * Reset classifier and insert new data
     * @param fullDataset dataset in form of column vectors divided by classes
     */
    void insertNewDataset(FullDataset fullDataset);

  private:
    std::mutex guard;
    bool learned;

    // deque as types are unmovable due to mutexes
    std::deque<LearnClass> learn;
    std::deque<CalculateClassP> learnedP;

    boost::compute::command_queue queue;
  };

  template <typename ValueType>
  GPUClassifier<ValueType>::GPUClassifier(
      GPUClassifier::FullDataset fullDataset)
      : learned(false) {
    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    queue = boost::compute::command_queue(context, device);

    for (ClassDataset &it : fullDataset) {
      learn.emplace_back(it.second, it.first, queue);
    }
  }

  template <typename ValueType> void GPUClassifier<ValueType>::learnClasses() {
    if (!learned) {
      std::unique_lock lock(guard);
      if (!learned) {
        for (auto &it : learn) {
          learnedP.emplace_back(
              CalculateClassP(it.operator()(), it.getClassId(), queue));
        }
        learned = true;
      }
    }
  }

  template <typename ValueType>
  std::vector<typename GPUClassifier<ValueType>::Probability>
  GPUClassifier<ValueType>::calculateProbabilities(
      GPUClassifier::Record record) {
    if (learned) {
      std::unique_lock lock(guard);
      if(learned){
        std::vector<typename GPUClassifier<ValueType>::Probability> returnVec;
        for (auto &it : learnedP) {
          returnVec.emplace_back(it.operator()(record), it.getClassId());
        }
        return returnVec;
      }
    }
    throw nbc4gpu::error::NotLearned();
  }

  template <typename ValueType>
  typename GPUClassifier<ValueType>::Probability
  GPUClassifier<ValueType>::calculateMostProbableClass(
      GPUClassifier::Record record) {
    // note: wraps calculateProbabilities method
    auto prob = calculateProbabilities(std::move(record));
    auto max = std::max_element(
        prob.begin(), prob.end(),
    [](const Probability &p1,
       const Probability &p2) {
      return p1.first < p2.first;
    });
    if (max != prob.end()){
      return *max;
    } else {
      throw nbc4gpu::error::ZeroValuesProvided("No classes in dataset");
    }
  }

  template <typename ValueType>
  void GPUClassifier<ValueType>::insertNewDataset(
      GPUClassifier<ValueType>::FullDataset fullDataset) {
    std::unique_lock lock(guard);
    // reset
    learned = false;
    learnedP.clear();
    learn.clear();
    // insert new dataset
    for (ClassDataset &it : fullDataset) {
      learn.emplace_back(it.second, it.first, queue);
    }
  }
} // namespace nbc4gpu

#endif // NBC4GPU_GPUCLASSIFIER_H
