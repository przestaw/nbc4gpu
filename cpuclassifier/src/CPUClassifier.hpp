//
// Created by przestaw on 18.12.2020.
//

#ifndef NBC4CPU_CPUCLASSIFIER_H
#define NBC4CPU_CPUCLASSIFIER_H

#include "CPUCalculateClassP.hpp"
#include "CPULearnClass.hpp"
#include <deque>

namespace nbc4cpu {
  template <typename ValueType> class CPUClassifier {
  public:
    using LearnClass      = nbc4cpu::CPULearnClass<ValueType>;
    using CalculateClassP = nbc4cpu::CPUCalculateClassP<ValueType>;

    using ClassDataset = std::pair<ValueType, typename LearnClass::Dataset>;
    using FullDataset  = std::vector<ClassDataset>;

    using Probability = std::pair<ValueType, ValueType>; // pair<P, class value>
    using Record      = typename CalculateClassP::Record;

    /**
     * Constructor
     * @param fullDataset dataset in form of column vectors divided by classes
     */
    explicit CPUClassifier(FullDataset fullDataset);

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
    bool learned;

    // deque as types are unmovable due to mutexes
    std::deque<LearnClass> learn;
    std::deque<CalculateClassP> learnedP;
  };

  template <typename ValueType>
  CPUClassifier<ValueType>::CPUClassifier(
      CPUClassifier::FullDataset fullDataset)
      : learned(false) {

    for (ClassDataset &it : fullDataset) {
      learn.emplace_back(it.second, it.first);
    }
  }

  template <typename ValueType> void CPUClassifier<ValueType>::learnClasses() {
      if (!learned) {
        for (auto &it : learn) {
          learnedP.emplace_back(
              CalculateClassP(it.operator()(), it.getClassId()));
        }
        learned = true;
      }
  }

  template <typename ValueType>
  std::vector<typename CPUClassifier<ValueType>::Probability>
  CPUClassifier<ValueType>::calculateProbabilities(
      CPUClassifier::Record record) {
      if(learned){
        std::vector<typename CPUClassifier<ValueType>::Probability> returnVec;
        for (auto &it : learnedP) {
          returnVec.emplace_back(it.operator()(record), it.getClassId());
        }
        return returnVec;
      }
    throw "not learned";
  }

  template <typename ValueType>
  typename CPUClassifier<ValueType>::Probability
  CPUClassifier<ValueType>::calculateMostProbableClass(
      CPUClassifier::Record record) {
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
      throw ("No classes in dataset");
    }
  }

  template <typename ValueType>
  void CPUClassifier<ValueType>::insertNewDataset(
      CPUClassifier<ValueType>::FullDataset fullDataset) {
    // reset
    learned = false;
    learnedP.clear();
    learn.clear();
    // insert new dataset
    for (ClassDataset &it : fullDataset) {
      learn.emplace_back(it.second, it.first);
    }
  }
} // namespace nbc4cpu

#endif // NBC4CPU_CPUCLASSIFIER_H
