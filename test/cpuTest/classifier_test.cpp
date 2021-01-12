//
// Created by przestaw on 18.12.2020.
//
#include <boost/test/unit_test.hpp>
#include <cpuclassifier/CPUClassifier.hpp>

BOOST_AUTO_TEST_SUITE(nbc4cpu_TestSuite)

BOOST_AUTO_TEST_SUITE(Classifier_TestSuite)

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size1SingleProb) {

  nbc4cpu::CPUClassifier<float>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}};
  auto classifier = nbc4cpu::CPUClassifier<float>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateMostProbableClass({1.5, 1.5});

  BOOST_CHECK_EQUAL(ret.second, 1.0);
  BOOST_CHECK_CLOSE(ret.first, 0.3183, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size1) {

  nbc4cpu::CPUClassifier<float>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}};
  auto classifier = nbc4cpu::CPUClassifier<float>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateMostProbableClass({1.5, 1.5});

  BOOST_CHECK_EQUAL(ret.second, 1.0);
  BOOST_CHECK_CLOSE(ret.first, 0.3183, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size1_NewDatasetWorks) {

  nbc4cpu::CPUClassifier<float>::FullDataset data1 = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}};
  auto classifier = nbc4cpu::CPUClassifier<float>(data1);
  classifier.learnClasses();

  auto ret = classifier.calculateMostProbableClass({1.5, 1.5});

  BOOST_CHECK_EQUAL(ret.second, 1.0);
  BOOST_CHECK_CLOSE(ret.first, 0.3183, 0.1);

  nbc4cpu::CPUClassifier<float>::FullDataset data2 = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}, {2.0, {{4.0, 3.0}, {3.0, 4.0}}}};
  classifier.insertNewDataset(data2);
  classifier.learnClasses();

  ret = classifier.calculateMostProbableClass({3.5, 3.5});

  BOOST_CHECK_EQUAL(ret.second, 2.0);
  BOOST_CHECK_CLOSE(ret.first, 0.3183, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size1MultiProb) {

  nbc4cpu::CPUClassifier<float>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}};
  auto classifier = nbc4cpu::CPUClassifier<float>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateProbabilities({1.5, 1.5});

  BOOST_REQUIRE_EQUAL(ret.size(), 1);

  BOOST_CHECK_EQUAL(ret[0].second, 1.0);
  BOOST_CHECK_CLOSE(ret[0].first, 0.3183, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size2SingleProb) {

  nbc4cpu::CPUClassifier<float>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}, {2.0, {{4.0, 3.0}, {3.0, 4.0}}}};
  auto classifier = nbc4cpu::CPUClassifier<float>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateMostProbableClass({3.5, 3.5});

  BOOST_CHECK_EQUAL(ret.second, 2.0);
  BOOST_CHECK_CLOSE(ret.first, 0.3183, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size2MultiProb1) {

  nbc4cpu::CPUClassifier<float>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}, {2.0, {{4.0, 3.0}, {3.0, 4.0}}}};
  auto classifier = nbc4cpu::CPUClassifier<float>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateProbabilities({1.5, 1.5});

  BOOST_REQUIRE_EQUAL(ret.size(), 2);

  BOOST_CHECK_EQUAL(ret[0].second, 1.0);
  BOOST_CHECK_CLOSE(ret[0].first, 0.3183, 0.1);
  BOOST_CHECK_EQUAL(ret[1].second, 2.0);
  BOOST_CHECK_CLOSE(ret[1].first, 0.0001067, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size2MultiProb2) {

  nbc4cpu::CPUClassifier<float>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}, {2.0, {{4.0, 3.0}, {3.0, 4.0}}}};
  auto classifier = nbc4cpu::CPUClassifier<float>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateProbabilities({3.5, 3.5});

  BOOST_REQUIRE_EQUAL(ret.size(), 2);

  BOOST_CHECK_EQUAL(ret[0].second, 1.0);
  BOOST_CHECK_CLOSE(ret[0].first, 0.0001067, 0.1);
  BOOST_CHECK_EQUAL(ret[1].second, 2.0);
  BOOST_CHECK_CLOSE(ret[1].first, 0.3183, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size4SingleProb) {

  nbc4cpu::CPUClassifier<float>::FullDataset data = {
      {1.0, {{1.0, 7.0}, {2.5, 7.0}}},
      {2.0, {{4.0, 3.0}, {3.0, 4.0}}},
      {3.0, {{0.0, 3.0}, {3.0, 14.0}}},
      {4.0, {{20.0, 0.0}, {30.0, 0.0}}}};
  auto classifier = nbc4cpu::CPUClassifier<float>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateMostProbableClass({5, 5});

  BOOST_CHECK_EQUAL(ret.second, 1.0);
  BOOST_CHECK_CLOSE(ret.first, 0.01143, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size4MultiProb) {

  nbc4cpu::CPUClassifier<float>::FullDataset data = {
      {1.0, {{1.0, 7.0}, {2.5, 7.0}}},
      {2.0, {{4.0, 3.0}, {3.0, 4.0}}},
      {3.0, {{0.0, 3.0}, {3.0, 14.0}}},
      {4.0, {{20.0, 0.0}, {30.0, 0.0}}}};
  auto classifier = nbc4cpu::CPUClassifier<float>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateProbabilities({5, 5});

  BOOST_REQUIRE_EQUAL(ret.size(), 4);

  BOOST_CHECK_EQUAL(ret[0].second, 1.0);
  BOOST_CHECK_CLOSE(ret[0].first, 0.01143, 0.1);
  BOOST_CHECK_EQUAL(ret[1].second, 2.0);
  BOOST_CHECK_CLOSE(ret[1].first, 0.003536, 0.1);
  BOOST_CHECK_EQUAL(ret[2].second, 3.0);
  BOOST_CHECK_CLOSE(ret[2].first, 0.002234, 0.1);
  BOOST_CHECK_EQUAL(ret[3].second, 4.0);
  BOOST_CHECK_CLOSE(ret[3].first, 0.0004459, 0.1);
}


BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size1_double) {

  nbc4cpu::CPUClassifier<double>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}};
  auto classifier = nbc4cpu::CPUClassifier<double>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateMostProbableClass({1.5, 1.5});

  BOOST_CHECK_EQUAL(ret.second, 1.0);
  BOOST_CHECK_CLOSE(ret.first, 0.3183, 0.1);

}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size1_NewDatasetWorks_double) {

  nbc4cpu::CPUClassifier<double>::FullDataset data1 = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}};
  auto classifier = nbc4cpu::CPUClassifier<double>(data1);
  classifier.learnClasses();

  auto ret = classifier.calculateMostProbableClass({1.5, 1.5});

  BOOST_CHECK_EQUAL(ret.second, 1.0);
  BOOST_CHECK_CLOSE(ret.first, 0.3183, 0.1);

  nbc4cpu::CPUClassifier<double>::FullDataset data2 = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}, {2.0, {{4.0, 3.0}, {3.0, 4.0}}}};
  classifier.insertNewDataset(data2);
  classifier.learnClasses();

  ret = classifier.calculateMostProbableClass({3.5, 3.5});

  BOOST_CHECK_EQUAL(ret.second, 2.0);
  BOOST_CHECK_CLOSE(ret.first, 0.3183, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size1SingleProb_double) {

  nbc4cpu::CPUClassifier<double>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}};
  auto classifier = nbc4cpu::CPUClassifier<double>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateMostProbableClass({1.5, 1.5});

  BOOST_CHECK_EQUAL(ret.second, 1.0);
  BOOST_CHECK_CLOSE(ret.first, 0.3183, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size1MultiProb_double) {

  nbc4cpu::CPUClassifier<double>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}};
  auto classifier = nbc4cpu::CPUClassifier<double>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateProbabilities({1.5, 1.5});

  BOOST_REQUIRE_EQUAL(ret.size(), 1);

  BOOST_CHECK_EQUAL(ret[0].second, 1.0);
  BOOST_CHECK_CLOSE(ret[0].first, 0.3183, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size2SingleProb_double) {

  nbc4cpu::CPUClassifier<double>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}, {2.0, {{4.0, 3.0}, {3.0, 4.0}}}};
  auto classifier = nbc4cpu::CPUClassifier<double>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateMostProbableClass({3.5, 3.5});

  BOOST_CHECK_EQUAL(ret.second, 2.0);
  BOOST_CHECK_CLOSE(ret.first, 0.3183, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size2MultiProb1_double) {

  nbc4cpu::CPUClassifier<double>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}, {2.0, {{4.0, 3.0}, {3.0, 4.0}}}};
  auto classifier = nbc4cpu::CPUClassifier<double>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateProbabilities({1.5, 1.5});

  BOOST_REQUIRE_EQUAL(ret.size(), 2);

  BOOST_CHECK_EQUAL(ret[0].second, 1.0);
  BOOST_CHECK_CLOSE(ret[0].first, 0.3183, 0.1);
  BOOST_CHECK_EQUAL(ret[1].second, 2.0);
  BOOST_CHECK_CLOSE(ret[1].first, 0.0001067, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size2MultiProb2_double) {

  nbc4cpu::CPUClassifier<double>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}, {2.0, {{4.0, 3.0}, {3.0, 4.0}}}};
  auto classifier = nbc4cpu::CPUClassifier<double>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateProbabilities({3.5, 3.5});

  BOOST_REQUIRE_EQUAL(ret.size(), 2);

  BOOST_CHECK_EQUAL(ret[0].second, 1.0);
  BOOST_CHECK_CLOSE(ret[0].first, 0.0001067, 0.1);
  BOOST_CHECK_EQUAL(ret[1].second, 2.0);
  BOOST_CHECK_CLOSE(ret[1].first, 0.3183, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size4SingleProb_double) {

  nbc4cpu::CPUClassifier<double>::FullDataset data = {
      {1.0, {{1.0, 7.0}, {2.5, 7.0}}},
      {2.0, {{4.0, 3.0}, {3.0, 4.0}}},
      {3.0, {{0.0, 3.0}, {3.0, 14.0}}},
      {4.0, {{20.0, 0.0}, {30.0, 0.0}}}};
  auto classifier = nbc4cpu::CPUClassifier<double>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateMostProbableClass({5, 5});

  BOOST_CHECK_EQUAL(ret.second, 1.0);
  BOOST_CHECK_CLOSE(ret.first, 0.01143, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size4MultiProb_double) {

  nbc4cpu::CPUClassifier<double>::FullDataset data = {
      {1.0, {{1.0, 7.0}, {2.5, 7.0}}},
      {2.0, {{4.0, 3.0}, {3.0, 4.0}}},
      {3.0, {{0.0, 3.0}, {3.0, 14.0}}},
      {4.0, {{20.0, 0.0}, {30.0, 0.0}}}};
  auto classifier = nbc4cpu::CPUClassifier<double>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateProbabilities({5, 5});

  BOOST_REQUIRE_EQUAL(ret.size(), 4);

  BOOST_CHECK_EQUAL(ret[0].second, 1.0);
  BOOST_CHECK_CLOSE(ret[0].first, 0.01143, 0.1);
  BOOST_CHECK_EQUAL(ret[1].second, 2.0);
  BOOST_CHECK_CLOSE(ret[1].first, 0.003536, 0.1);
  BOOST_CHECK_EQUAL(ret[2].second, 3.0);
  BOOST_CHECK_CLOSE(ret[2].first, 0.002234, 0.1);
  BOOST_CHECK_EQUAL(ret[3].second, 4.0);
  BOOST_CHECK_CLOSE(ret[3].first, 0.0004459, 0.1);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
