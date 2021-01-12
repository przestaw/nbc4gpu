//
// Created by przestaw on 28.09.2020.
//
#include <boost/test/unit_test.hpp>
#include <cpuclassifier/CPULearnColumn.hpp>
#include <numeric>

BOOST_AUTO_TEST_SUITE(nbc4cpu_TestSuite)

BOOST_AUTO_TEST_SUITE(LearnColumn_TestSuite)


BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size6) {

  std::vector<float> vec = {1, 1, 1, 2, 2, 2};

  nbc4cpu::CPULearnColumn<float> learner =
      nbc4cpu::CPULearnColumn<float>(vec);
  const auto res = learner();

  BOOST_CHECK_CLOSE(res.first, 1.5, 0.1);
  BOOST_CHECK_CLOSE(res.second, 0.3, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size12) {

  std::vector<float> vec = {91, 11, 1111, 32, 26, 2, 22, 894, 345};

  nbc4cpu::CPULearnColumn<float> learner =
      nbc4cpu::CPULearnColumn<float>(vec);
  const auto res = learner();

  BOOST_CHECK_CLOSE(res.first, 281.55555555556, 0.1);
  BOOST_CHECK_CLOSE(res.second, 181213.77777778, 0.1);
}

BOOST_AUTO_TEST_CASE(CompareWithCpu_LargeVectorTest) {

  srand(1234); // set rand seed to make test repeatable

  std::vector<float> vec(500000);
  std::generate(vec.begin(), vec.end(), []() { return rand() % 250; });

  nbc4cpu::CPULearnColumn<float> learner =
      nbc4cpu::CPULearnColumn<float>(vec);
  const auto res = learner();

  float sum = 0.0;
  sum       = std::accumulate(vec.begin(), vec.end(), sum);
  float avg = sum / static_cast<float>(vec.size());

  BOOST_CHECK_CLOSE(res.first, avg, 1);

  float variance = 0;
  for (float i : vec) {
    float dif = i - avg;
    variance += dif * dif;
  }
  variance = variance / static_cast<float>(vec.size() - 1);

  BOOST_CHECK_CLOSE(res.second, variance, 1);
}


BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size6_double) {

  std::vector<double> vec = {1, 1, 1, 2, 2, 2};

  nbc4cpu::CPULearnColumn<double> learner =
      nbc4cpu::CPULearnColumn<double>(vec);
  const auto res = learner();

  BOOST_CHECK_CLOSE(res.first, 1.5, 0.1);
  BOOST_CHECK_CLOSE(res.second, 0.3, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size12_double) {

  std::vector<double> vec = {91, 11, 1111, 32, 26, 2, 22, 894, 345};

  nbc4cpu::CPULearnColumn<double> learner =
      nbc4cpu::CPULearnColumn<double>(vec);
  const auto res = learner();

  BOOST_CHECK_CLOSE(res.first, 281.55555555556, 0.1);
  BOOST_CHECK_CLOSE(res.second, 181213.77777778, 0.1);
}

BOOST_AUTO_TEST_CASE(CompareWithCpu_LargeVectorTest_double) {

  srand(1234); // set rand seed to make test repeatable

  std::vector<double> vec(5000000);
  std::generate(vec.begin(), vec.end(), []() { return rand() % 250; });

  nbc4cpu::CPULearnColumn<double> learner =
      nbc4cpu::CPULearnColumn<double>(vec);
  const auto res = learner();

  double sum = 0.0;
  sum        = std::accumulate(vec.begin(), vec.end(), sum);
  double avg = sum / static_cast<double>(vec.size());

  BOOST_CHECK_CLOSE(res.first, avg, 1);

  double variance = 0;
  for (double i : vec) {
    double dif = i - avg;
    variance += dif * dif;
  }
  variance = variance / static_cast<double>(vec.size() - 1);

  BOOST_CHECK_CLOSE(res.second, variance, 1);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
