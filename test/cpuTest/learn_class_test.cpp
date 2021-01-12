//
// Created by mateusz
//
#include <boost/test/unit_test.hpp>
#include <cpuclassifier/CPULearnClass.hpp>

BOOST_AUTO_TEST_SUITE(nbc4cpu_TestSuite)

BOOST_AUTO_TEST_SUITE(LearnClass_TestSuite)

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size0) {

  std::vector<std::vector<float>> dataset = {};

  nbc4cpu::CPULearnClass<float> learner =
      nbc4cpu::CPULearnClass<float>(dataset, 1);
  const auto res = learner();

  BOOST_CHECK_EQUAL(res.size(), 0);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size1) {

  std::vector<std::vector<float>> dataset = {
      {91, 11, 1111, 32, 26, 2, 22, 894, 345}};

  nbc4cpu::CPULearnClass<float> learner =
      nbc4cpu::CPULearnClass<float>(dataset, 1);
  const auto res = learner();

  BOOST_REQUIRE_EQUAL(res.size(), 1);

  BOOST_CHECK_CLOSE(res[0].first, 281.55555555556, 0.1);
  BOOST_CHECK_CLOSE(res[0].second, 181213.77777778, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size2) {

  std::vector<std::vector<float>> dataset = {
      {91, 11, 1111, 32, 26, 2, 22, 894, 345},
      {302, 202, 106, 2, 9, 18, 22, 69, 96}};

  nbc4cpu::CPULearnClass<float> learner =
      nbc4cpu::CPULearnClass<float>(dataset, 1);
  const auto res = learner();

  BOOST_REQUIRE_EQUAL(res.size(), 2);

  BOOST_CHECK_CLOSE(res[0].first, 281.55555555556, 0.1);
  BOOST_CHECK_CLOSE(res[0].second, 181213.77777778, 0.1);

  BOOST_CHECK_CLOSE(res[1].first, 91.777777777778, 0.1);
  BOOST_CHECK_CLOSE(res[1].second, 10288.194444444, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size4) {

  std::vector<std::vector<float>> dataset = {
      {91, 11, 1111, 32, 26, 2, 22, 894, 345},
      {302, 202, 106, 2, 9, 18, 22, 69, 96},
      {901, 101, 1111, 302, 2.6, 2, 22, 894, 345},
      {91, 11, 1111, 32, 26, 2, 22, 894, 345}};

  nbc4cpu::CPULearnClass<float> learner =
      nbc4cpu::CPULearnClass<float>(dataset, 1);
  const auto res = learner();

  BOOST_REQUIRE_EQUAL(res.size(), 4);

  BOOST_CHECK_CLOSE(res[0].first, 281.55555555556, 0.1);
  BOOST_CHECK_CLOSE(res[0].second, 181213.77777778, 0.1);

  BOOST_CHECK_CLOSE(res[1].first, 91.777777777778, 0.1);
  BOOST_CHECK_CLOSE(res[1].second, 10288.194444444, 0.1);

  BOOST_CHECK_CLOSE(res[2].first, 408.95555555555992, 10);
  BOOST_CHECK_CLOSE(res[2].second, 195135.77777777999, 0.1);

  BOOST_CHECK_CLOSE(res[3].first, 281.55555555556, 0.1);
  BOOST_CHECK_CLOSE(res[3].second, 181213.77777778, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size0_double) {

  std::vector<std::vector<double>> dataset = {};

  nbc4cpu::CPULearnClass<double> learner =
      nbc4cpu::CPULearnClass<double>(dataset, 1);
  const auto res = learner();

  BOOST_CHECK_EQUAL(res.size(), 0);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size1_double) {

  std::vector<std::vector<double>> dataset = {
      {91, 11, 1111, 32, 26, 2, 22, 894, 345}};

  nbc4cpu::CPULearnClass<double> learner =
      nbc4cpu::CPULearnClass<double>(dataset, 1);
  const auto res = learner();

  BOOST_REQUIRE_EQUAL(res.size(), 1);

  BOOST_CHECK_CLOSE(res[0].first, 281.55555555556, 0.1);
  BOOST_CHECK_CLOSE(res[0].second, 181213.77777778, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size2_double) {

  std::vector<std::vector<double>> dataset = {
      {91, 11, 1111, 32, 26, 2, 22, 894, 345},
      {302, 202, 106, 2, 9, 18, 22, 69, 96}};

  nbc4cpu::CPULearnClass<double> learner =
      nbc4cpu::CPULearnClass<double>(dataset, 1);
  const auto res = learner();

  BOOST_REQUIRE_EQUAL(res.size(), 2);

  BOOST_CHECK_CLOSE(res[0].first, 281.55555555556, 0.1);
  BOOST_CHECK_CLOSE(res[0].second, 181213.77777778, 0.1);

  BOOST_CHECK_CLOSE(res[1].first, 91.777777777778, 0.1);
  BOOST_CHECK_CLOSE(res[1].second, 10288.194444444, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size4_double) {

  std::vector<std::vector<double>> dataset = {
      {91, 11, 1111, 32, 26, 2, 22, 894, 345},
      {302, 202, 106, 2, 9, 18, 22, 69, 96},
      {901, 101, 1111, 302, 2.6, 2, 22, 894, 345},
      {91, 11, 1111, 32, 26, 2, 22, 894, 345}};

  nbc4cpu::CPULearnClass<double> learner =
      nbc4cpu::CPULearnClass<double>(dataset, 1);
  const auto res = learner();

  BOOST_REQUIRE_EQUAL(res.size(), 4);

  BOOST_CHECK_CLOSE(res[0].first, 281.55555555556, 0.1);
  BOOST_CHECK_CLOSE(res[0].second, 181213.77777778, 0.1);

  BOOST_CHECK_CLOSE(res[1].first, 91.777777777778, 0.1);
  BOOST_CHECK_CLOSE(res[1].second, 10288.194444444, 0.1);

  BOOST_CHECK_CLOSE(res[2].first, 408.95555555555992, 10);
  BOOST_CHECK_CLOSE(res[2].second, 195135.77777777999, 0.1);

  BOOST_CHECK_CLOSE(res[3].first, 281.55555555556, 0.1);
  BOOST_CHECK_CLOSE(res[3].second, 181213.77777778, 0.1);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
