//
// Created by przestaw on 16.12.2020.
//
#include <boost/test/unit_test.hpp>
#include <cpuclassifier/CPUCalculateClassP.hpp>

BOOST_AUTO_TEST_SUITE(nbc4cpu_TestSuite)

BOOST_AUTO_TEST_SUITE(ClassPropability_TestSuite)


BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size1_1) {

  nbc4cpu::CPUCalculateClassP<float>::Statistics stats = {{1, 1}};
  nbc4cpu::CPUCalculateClassP<float> classifier(stats, 0);
  // P = 1/(sqr(2*pi) * sqr(var)) * exp(-((x-avg)^2 / 2*var))

  // P = 1/(sqr(2*pi) * 1) * exp(-((1-1)^2 / 2*1)) = 0.3989... * exp(0)  =
  // 0.3989
  BOOST_CHECK_CLOSE(classifier.operator()({1}), 0.3989, 0.1);

  BOOST_CHECK_CLOSE(classifier.operator()({0}), 0.2419, 0.1);

  BOOST_CHECK_CLOSE(classifier.operator()({2}), 0.2419, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size1_2) {

  nbc4cpu::CPUCalculateClassP<float>::Statistics stats = {{2, 5}};
  nbc4cpu::CPUCalculateClassP<float> classifier(stats, 0);

  BOOST_CHECK_CLOSE(classifier.operator()({1}), 0.161449, 0.1);

  BOOST_CHECK_CLOSE(classifier.operator()({2}), 0.178429, 0.1);

  BOOST_CHECK_CLOSE(classifier.operator()({0}), 0.119605, 0.1);

  BOOST_CHECK_CLOSE(classifier.operator()({3}), 0.161449, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size4_1) {

  nbc4cpu::CPUCalculateClassP<float>::Statistics stats = {
      {2, 1}, {2, 1}, {2, 1}, {2, 1}};
  nbc4cpu::CPUCalculateClassP<float> classifier(stats, 0);

  BOOST_CHECK_CLOSE(classifier.operator()({1, 0, 2, 3}), 0.001261597, 0.1);
  BOOST_CHECK_CLOSE(classifier.operator()({0, 1, 2, 3}), 0.001261597, 0.1);
  BOOST_CHECK_CLOSE(classifier.operator()({3, 1, 0, 2}), 0.001261597, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size4_2) {

  nbc4cpu::CPUCalculateClassP<float>::Statistics stats = {
      {2, 5}, {2, 3}, {0.15, 4}, {3.5, 1.5}};
  nbc4cpu::CPUCalculateClassP<float> classifier(stats, 0);

  BOOST_CHECK_CLOSE(classifier.operator()({1, 2, 0, 3}), 0.002217, 0.1);
  BOOST_CHECK_CLOSE(classifier.operator()({2.5, 3, 0.5, 4}), 0.001998, 0.1);
  BOOST_CHECK_CLOSE(classifier.operator()({1, 1, 1, 1}), 0.00023274, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size1_1_double) {

  nbc4cpu::CPUCalculateClassP<double>::Statistics stats = {{1, 1}};
  nbc4cpu::CPUCalculateClassP<double> classifier(stats, 0);
  // P = 1/(sqr(2*pi) * sqr(var)) * exp(-((x-avg)^2 / 2*var))

  // P = 1/(sqr(2*pi) * 1) * exp(-((1-1)^2 / 2*1)) = 0.3989... * exp(0)  =
  // 0.3989
  BOOST_CHECK_CLOSE(classifier.operator()({1}), 0.3989, 0.1);

  BOOST_CHECK_CLOSE(classifier.operator()({0}), 0.2419, 0.1);

  BOOST_CHECK_CLOSE(classifier.operator()({2}), 0.2419, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size1_2_double) {

  nbc4cpu::CPUCalculateClassP<double>::Statistics stats = {{2, 5}};
  nbc4cpu::CPUCalculateClassP<double> classifier(stats, 0);

  BOOST_CHECK_CLOSE(classifier.operator()({1}), 0.161449, 0.1);

  BOOST_CHECK_CLOSE(classifier.operator()({2}), 0.178429, 0.1);

  BOOST_CHECK_CLOSE(classifier.operator()({0}), 0.119605, 0.1);

  BOOST_CHECK_CLOSE(classifier.operator()({3}), 0.161449, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size4_1_double) {

  nbc4cpu::CPUCalculateClassP<double>::Statistics stats = {
      {2, 1}, {2, 1}, {2, 1}, {2, 1}};
  nbc4cpu::CPUCalculateClassP<double> classifier(stats, 0);

  BOOST_CHECK_CLOSE(classifier.operator()({1, 0, 2, 3}), 0.001261597, 0.1);
  BOOST_CHECK_CLOSE(classifier.operator()({0, 1, 2, 3}), 0.001261597, 0.1);
  BOOST_CHECK_CLOSE(classifier.operator()({3, 1, 0, 2}), 0.001261597, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size4_2_double) {

  nbc4cpu::CPUCalculateClassP<double>::Statistics stats = {
      {2, 5}, {2, 3}, {0.15, 4}, {3.5, 1.5}};
  nbc4cpu::CPUCalculateClassP<double> classifier(stats, 0);

  BOOST_CHECK_CLOSE(classifier.operator()({1, 2, 0, 3}), 0.002217, 0.1);
  BOOST_CHECK_CLOSE(classifier.operator()({2.5, 3, 0.5, 4}), 0.001998, 0.1);
  BOOST_CHECK_CLOSE(classifier.operator()({1, 1, 1, 1}), 0.00023274, 0.1);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
