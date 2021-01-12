//
// Created by przestaw on 16.12.2020.
//
#include <boost/test/unit_test.hpp>
#include <gpuclassifier/GPUCalculateClassP.hpp>

BOOST_AUTO_TEST_SUITE(nbc4gpu_TestSuite)

BOOST_AUTO_TEST_SUITE(ClassPropability_TestSuite)

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size0) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUCalculateClassP<float>::Statistics stats = {};
  BOOST_CHECK_THROW((nbc4gpu::GPUCalculateClassP<float>(stats, 1, queue)),
                    nbc4gpu::error::ZeroValuesProvided);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_MismatchSize) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUCalculateClassP<float>::Statistics stats = {{1, 1}};
  nbc4gpu::GPUCalculateClassP<float> classifier(stats, 0, queue);
  BOOST_CHECK_THROW(classifier({1, 2, 3, 4}), nbc4gpu::error::MismatchedSize);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size1_1) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUCalculateClassP<float>::Statistics stats = {{1, 1}};
  nbc4gpu::GPUCalculateClassP<float> classifier(stats, 0, queue);
  // P = 1/(sqr(2*pi) * sqr(var)) * exp(-((x-avg)^2 / 2*var))

  // P = 1/(sqr(2*pi) * 1) * exp(-((1-1)^2 / 2*1)) = 0.3989... * exp(0)  =
  // 0.3989
  BOOST_CHECK_CLOSE(classifier.operator()({1}), 0.3989, 0.1);

  BOOST_CHECK_CLOSE(classifier.operator()({0}), 0.2419, 0.1);

  BOOST_CHECK_CLOSE(classifier.operator()({2}), 0.2419, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size1_2) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUCalculateClassP<float>::Statistics stats = {{2, 5}};
  nbc4gpu::GPUCalculateClassP<float> classifier(stats, 0, queue);

  BOOST_CHECK_CLOSE(classifier.operator()({1}), 0.161449, 0.1);

  BOOST_CHECK_CLOSE(classifier.operator()({2}), 0.178429, 0.1);

  BOOST_CHECK_CLOSE(classifier.operator()({0}), 0.119605, 0.1);

  BOOST_CHECK_CLOSE(classifier.operator()({3}), 0.161449, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size4_1) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUCalculateClassP<float>::Statistics stats = {
      {2, 1}, {2, 1}, {2, 1}, {2, 1}};
  nbc4gpu::GPUCalculateClassP<float> classifier(stats, 0, queue);

  BOOST_CHECK_CLOSE(classifier.operator()({1, 0, 2, 3}), 0.001261597, 0.1);
  BOOST_CHECK_CLOSE(classifier.operator()({0, 1, 2, 3}), 0.001261597, 0.1);
  BOOST_CHECK_CLOSE(classifier.operator()({3, 1, 0, 2}), 0.001261597, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size4_2) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUCalculateClassP<float>::Statistics stats = {
      {2, 5}, {2, 3}, {0.15, 4}, {3.5, 1.5}};
  nbc4gpu::GPUCalculateClassP<float> classifier(stats, 0, queue);

  BOOST_CHECK_CLOSE(classifier.operator()({1, 2, 0, 3}), 0.002217, 0.1);
  BOOST_CHECK_CLOSE(classifier.operator()({2.5, 3, 0.5, 4}), 0.001998, 0.1);
  BOOST_CHECK_CLOSE(classifier.operator()({1, 1, 1, 1}), 0.00023274, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size0_double) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUCalculateClassP<double>::Statistics stats = {};
  BOOST_CHECK_THROW((nbc4gpu::GPUCalculateClassP<double>(stats, 1.0, queue)),
                    nbc4gpu::error::ZeroValuesProvided);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_MismatchSize_double) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUCalculateClassP<double>::Statistics stats = {{1, 1}};
  nbc4gpu::GPUCalculateClassP<double> classifier(stats, 0, queue);
  BOOST_CHECK_THROW(classifier({1, 2, 3, 4}), nbc4gpu::error::MismatchedSize);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size1_1_double) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUCalculateClassP<double>::Statistics stats = {{1, 1}};
  nbc4gpu::GPUCalculateClassP<double> classifier(stats, 0, queue);
  // P = 1/(sqr(2*pi) * sqr(var)) * exp(-((x-avg)^2 / 2*var))

  // P = 1/(sqr(2*pi) * 1) * exp(-((1-1)^2 / 2*1)) = 0.3989... * exp(0)  =
  // 0.3989
  BOOST_CHECK_CLOSE(classifier.operator()({1}), 0.3989, 0.1);

  BOOST_CHECK_CLOSE(classifier.operator()({0}), 0.2419, 0.1);

  BOOST_CHECK_CLOSE(classifier.operator()({2}), 0.2419, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size1_2_double) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUCalculateClassP<double>::Statistics stats = {{2, 5}};
  nbc4gpu::GPUCalculateClassP<double> classifier(stats, 0, queue);

  BOOST_CHECK_CLOSE(classifier.operator()({1}), 0.161449, 0.1);

  BOOST_CHECK_CLOSE(classifier.operator()({2}), 0.178429, 0.1);

  BOOST_CHECK_CLOSE(classifier.operator()({0}), 0.119605, 0.1);

  BOOST_CHECK_CLOSE(classifier.operator()({3}), 0.161449, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size4_1_double) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUCalculateClassP<double>::Statistics stats = {
      {2, 1}, {2, 1}, {2, 1}, {2, 1}};
  nbc4gpu::GPUCalculateClassP<double> classifier(stats, 0, queue);

  BOOST_CHECK_CLOSE(classifier.operator()({1, 0, 2, 3}), 0.001261597, 0.1);
  BOOST_CHECK_CLOSE(classifier.operator()({0, 1, 2, 3}), 0.001261597, 0.1);
  BOOST_CHECK_CLOSE(classifier.operator()({3, 1, 0, 2}), 0.001261597, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size4_2_double) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUCalculateClassP<double>::Statistics stats = {
      {2, 5}, {2, 3}, {0.15, 4}, {3.5, 1.5}};
  nbc4gpu::GPUCalculateClassP<double> classifier(stats, 0, queue);

  BOOST_CHECK_CLOSE(classifier.operator()({1, 2, 0, 3}), 0.002217, 0.1);
  BOOST_CHECK_CLOSE(classifier.operator()({2.5, 3, 0.5, 4}), 0.001998, 0.1);
  BOOST_CHECK_CLOSE(classifier.operator()({1, 1, 1, 1}), 0.00023274, 0.1);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()