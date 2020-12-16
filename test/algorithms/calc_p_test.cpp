//
// Created by przestaw on 16.12.2020.
//
#include <boost/test/unit_test.hpp>
#include <classifier/GPUCalculateClassP.hpp>

BOOST_AUTO_TEST_SUITE(nbc4gpu_TestSuite)

BOOST_AUTO_TEST_SUITE(ClassPropability_TestSuite)

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size0) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUCalculateClassP<double, double>::Statistics stats = {};
  BOOST_CHECK_THROW(
      (nbc4gpu::GPUCalculateClassP<double, double>(stats, queue)),
      nbc4gpu::error::ZeroValuesProvided);
}

  BOOST_AUTO_TEST_CASE(PrecalulatedTest_MismatchSize) {
    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

    nbc4gpu::GPUCalculateClassP<double, double>::Statistics stats = {{1, 1}};
    nbc4gpu::GPUCalculateClassP<double, double> classifier(stats, queue);
    BOOST_CHECK_THROW(
        classifier({1,2,3,4}),
        nbc4gpu::error::MismatchedSize);
  }

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size1) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUCalculateClassP<double, double>::Statistics stats = {{1, 1}};
  nbc4gpu::GPUCalculateClassP<double, double> classifier(stats, queue);
  // P = 1/(sqr(2*pi) * var) * exp(-((x-avg)^2 / 2*var))

  // P = 1/(sqr(2*pi) * 1) * exp(-((1-1)^2 / 2*1)) = 0.3989... * exp(0)  =
  // 0.3989
  BOOST_CHECK_CLOSE(classifier.operator()({1}), 0.3989, 0.1);

  BOOST_CHECK_CLOSE(classifier.operator()({0}), 0.2419, 0.1);

  BOOST_CHECK_CLOSE(classifier.operator()({2}), 0.2419, 0.1);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()