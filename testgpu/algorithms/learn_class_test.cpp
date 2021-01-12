//
// Created by przestaw on 14.12.2020.
//
#include <boost/test/unit_test.hpp>
#include <gpuclassifier/GPULearnClass.hpp>

BOOST_AUTO_TEST_SUITE(nbc4gpu_TestSuite)

BOOST_AUTO_TEST_SUITE(LearnClass_TestSuite)

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size0) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  std::vector<std::vector<float>> dataset = {};

  nbc4gpu::GPULearnClass<float> learner =
      nbc4gpu::GPULearnClass<float>(dataset, 1, queue);
  const auto res = learner();

  BOOST_CHECK_EQUAL(res.size(), 0);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size1) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  std::vector<std::vector<float>> dataset = {
      {91, 11, 1111, 32, 26, 2, 22, 894, 345}};

  nbc4gpu::GPULearnClass<float> learner =
      nbc4gpu::GPULearnClass<float>(dataset, 1, queue);
  const auto res = learner();

  BOOST_REQUIRE_EQUAL(res.size(), 1);

  BOOST_CHECK_CLOSE(res[0].first, 281.55555555556, 0.1);
  BOOST_CHECK_CLOSE(res[0].second, 181213.77777778, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size2) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  std::vector<std::vector<float>> dataset = {
      {91, 11, 1111, 32, 26, 2, 22, 894, 345},
      {302, 202, 106, 2, 9, 18, 22, 69, 96}};

  nbc4gpu::GPULearnClass<float> learner =
      nbc4gpu::GPULearnClass<float>(dataset, 1, queue);
  const auto res = learner();

  BOOST_REQUIRE_EQUAL(res.size(), 2);

  BOOST_CHECK_CLOSE(res[0].first, 281.55555555556, 0.1);
  BOOST_CHECK_CLOSE(res[0].second, 181213.77777778, 0.1);

  BOOST_CHECK_CLOSE(res[1].first, 91.777777777778, 0.1);
  BOOST_CHECK_CLOSE(res[1].second, 10288.194444444, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size4) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  std::vector<std::vector<float>> dataset = {
      {91, 11, 1111, 32, 26, 2, 22, 894, 345},
      {302, 202, 106, 2, 9, 18, 22, 69, 96},
      {901, 101, 1111, 302, 2.6, 2, 22, 894, 345},
      {91, 11, 1111, 32, 26, 2, 22, 894, 345}};

  nbc4gpu::GPULearnClass<float> learner =
      nbc4gpu::GPULearnClass<float>(dataset, 1, queue);
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
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  std::vector<std::vector<double>> dataset = {};

  nbc4gpu::GPULearnClass<double> learner =
      nbc4gpu::GPULearnClass<double>(dataset, 1, queue);
  const auto res = learner();

  BOOST_CHECK_EQUAL(res.size(), 0);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size1_double) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  std::vector<std::vector<double>> dataset = {
      {91, 11, 1111, 32, 26, 2, 22, 894, 345}};

  nbc4gpu::GPULearnClass<double> learner =
      nbc4gpu::GPULearnClass<double>(dataset, 1, queue);
  const auto res = learner();

  BOOST_REQUIRE_EQUAL(res.size(), 1);

  BOOST_CHECK_CLOSE(res[0].first, 281.55555555556, 0.1);
  BOOST_CHECK_CLOSE(res[0].second, 181213.77777778, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size2_double) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  std::vector<std::vector<double>> dataset = {
      {91, 11, 1111, 32, 26, 2, 22, 894, 345},
      {302, 202, 106, 2, 9, 18, 22, 69, 96}};

  nbc4gpu::GPULearnClass<double> learner =
      nbc4gpu::GPULearnClass<double>(dataset, 1, queue);
  const auto res = learner();

  BOOST_REQUIRE_EQUAL(res.size(), 2);

  BOOST_CHECK_CLOSE(res[0].first, 281.55555555556, 0.1);
  BOOST_CHECK_CLOSE(res[0].second, 181213.77777778, 0.1);

  BOOST_CHECK_CLOSE(res[1].first, 91.777777777778, 0.1);
  BOOST_CHECK_CLOSE(res[1].second, 10288.194444444, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalulatedTest_Size4_double) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  std::vector<std::vector<double>> dataset = {
      {91, 11, 1111, 32, 26, 2, 22, 894, 345},
      {302, 202, 106, 2, 9, 18, 22, 69, 96},
      {901, 101, 1111, 302, 2.6, 2, 22, 894, 345},
      {91, 11, 1111, 32, 26, 2, 22, 894, 345}};

  nbc4gpu::GPULearnClass<double> learner =
      nbc4gpu::GPULearnClass<double>(dataset, 1, queue);
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