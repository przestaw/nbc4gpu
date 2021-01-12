//
// Created by przestaw on 18.12.2020.
//
#include <boost/test/unit_test.hpp>
#include <gpuclassifier/GPUClassifier.hpp>

BOOST_AUTO_TEST_SUITE(nbc4gpu_TestSuite)

BOOST_AUTO_TEST_SUITE(Classifier_TestSuite)

BOOST_AUTO_TEST_CASE(ThrowTestTest_Size0) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUClassifier<float>::FullDataset data = {};
  auto classifier = nbc4gpu::GPUClassifier<float>(data);
  BOOST_CHECK_THROW((classifier.calculateProbabilities({})),
                    nbc4gpu::error::NotLearned);
}

BOOST_AUTO_TEST_CASE(ThrowTestTest_Size1) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUClassifier<float>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}};
  auto classifier = nbc4gpu::GPUClassifier<float>(data);
  BOOST_CHECK_THROW((classifier.calculateProbabilities({})),
                    nbc4gpu::error::NotLearned);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size1SingleProb) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUClassifier<float>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}};
  auto classifier = nbc4gpu::GPUClassifier<float>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateMostProbableClass({1.5, 1.5});

  BOOST_CHECK_EQUAL(ret.second, 1.0);
  BOOST_CHECK_CLOSE(ret.first, 0.3183, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size1_NewDatasetThrows) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUClassifier<float>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}};
  auto classifier = nbc4gpu::GPUClassifier<float>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateMostProbableClass({1.5, 1.5});

  BOOST_CHECK_EQUAL(ret.second, 1.0);
  BOOST_CHECK_CLOSE(ret.first, 0.3183, 0.1);

  classifier.insertNewDataset({});
  BOOST_CHECK_THROW((classifier.calculateProbabilities({})),
                    nbc4gpu::error::NotLearned);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size1_NewDatasetWorks) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUClassifier<float>::FullDataset data1 = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}};
  auto classifier = nbc4gpu::GPUClassifier<float>(data1);
  classifier.learnClasses();

  auto ret = classifier.calculateMostProbableClass({1.5, 1.5});

  BOOST_CHECK_EQUAL(ret.second, 1.0);
  BOOST_CHECK_CLOSE(ret.first, 0.3183, 0.1);

  nbc4gpu::GPUClassifier<float>::FullDataset data2 = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}, {2.0, {{4.0, 3.0}, {3.0, 4.0}}}};
  classifier.insertNewDataset(data2);
  classifier.learnClasses();

  ret = classifier.calculateMostProbableClass({3.5, 3.5});

  BOOST_CHECK_EQUAL(ret.second, 2.0);
  BOOST_CHECK_CLOSE(ret.first, 0.3183, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size1MultiProb) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUClassifier<float>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}};
  auto classifier = nbc4gpu::GPUClassifier<float>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateProbabilities({1.5, 1.5});

  BOOST_REQUIRE_EQUAL(ret.size(), 1);

  BOOST_CHECK_EQUAL(ret[0].second, 1.0);
  BOOST_CHECK_CLOSE(ret[0].first, 0.3183, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size2SingleProb) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUClassifier<float>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}, {2.0, {{4.0, 3.0}, {3.0, 4.0}}}};
  auto classifier = nbc4gpu::GPUClassifier<float>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateMostProbableClass({3.5, 3.5});

  BOOST_CHECK_EQUAL(ret.second, 2.0);
  BOOST_CHECK_CLOSE(ret.first, 0.3183, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size2MultiProb1) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUClassifier<float>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}, {2.0, {{4.0, 3.0}, {3.0, 4.0}}}};
  auto classifier = nbc4gpu::GPUClassifier<float>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateProbabilities({1.5, 1.5});

  BOOST_REQUIRE_EQUAL(ret.size(), 2);

  BOOST_CHECK_EQUAL(ret[0].second, 1.0);
  BOOST_CHECK_CLOSE(ret[0].first, 0.3183, 0.1);
  BOOST_CHECK_EQUAL(ret[1].second, 2.0);
  BOOST_CHECK_CLOSE(ret[1].first, 0.0001067, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size2MultiProb2) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUClassifier<float>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}, {2.0, {{4.0, 3.0}, {3.0, 4.0}}}};
  auto classifier = nbc4gpu::GPUClassifier<float>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateProbabilities({3.5, 3.5});

  BOOST_REQUIRE_EQUAL(ret.size(), 2);

  BOOST_CHECK_EQUAL(ret[0].second, 1.0);
  BOOST_CHECK_CLOSE(ret[0].first, 0.0001067, 0.1);
  BOOST_CHECK_EQUAL(ret[1].second, 2.0);
  BOOST_CHECK_CLOSE(ret[1].first, 0.3183, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size4SingleProb) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUClassifier<float>::FullDataset data = {
      {1.0, {{1.0, 7.0}, {2.5, 7.0}}},
      {2.0, {{4.0, 3.0}, {3.0, 4.0}}},
      {3.0, {{0.0, 3.0}, {3.0, 14.0}}},
      {4.0, {{20.0, 0.0}, {30.0, 0.0}}}};
  auto classifier = nbc4gpu::GPUClassifier<float>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateMostProbableClass({5, 5});

  BOOST_CHECK_EQUAL(ret.second, 1.0);
  BOOST_CHECK_CLOSE(ret.first, 0.01143, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size4MultiProb) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUClassifier<float>::FullDataset data = {
      {1.0, {{1.0, 7.0}, {2.5, 7.0}}},
      {2.0, {{4.0, 3.0}, {3.0, 4.0}}},
      {3.0, {{0.0, 3.0}, {3.0, 14.0}}},
      {4.0, {{20.0, 0.0}, {30.0, 0.0}}}};
  auto classifier = nbc4gpu::GPUClassifier<float>(data);
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

BOOST_AUTO_TEST_CASE(ThrowTestTest_Size0_double) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUClassifier<double>::FullDataset data = {};
  auto classifier = nbc4gpu::GPUClassifier<double>(data);
  BOOST_CHECK_THROW((classifier.calculateProbabilities({})),
                    nbc4gpu::error::NotLearned);
}

BOOST_AUTO_TEST_CASE(ThrowTestTest_Size1_double) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUClassifier<double>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}};
  auto classifier = nbc4gpu::GPUClassifier<double>(data);
  BOOST_CHECK_THROW((classifier.calculateProbabilities({})),
                    nbc4gpu::error::NotLearned);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size1_NewDatasetThrows_double) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUClassifier<double>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}};
  auto classifier = nbc4gpu::GPUClassifier<double>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateMostProbableClass({1.5, 1.5});

  BOOST_CHECK_EQUAL(ret.second, 1.0);
  BOOST_CHECK_CLOSE(ret.first, 0.3183, 0.1);

  classifier.insertNewDataset({});
  BOOST_CHECK_THROW((classifier.calculateProbabilities({})),
                    nbc4gpu::error::NotLearned);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size1_NewDatasetWorks_double) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUClassifier<double>::FullDataset data1 = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}};
  auto classifier = nbc4gpu::GPUClassifier<double>(data1);
  classifier.learnClasses();

  auto ret = classifier.calculateMostProbableClass({1.5, 1.5});

  BOOST_CHECK_EQUAL(ret.second, 1.0);
  BOOST_CHECK_CLOSE(ret.first, 0.3183, 0.1);

  nbc4gpu::GPUClassifier<double>::FullDataset data2 = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}, {2.0, {{4.0, 3.0}, {3.0, 4.0}}}};
  classifier.insertNewDataset(data2);
  classifier.learnClasses();

  ret = classifier.calculateMostProbableClass({3.5, 3.5});

  BOOST_CHECK_EQUAL(ret.second, 2.0);
  BOOST_CHECK_CLOSE(ret.first, 0.3183, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size1SingleProb_double) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUClassifier<double>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}};
  auto classifier = nbc4gpu::GPUClassifier<double>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateMostProbableClass({1.5, 1.5});

  BOOST_CHECK_EQUAL(ret.second, 1.0);
  BOOST_CHECK_CLOSE(ret.first, 0.3183, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size1MultiProb_double) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUClassifier<double>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}};
  auto classifier = nbc4gpu::GPUClassifier<double>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateProbabilities({1.5, 1.5});

  BOOST_REQUIRE_EQUAL(ret.size(), 1);

  BOOST_CHECK_EQUAL(ret[0].second, 1.0);
  BOOST_CHECK_CLOSE(ret[0].first, 0.3183, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size2SingleProb_double) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUClassifier<double>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}, {2.0, {{4.0, 3.0}, {3.0, 4.0}}}};
  auto classifier = nbc4gpu::GPUClassifier<double>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateMostProbableClass({3.5, 3.5});

  BOOST_CHECK_EQUAL(ret.second, 2.0);
  BOOST_CHECK_CLOSE(ret.first, 0.3183, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size2MultiProb1_double) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUClassifier<double>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}, {2.0, {{4.0, 3.0}, {3.0, 4.0}}}};
  auto classifier = nbc4gpu::GPUClassifier<double>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateProbabilities({1.5, 1.5});

  BOOST_REQUIRE_EQUAL(ret.size(), 2);

  BOOST_CHECK_EQUAL(ret[0].second, 1.0);
  BOOST_CHECK_CLOSE(ret[0].first, 0.3183, 0.1);
  BOOST_CHECK_EQUAL(ret[1].second, 2.0);
  BOOST_CHECK_CLOSE(ret[1].first, 0.0001067, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size2MultiProb2_double) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUClassifier<double>::FullDataset data = {
      {1.0, {{1.0, 2.0}, {2.0, 1.0}}}, {2.0, {{4.0, 3.0}, {3.0, 4.0}}}};
  auto classifier = nbc4gpu::GPUClassifier<double>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateProbabilities({3.5, 3.5});

  BOOST_REQUIRE_EQUAL(ret.size(), 2);

  BOOST_CHECK_EQUAL(ret[0].second, 1.0);
  BOOST_CHECK_CLOSE(ret[0].first, 0.0001067, 0.1);
  BOOST_CHECK_EQUAL(ret[1].second, 2.0);
  BOOST_CHECK_CLOSE(ret[1].first, 0.3183, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size4SingleProb_double) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUClassifier<double>::FullDataset data = {
      {1.0, {{1.0, 7.0}, {2.5, 7.0}}},
      {2.0, {{4.0, 3.0}, {3.0, 4.0}}},
      {3.0, {{0.0, 3.0}, {3.0, 14.0}}},
      {4.0, {{20.0, 0.0}, {30.0, 0.0}}}};
  auto classifier = nbc4gpu::GPUClassifier<double>(data);
  classifier.learnClasses();

  auto ret = classifier.calculateMostProbableClass({5, 5});

  BOOST_CHECK_EQUAL(ret.second, 1.0);
  BOOST_CHECK_CLOSE(ret.first, 0.01143, 0.1);
}

BOOST_AUTO_TEST_CASE(PrecalculatedTestTest_Size4MultiProb_double) {
  boost::compute::device device = boost::compute::system::default_device();
  boost::compute::context context(device);
  boost::compute::command_queue queue(context, device);

  nbc4gpu::GPUClassifier<double>::FullDataset data = {
      {1.0, {{1.0, 7.0}, {2.5, 7.0}}},
      {2.0, {{4.0, 3.0}, {3.0, 4.0}}},
      {3.0, {{0.0, 3.0}, {3.0, 14.0}}},
      {4.0, {{20.0, 0.0}, {30.0, 0.0}}}};
  auto classifier = nbc4gpu::GPUClassifier<double>(data);
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