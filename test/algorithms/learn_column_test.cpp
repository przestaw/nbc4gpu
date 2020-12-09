//
// Created by przestaw on 28.09.2020.
//
#include <boost/test/unit_test.hpp>
#include <classifier/learnColumn.h>

BOOST_AUTO_TEST_SUITE(nbc4gpu_TestSuite)

BOOST_AUTO_TEST_SUITE(LwarnColumn_TestSuite)

BOOST_AUTO_TEST_CASE(EqualDummyTest) {
    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

    srand(1234); // set rand seed to make test repeatable

    std::vector<double> vec(5000000);
    std::generate(vec.begin(), vec.end(), []() { return rand() % 250;
    });

    nbc4gpu::GPULearnColumn<double> learner =
        nbc4gpu::GPULearnColumn<double>(vec, queue);
    const auto res = learner();

    double sum = 0.0;
    sum        = std::accumulate(vec.begin(), vec.end(), sum);
    BOOST_CHECK_EQUAL(res.first, sum / static_cast<double>(vec.size()));
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()