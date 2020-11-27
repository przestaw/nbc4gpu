#include <boost/program_options.hpp>
#include <error/exeception.h>
#include <iostream>
#include <optional>
#include <project_defines.h>
#include <utility>
// Boost.compute has a lot of errors detected by Clang/GCC
DIAGNOSTIC_PUSH
#include <boost/compute/algorithm/accumulate.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/core.hpp>
DIAGNOSTIC_POP

namespace options = boost::program_options;
namespace compute = boost::compute;

/**
 * ProgramParams struct for storing runtime params
 */
class ProgramParams {
public:
  ProgramParams(bool verbose) : verbose_(verbose) {}

  [[nodiscard]] bool isVerbose() const { return verbose_; }

private:
  bool verbose_;
};

/**
 * Function for arguments parsing
 * @param argc arguments count from stdin
 * @param argv arguments array
 */
inline std::optional<ProgramParams> parseParams(int argc, char *argv[]) {
  try {
    options::options_description desc{"Options"};
    desc.add_options()("help,h", "Help screen");
    desc.add_options()("verbose,v", "Verbose output");

    options::variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    /* Parse options */
    if (vm.count("help")) {
      std::cout << desc << '\n';
      return std::optional<ProgramParams>();
    } else {
      //      auto checkPresence = [&vm](const std::string &name) {
      //        if (!vm.count(name)) {
      //          throw nbc4gpu::error::missing_argument(name);
      //        }
      //      };

      bool verbose = vm.count("verbose");

      return ProgramParams(verbose);
    }
  } catch (const options::error &ex) {
    std::cerr << ex.what() << " !\n";
    return std::nullopt;
  }
}

double avg(const std::vector<double> &col) {
  // transfer the values to the device
  // automatic -> TODO : use command queue and wait for result
  compute::vector<double> device_vector = col;
  double sum =
      compute::accumulate(device_vector.begin(), device_vector.end(), 0);

  return sum / static_cast<double>(col.size());
}

/**
 * Main function
 * @return EXIT_SUCCESS on success
 */
int main(int argc, char *argv[]) {
  std::cout << "nbc4gpu : naive bayes classifier 4 GPU-s\n"
            << std::endl; // flush stream

  // temporary - config device
  compute::device device = boost::compute::system::default_device();
  compute::context context(device);
  compute::command_queue queue(context, device);

  auto params = parseParams(argc, argv);
  if (params.has_value()) {
    // verbose output
    if (params->isVerbose()) {
      // attach progress bar
      std::cout << "RUNNING\n";

      // print the device's name and platform
      std::cout << "hello from " << device.name();
      std::cout << " (platform: " << device.platform().name() << ")"
                << std::endl;

      std::vector<double> vec(50000);
      std::generate(vec.begin(), vec.end(), []() { return rand() % 50; });

      //std::for_each(vec.begin(), vec.end(), [](double val){ std::cout << val << ", " ;});
      std::cout << std::endl << "avg GPU = " << avg(vec);

      double sum = 0.0;
      sum = std::accumulate(vec.begin(), vec.end(), sum);
      std::cout << std::endl << "avg CPU = " << sum/static_cast<double>(vec.size());
    }
    return EXIT_SUCCESS;
  }
  return EXIT_FAILURE;
}
