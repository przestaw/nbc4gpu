#include <project_defines.h>
#include <boost/program_options.hpp>
#include <error/exeception.h>
#include <iostream>
#include <optional>
#include <utility>
// Boost.compute has a lot of errors detected by Clang/GCC
DIAGNOSTIC_PUSH
#include "src/classifier/learnColumn.h"
#include <boost/compute/algorithm/accumulate.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/core.hpp>
#include <boost/compute/lambda.hpp>
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
      // print the device's name and platform
      std::cout << "Device:  " << device.name()
                << "\n(platform: " << device.platform().vendor() << " - "
                << device.platform().version() << ")" << std::endl;
    }
    return 0;
  }
  return -1;
}
