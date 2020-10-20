#include <boost/program_options.hpp>
#include <error/exeception.h>
#include <iostream>
#include <utility>
#include <boost/compute/core.hpp>


using namespace boost::program_options;

/**
 * ProgramParams struct for storing runtime params
 */
class ProgramParams{
public:
  ProgramParams(bool verbose)
      : verbose_(verbose) {}

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
    options_description desc{"Options"};
    desc.add_options()("help,h", "Help screen");
    desc.add_options()("verbose,v", "Verbose output");

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    /* Parse options */
    if (vm.count("help")) {
      std::cout << desc << '\n';
      return std::optional<ProgramParams>();
    } else {
      auto checkPresence = [&vm](const std::string &name) {
        if (!vm.count(name)) {
          throw nbc4gpu::error::missing_argument(name);
        }
      };

      bool verbose = vm.count("verbose");

      return ProgramParams(verbose);
    }
  } catch (const boost::program_options::error &ex) {
    std::cerr << ex.what() << " !\n";
    return std::optional<ProgramParams>();
  }
}

/**
 * Main function
 * @return EXIT_SUCCESS on success
 */
int main(int argc, char *argv[]) {
  std::cout << "nbc4gpu : naive bayes classifier 4 GPU-s\n" << std::endl; // flush stream

  auto params = parseParams(argc, argv);
  if (params.has_value()) {
    boost::compute::device device = boost::compute::system::default_device();

    // verbose output
    if(params->isVerbose()){
      // attach progress bar
      std::cout << "RUNNING\n";

      // print the device's name and platform
      std::cout << "hello from " << device.name();
      std::cout << " (platform: " << device.platform().name() << ")" << std::endl;
    }

    return EXIT_SUCCESS;
  }
  return EXIT_FAILURE;
}
