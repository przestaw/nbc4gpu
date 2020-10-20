#ifndef TEMPLATE_EXECEPTION_H
#define TEMPLATE_EXECEPTION_H

#include <boost/program_options/errors.hpp>
#include <iostream>
#include <stdexcept>

namespace nbc4gpu {
  namespace error {
    class exception : public std::runtime_error {
    public:
      explicit exception(const std::string &desc) : runtime_error(desc) {}
    };

    class runtime_exception : public exception {
    public:
      explicit runtime_exception(const std::string &desc) : exception(desc) {}
    };

    class missing_argument : public boost::program_options::error,
                             public nbc4gpu::error::exception {
    public:
      explicit missing_argument(const std::string &desc)
          : boost::program_options::error(
              "Missing required argument " + desc
              + "\nPlease include argument with value"),
            nbc4gpu::error::exception("Missing required argument " + desc
                                  + "\nPlease include argument with value") {}
    };
  } // namespace error
} // namespace bgm

#endif // TEMPLATE_EXECEPTION_H
