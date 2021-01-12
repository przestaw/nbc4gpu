#ifndef TEMPLATE_EXECEPTION_H
#define TEMPLATE_EXECEPTION_H

#include <boost/program_options/errors.hpp>
#include <iostream>
#include <stdexcept>

namespace nbc4gpu {
  namespace error {
    class Exception : public std::runtime_error {
    public:
      explicit Exception(const std::string &desc) : runtime_error(desc) {}
    };

    class RuntimeException : public Exception {
    public:
      explicit RuntimeException(const std::string &desc) : Exception(desc) {}
    };

    class MissingArgument : public boost::program_options::error,
                            public nbc4gpu::error::Exception {
    public:
      explicit MissingArgument(const std::string &desc)
          : boost::program_options::error(
              "Missing required argument " + desc
              + "\nPlease include argument with value"),
            nbc4gpu::error::Exception(
                "Missing required argument " + desc
                + "\nPlease include argument with value") {}
    };

    class MismatchedSize : public RuntimeException {
    public:
      explicit MismatchedSize(const std::string &desc)
          : RuntimeException("Mismatched size of arguments: " + desc) {}
    };

    class ZeroValuesProvided : public RuntimeException {
    public:
      explicit ZeroValuesProvided(const std::string &desc)
          : RuntimeException("Zero values provided for : " + desc) {}
    };

    class NotLearned : public RuntimeException {
    public:
      explicit NotLearned() : RuntimeException("Classifier is not learned") {}
    };
  } // namespace error
} // namespace nbc4gpu

#endif // TEMPLATE_EXECEPTION_H
