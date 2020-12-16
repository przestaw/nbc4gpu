//
// Created by przestaw on 20.10.2020.
//
#include <Defines.h>
DIAGNOSTIC_PUSH
#include <boost/compute.hpp>
DIAGNOSTIC_POP
#include "classifier.h"
#include <error/Execeptions.h>

using namespace boost::compute;

nbc4gpu::Classifier::Classifier() : field(0) {}

double nbc4gpu::Classifier::predict(const nbc4gpu::Classifier::Record& record) {
  UNUSED_VAL(record);
  throw nbc4gpu::error::RuntimeException("Not implemented");
}
