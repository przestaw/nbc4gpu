//
// Created by przestaw on 20.10.2020.
//

#ifndef NBC4GPU_CLASSIFIER_H
#define NBC4GPU_CLASSIFIER_H

#include <vector>

namespace nbc4gpu {
  /**
   * Dummy class for project test
   */
  class Classifier {
    using LearnClassifier = int;  // TODO
    using Predictor = int;        // TODO

    using Attribute = double;
    using Record = std::vector<Attribute>; //!< row containing value in each column

    using Column = std::vector<Attribute>; //!< contains value for each row
    using Dataset = std::vector<Column>;   //!< contains all rows for given class
  public:
    /**
     * Constructor
     */
    Classifier();

    /**
     * Predict method
     * @param record record
     * @return probability
     */
    double predict(const Record &record);
  private:

    int field;
  };
} // namespace nbc4gpu

#endif // NBC4GPU_CLASSIFIER_H
