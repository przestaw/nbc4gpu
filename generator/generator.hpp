#ifndef GENERATOR_H
#define GENERATOR_H
#include <vector>
#include <cstdlib>
#include <algorithm>

template <typename ValueType> class Generator
{
public:
    Generator(){};
    using Column  = std::vector<ValueType>; //!< contains value for each row
    using Dataset = std::vector<Column>; //!< contains all rows for given class
    using ClassDataset = std::pair<ValueType, Dataset>;
    using FullDataset  = std::vector<ClassDataset>;
    void fillFullDataset(int numOfClasses, int numOfColumns, int numOfRecords, int seed, FullDataset &fullDataset){
        srand(seed);
        for(int i = 0; i < numOfClasses; i++){
            ClassDataset &classDataset = fullDataset.emplace_back(i, numOfColumns);
                for(auto &column: classDataset.second){
                    column.resize(numOfRecords);
                    std::generate(column.begin(), column.end(), []() { return rand() %100;});
                }
        }
    }
    void fillVector(int numOfElems, std::vector<ValueType> &record, int seed){
        srand(seed);
        record.resize(numOfElems);
        std::generate(record.begin(), record.end(), []() { return rand() %100;});
    }
};

#endif // GENERATOR_H
