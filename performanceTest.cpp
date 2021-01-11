#include <classifier/GPUClassifier.hpp>
#include "generator.hpp"
#include <chrono>
using namespace std;
int main(){
    Generator<float> generator;
    int maxRecords = 10000000;
    int numOfTests = 50;
    for(int recordsNum = 1; recordsNum <=maxRecords; recordsNum*=10){
        double count = 0;
        for(int i=0;i<numOfTests;i++){
            nbc4gpu::GPUClassifier<float>::FullDataset data;
            generator.fillFullDataset(2, 10, recordsNum, i, data);
            auto classifier = nbc4gpu::GPUClassifier<float>(data);
            auto start = std::chrono::high_resolution_clock::now();
            classifier.learnClasses();
            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish - start;
            count += elapsed.count();
        }
        cout<<recordsNum<<" "<<count/numOfTests<<endl;
    }
    return 0;
}
