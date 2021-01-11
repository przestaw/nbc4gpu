#include <iostream>
#include "CPUCalculateClassP.hpp"
#include "CPUClassifier.hpp"
#include "CPULearnColumn.hpp"
#include "CPULearnClass.hpp"
#include "generator.hpp"
#include <chrono>
using namespace std;

int main()
{
    cout<<"NBC for CPU:"<<endl;
    Generator<float> generator;
    int maxRecords = 10000000;
    int numOfTests = 50;
    for(int recordsNum = 1; recordsNum <=maxRecords; recordsNum*=10){
        double count = 0;
        for(int i=0;i<numOfTests;i++){
            nbc4cpu::CPUClassifier<float>::FullDataset data;
            generator.fillFullDataset(2, 10, recordsNum, i, data);
            auto classifier = nbc4cpu::CPUClassifier<float>(data);
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
