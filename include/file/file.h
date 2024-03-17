//
// Created by root on 3/10/24.
//

#ifndef GRAPH_SEARCH_FILE_H
#define GRAPH_SEARCH_FILE_H

#include <iostream>
#include <fstream>
#include <vector>

class WriteOpt {
public:
    template<class T>
    static void WriteCSVFile(const std::vector <std::vector <T> >&  data,const std::string & filePath);
//    static void WriteCSVFile(const std::vector<std::vector<std::string> >& data,const std::string & filePath)
    static void WriteTXTFile(std::vector <std::string>&  data,const std::string & filePath);
    static std::vector<std::vector<float>> ConvertTo2DVector(float* data, int dim, int max_elements);
};

#endif //GRAPH_SEARCH_FILE_H
