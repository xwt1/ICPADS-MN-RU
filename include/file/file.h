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
    static void WriteIndexLogFile(const std::string log_info_data_path,const std::string &data_file_path,const std::string &index_file_path,const int &dim, const int &vector_num);
};

class ReadOpt{
public:
    template<typename T>
    static bool ReadFvecsFileIntoArray(const std::string& filePath, void* p, int& totalVectors, const int& dim, const int &maxElements) {
        std::ifstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filePath << std::endl;
            return false;
        }
        int vectorDim = -1;
        // 尝试读取第一个向量的维度
        file.read(reinterpret_cast<char*>(&vectorDim), sizeof(int));
        if (vectorDim <= 0 || vectorDim != dim) {
            std::cerr << "Invalid vector dimension or dimension exceeds maximum allowed size." << std::endl;
            return false;
        }

        T* array = reinterpret_cast<T*>(p); // 将void*转换为T*，以便按T类型读取和存储数据

        // 重新定位到文件开头
        file.seekg(0, std::ios::beg);

        int vectorCount = 0;
        while (true) {
            if (!file.read(reinterpret_cast<char*>(&vectorDim), sizeof(int))) {
                // 文件结束或读取错误
                break;
            }
            if (vectorDim != dim) {
                std::cerr << "dim is not consistent" << std::endl;
                return false;
            }
            if (vectorCount >= maxElements) {
                std::cout << "Array size exceeded, but you can use the data that has been loaded" << std::endl;
                totalVectors = vectorCount;
                file.close();
                return true;
            }
            if (!file.read(reinterpret_cast<char*>(array + vectorCount * dim), sizeof(T) * dim)) {
                // 数据读取不完整
                std::cerr << "Incomplete data read." << std::endl;
                return false;
            }
            vectorCount++;
        }

        totalVectors = vectorCount;
        file.close();
        return true;
    }
};

#endif //GRAPH_SEARCH_FILE_H
