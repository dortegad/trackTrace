#ifndef UTIL_BOW_H
#define UTIL_BOW_H

#include <vector>
#include <string>
using namespace std;

class UTIL_Bow
{
public:
    static void generateDictionaryDir(std::vector<std::string> &files,
                                      const std::string &fileDictName,
                                      int diccSize);
};

#endif // UTIL_BOW_H
