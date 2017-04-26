#ifndef UTIL_FILES_H
#define UTIL_FILES_H

#include <vector>
#include <map>
#include <string>

class UTIL_Files
{
public:
    static void filesDir(const std::string &dirName,
                         const std::string &extension,
                         std::vector<std::string> &files);
    static void filesDir(const std::string &dirName,
                         std::vector<std::string> &files);
    static void filesFilter(std::vector<std::string> &files,
                            std::vector<std::string> &filtersLabel);
    static void readRelationsLabels(const std::string &fileRelationsLabels,
                                    std::map<std::string, int> &relationsLabels);
    static void readLabels(const std::string &fileLabels,
                           std::vector<std::string> &labels);
    static void createDir(const std::string &dirName);
    static std::string fileName(const std::string &pathFileName);
    static void readFilesTracks(const std::string &fileFilesTracks,
                                std::map < std::string, std::vector<std::string> > &filesTracks);
};

#endif // UTIL_FILES_H
