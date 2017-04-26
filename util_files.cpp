
#include "util_files.h"


#include <dirent.h>
#include <sys/stat.h>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
using namespace std;


//-------------------------------------------------------------------------------------------------------------
void UTIL_Files::filesDir(const std::string &dirName,
                          std::vector< std::string > &files)
{
    //Se genera un vector con los nombre de fichero que tienen una determianda extension
    DIR *pDIR;
    struct dirent *entry;
    files.clear();
    if( pDIR=opendir(dirName.c_str()) )
    {
        while(entry = readdir(pDIR))
        {
            if ( (entry->d_name != std::string(".")) &&
                 (entry->d_name != std::string("..")) )
            {
                std::stringstream fileName;
                fileName << dirName << "/" << entry->d_name;
                files.push_back(fileName.str());
            }
        }
        closedir(pDIR);
    }
}

//-------------------------------------------------------------------------------------------------------------
void UTIL_Files::filesDir(const std::string &dirName,
                          const std::string &extension,
                          std::vector< std::string > &files)
{
    //Se genera un vector con los nombre de fichero que tienen una determianda extension
    DIR *pDIR;
    struct dirent *entry;
    files.clear();
    if( pDIR=opendir(dirName.c_str()) )
    {
        while(entry = readdir(pDIR))
        {
            if ( (entry->d_name != std::string(".")) &&
                 (entry->d_name != std::string("..")) )
            {
                std::string entryName = entry->d_name;
                std::string extEntry = entryName.substr(entryName.find_last_of(".")+1,entryName.length());
                if (extEntry == extension)
                {
                    std::stringstream fileName;
                    fileName << dirName << "/" << entry->d_name;
                    files.push_back(fileName.str());
                }
            }
        }
        closedir(pDIR);
    }
}

//-------------------------------------------------------------------------------------------------------------
void UTIL_Files::filesFilter(std::vector< std::string > &files,
                             std::vector<std::string> &filtersLabel)
{
    //Dejamos solo los ficheros que empiezan por alguna de las etiquetas
    std::vector<std::string> filtersFiles;
    std::vector< std::string >:: iterator itFiles = files.begin();
    for (; itFiles!=files.end(); itFiles++)
    {
        std::string fich = *itFiles;
        std::string fichName = fich.substr(fich.find_last_of("/")+1,fich.size());
        std::string labelFich = fichName.substr(0,fichName.find_first_of("_"));
        if (std::find(filtersLabel.begin(),filtersLabel.end(),labelFich) != filtersLabel.end())
            filtersFiles.push_back(fich);
    }
    files = filtersFiles;
}

//-------------------------------------------------------------------------------------------------------------
void UTIL_Files::readRelationsLabels(const std::string &fileRelationsLabels,
                                     std::map < std::string, int> &relationsLabels)
{
    std::ifstream file(fileRelationsLabels.c_str());
    if (file.is_open())
    {
        std::string line;
        while ( getline (file,line) )
        {
            std::stringstream sline;
            sline.str(line);
            std::string nameClass;
            sline >> nameClass;
            int idClass;
            sline >> idClass;

            relationsLabels[nameClass] = idClass;
        }
        file.close();
    }
}

//-------------------------------------------------------------------------------------------------------------
void UTIL_Files::readFilesTracks(const std::string &fileFilesTracks,
                                 std::map < std::string, std::vector<string> > &filesTracks)
{
    std::ifstream file(fileFilesTracks.c_str());
    if (file.is_open())
    {
        std::string line;
        while ( getline (file,line) )
        {
            std::stringstream sline;
            sline.str(line);
            std::string nameTrack;
            sline >> nameTrack;
            std::string nameFile;
            sline >> nameFile;

            filesTracks[nameTrack].push_back(nameFile);
            std::cout << nameTrack << "  " << nameFile << std::endl;
        }
        file.close();
    }
}

//-------------------------------------------------------------------------------------------------------------
void UTIL_Files::readLabels(const std::string &fileLabels,
                            std::vector < std::string> &labels)
{
    labels.clear();
    std::ifstream file(fileLabels.c_str());
    if (file.is_open())
    {
        std::string line;
        while ( getline (file,line) )
        {
            labels.push_back(line);
        }
        file.close();
    }
}

//-------------------------------------------------------------------------------------------------------------
void UTIL_Files::createDir(const std::string &dirName)
{
    DIR* dir = opendir(dirName.c_str());
    if (dir)
        closedir(dir);
    else
        mkdir(dirName.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

//-------------------------------------------------------------------------------------------------------------
std::string UTIL_Files::fileName(const std::string &pathFileName)
{
    return pathFileName.substr(pathFileName.find_last_of("/")+1,pathFileName.size());
}
