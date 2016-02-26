#include "util_bow.h"

#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "constants.h"

#include <fstream>
using namespace std;



//-------------------------------------------------------------------------------------------------------------
void UTIL_Bow::generateDictionaryDir(std::vector < std::string> &files,
                                     const std::string &fileDictName,
                                     int diccSize)
{
    //Mediante k-medias se establecen los tipos de descriptores que hay
    cv::TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
    int retries=1;
    int flags = cv::KMEANS_PP_CENTERS;
    cv::BOWKMeansTrainer bowTrainer(diccSize,tc,retries,flags);

    //Se lee cada fichero de descriptores
    std::vector < std::string>::iterator it_files = files.begin();
    for (; it_files != files.end(); it_files++)
    {
        std::cout << "adding " << *it_files << std::endl;

        cv::FileStorage file(*it_files,cv::FileStorage::READ);
        cv::Mat descriptors;
        file[Constants::DESCRIPTORS_LABEL] >> descriptors;

        bowTrainer.add(descriptors);

        descriptors.release();
        file.release();

        std::cout << "added " << *it_files << std::endl;
        std::cout << "added " << bowTrainer.getDescriptors().size() << " descriptos " << std::endl;
    }

    std::cout << "clustering descriptors..." << std::endl;

    cv::Mat dictionary;
    dictionary = bowTrainer.cluster();

    std::cout << "end cluster" << std::endl;

    //Guardamos el dicnaionario que nos permite saber a que tipo pertenece un desciptor dato
    std::ifstream fin(fileDictName.c_str());
    if (fin)
        remove(fileDictName.c_str());
    cv::FileStorage fileDict(fileDictName,cv::FileStorage::WRITE);
    fileDict << Constants::DICTIONARY_LABEL << dictionary;

    std::cout << "generate dictironary " << fileDictName << std::endl;

    dictionary.release();
    fileDict.release();

}

