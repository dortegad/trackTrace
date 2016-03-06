#ifndef MWTRACKTRACE_H
#define MWTRACKTRACE_H

#include <QMainWindow>

#include <dialoggeneratedb.h>

#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>


namespace Ui {
class MWTrackTrace;
}

class MWTrackTrace : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MWTrackTrace(QWidget *parent = 0);
    ~MWTrackTrace();


private slots:
    void on_pbDictionary_clicked();

    void on_pBGenerateDescriptorsBOW_clicked();

    void on_pBVerboseGenerateDescriptorsBOW_clicked();

    void on_pBGenerateSVM_clicked();

    void on_pBTestSVM_clicked();

    void on_pBAll_clicked();

    void on_pBGenerateKeyPoints_clicked();

    void on_pBThresholdSift_clicked();

    void on_pBAutoDetectSift_clicked();

    void on_pBgenerateDB_clicked();

private:
    Ui::MWTrackTrace *ui;

    DialogGenerateDB *generateDB;

    cv::PCA pca;

public :

    //DICCTIONARY
    cv::BOWImgDescriptorExtractor *bow;
    void generateDictionaryDir(std::vector<std::string> &files,
                               const std::string &fileDictName);
    void generateDictionaryDirs();

    //BOW
    std::vector < std::string> filesForBOW;
    std::vector < std::string>::iterator it_filesForBow;
    bool vervoseGenerateDesBOW;
    bool initGenerateDescriptorsBOW;
    void readDictionary(const std::string &fileNameDictionary);
    void bowCompute(const std::string &fileName);
    void saveSamples(const std::string &fileName,
                     cv::Mat &img,
                     std::vector<cv::KeyPoint> &keypoints,
                     std::vector< std::vector< int > > &pointIdxsOfClusters);
    void saveBow(const std::string &fileName,
                cv::Mat &img,
                std::vector<cv::KeyPoint> &keypoints,
                std::vector< std::vector< int > > &pointIdxsOfClusters);


    //PCA
    void generatePCA(cv::Mat &labelMat,
                     cv::Mat &dataMat);

    //SVD
    void readDataTrainSVMBow(const std::string &dirData,
                              const std::string &fileRelationsLabels,
                              cv::Mat &labelMat,
                              cv::Mat &dataMat);
    void readDataTrainSVMGray(const std::string &dirData,
                              const std::string &fileRelationsLabels,
                              cv::Mat &labelMat,
                              cv::Mat &dataMat);
    void generateSVM();

    void descriptorBow(const std::string &fileName,
                       cv::Mat &descBow);

    void descriptorGray(const std::string &fileName,
                        cv::Mat &descGray);

    void all();

    void filesDir(const std::string &dirName, const std::string &extension, std::vector<std::string> &files);
    void filesFilter(std::vector<std::string> &files, std::vector<std::string> &filtersLabel);
    void readRelationsLabels(const std::string &fileRelationsLabels, std::map<std::string, int> &relationsLabels);
    void readLabels(const std::string &fileLabels, std::vector<std::string> &labels);
};

#endif // MWTRACKTRACE_H
