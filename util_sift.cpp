#include "util_sift.h"


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "constants.h"

#include <iostream>
#include <map>
using namespace std;

//-------------------------------------------------------------------------------------------------------------
void UTIL_Sift::createKeyPoints(const cv::Mat &img,
                                int keyPointSize,
                                int density,
                                std::vector < cv::KeyPoint > &keyPoints)
{
    for (int i=density; i<img.cols; i+=density)
    {
        for (int j=density; j<img.rows; j+=density)
        {
            cv::KeyPoint kp(cv::Point2f(i,j),keyPointSize);
            keyPoints.push_back(kp);
        }
    }
}

//-------------------------------------------------------------------------------------------------------------
void UTIL_Sift::siftsGrid(const cv::Mat &img,
                          int keyPointSize,
                          int density,
                          std::vector < cv::KeyPoint > &keyPoints,
                          cv::Mat &descriptors)
{
    createKeyPoints(img, keyPointSize, density, keyPoints);
    computeSift(img,
                keyPoints,
                descriptors);
    std::cout << "generate grid sift descriptors" << std::endl;
}

//-------------------------------------------------------------------------------------------------------------
void UTIL_Sift::computeSift(const cv::Mat &img,
                            std::vector < cv::KeyPoint > &keyPoints,
                            cv::Mat &descriptors)
{

    cv::Ptr< cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
    sift->compute(img,keyPoints,descriptors);
    std::cout << "generate sift keypoints" << std::endl;
}

//-------------------------------------------------------------------------------------------------------------
void UTIL_Sift::siftsAutodetec(const cv::Mat &img,
                               std::vector < cv::KeyPoint >  &keyPoints,
                               cv::Mat &descriptors)
{
    std::vector<float> scores;
    computeAndDetectSift(img,
                         keyPoints,
                         descriptors,
                         scores);
    std::cout << "autodetect sift descriptors" << std::endl;
}


//-------------------------------------------------------------------------------------------------------------
void UTIL_Sift::filterDescriptorsSift(cv::Mat &descriptors,
                                      float threshold)
{
    //Se generea el score (norma_2) de cada sift
    std::vector <float> scores;
    scoresDescriptorsSift(descriptors,scores);

    //Solo los sfit cuyo score supere el umbral permanecerán
    cv::Mat filterDescriptors;
    for (int i=0; i<scores.size(); i++)
    {
        float score = scores[i];
        if (score > threshold)
            filterDescriptors.push_back(descriptors.row(i));
    }
    descriptors = filterDescriptors;
    std::cout << "filter descriptors to " << threshold << std::endl;
}


//-------------------------------------------------------------------------------------------------------------
void UTIL_Sift::computeAndDetectSift(const cv::Mat &img,
                                     std::vector < cv::KeyPoint > &keyPoints,
                                     cv::Mat &descriptors,
                                     std::vector<float> &scores)
{
    //Dejamos a sift qeu detecte los puntos
    cv::Ptr< cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create(2000,3,0.04,10,1.6);
    sift->detectAndCompute(img,cv::Mat(),keyPoints,descriptors);

    //Extraemos en un vector los scores almacenados en el response de cada keypoint
    scores.clear();
    int numKeyPoints = keyPoints.size();
    for (int i=0; i<numKeyPoints; i++)
    {
        float score = keyPoints[i].response;
        scores.push_back(score);
    }
}

//-------------------------------------------------------------------------------------------------------------
void UTIL_Sift::scoresDescriptorsSift(cv::Mat &descriptors,
                                      std::vector<float> &scores)
{
    //Calculamos la norma de cada sift
    scores.clear();
    for (int i=0; i<descriptors.rows; i++)
    {
        float score = cv::norm(descriptors.row(i));
        //std::cout << score << std::endl;
        scores.push_back(score);
    }
    std::cout << "generate scores sift descriptors" << std::endl;
}

//-------------------------------------------------------------------------------------------------------------
void UTIL_Sift::sifts(const cv::Mat &img,
                      int keyPointSize,
                      int density,
                      bool rSift,
                      bool filterDescriptors,
                      float thresholdFilter,
                      cv::Mat &descriptors,
                      std::vector < cv::KeyPoint > &keyPoints)
{
    //Si calculan los sift en regilla de la imagen
    siftsGrid(img,
              keyPointSize,
              density,
              keyPoints,
              descriptors);

    //Si se selecciona filtraran los keypoint que tenga bajos scores (NORM_2)
    if (filterDescriptors)
        filterDescriptorsSift(descriptors,thresholdFilter);

    //Si se selecciona se transformaran los sift con rootsift
    if (rSift)
        rootSift(descriptors);
}

//-------------------------------------------------------------------------------------------------------------
void UTIL_Sift::sifts(std::vector<std::string> &files,
                      int keyPointSize,
                      int density,
                      bool rSift,
                      bool filterDescriptors,
                      float thresholdFilter,
                      cv::Mat &descriptors)
{
    //Se generan los sift de cada imagen y se concatenan todos
    std::vector < std::string>::iterator it_files = files.begin();
    for (; it_files != files.end(); it_files++)
    {
        std::string file = *it_files;
        cv::Mat image = cv::imread(file,cv::IMREAD_GRAYSCALE);

        cv::Mat descSift;
        std::vector <cv::KeyPoint> keyPoints;
        UTIL_Sift::sifts(image,
                         keyPointSize,
                         density,
                         rSift,
                         filterDescriptors,
                         thresholdFilter,
                         descSift,
                         keyPoints);
        cv::vconcat(descriptors,descSift,descriptors);

        image.release();
    }
}

//-------------------------------------------------------------------------------------------------------------
void UTIL_Sift::siftsRegions(const cv::Mat &img,
                      int keyPointSize,
                      int density,
                      bool rSift,
                      bool filterDescriptors,
                      float thresholdFilter,
                      cv::Mat &descriptors,
                      std::vector < cv::KeyPoint > &keyPoints)
{
    //Si calculan los sift de toda la imagen
    sifts(img,
         keyPointSize,
         density,
         rSift,
         filterDescriptors,
         thresholdFilter,
         descriptors,
         keyPoints);

    //Se recorren los keypoints creando un indice organizandolos por regiones
    std::map<int, std::vector<int> > orden;
    cv::Rect reg_01(0,0,img.cols/2,img.rows/2);
    cv::Rect reg_02(img.cols/2,0,img.cols,img.rows/2);
    cv::Rect reg_03(0,img.rows/2,img.cols/2,img.rows);
    int numKeyPoints = keyPoints.size();
    for (int i=0; i<numKeyPoints; i++)
    {
        cv::Point2f point = keyPoints[i].pt;
        if (reg_01.contains(point))
            orden[0].push_back(i);
        else if (reg_02.contains(point))
            orden[1].push_back(i);
        else if (reg_03.contains(point))
            orden[2].push_back(i);
        else
            orden[3].push_back(i);
    }


    //Se recolocan los keyPoints y los descriptors para ordenarlos por secciones
    cv::Mat newDescriptors;
    std::vector < cv::KeyPoint > newKeyPoints;
    std::map<int, std::vector<int> >::iterator itOrden;
    for (itOrden=orden.begin(); itOrden!=orden.end(); itOrden++)
    {
        std::vector<int> ordenRegion = (*itOrden).second;
        for (int i=0; i<ordenRegion.size(); i++)
        {
            int pos = ordenRegion[i];
            newKeyPoints.push_back(keyPoints[pos]);
            cv::hconcat(newDescriptors,descriptors.row(i),newDescriptors);
        }
    }
    keyPoints = newKeyPoints;
    descriptors = newDescriptors;


    //Presentamos una imagen con los keypoint colocados
    cv::Mat imgAux = img.clone();
    for (itOrden=orden.begin(); itOrden!=orden.end(); itOrden++)
    {
        std::vector<int> ordenRegion = (*itOrden).second;
        for (int i=0; i<ordenRegion.size(); i++)
        {
            int pos = ordenRegion[i];
            cv::Point2f point = keyPoints[pos].pt;
            int region = (*itOrden).first;
            cv::Scalar color;
            if (region == 0)
                color = cv::Scalar(0,0,255);
            else if (region == 1)
                color = cv::Scalar(0,255,0);
            else if (region == 2)
                color = cv::Scalar(255,0,0);
            else
                color = cv::Scalar(255,255,0);

            cv::rectangle(imgAux,point,point,color);

        }
    }
    cv::resize(imgAux,imgAux,cv::Size(500,500));
    cv::imshow("SiftRegiones",imgAux);

}

//-------------------------------------------------------------------------------------------------------------
void UTIL_Sift::rootSift(cv::Mat &descriptors)
{
    cv::Mat sumVec;
    descriptors = cv::abs(descriptors);
    cv::reduce(descriptors,sumVec,1 /*COLUMNAS*/,CV_REDUCE_SUM,CV_32FC1);
    for (unsigned int row=0; row<descriptors.rows; row++)
    {
        int offset = row*descriptors.cols;
        for (unsigned int col=0; col<descriptors.cols; col++)
        {
            descriptors.at<float>(offset+col) = std::sqrt(descriptors.at<float>(offset+col)) / sumVec.at<float>(row); //L1-NORMALIZE
        }
    }
    std::cout << "generate root-sift descriptors" << std::endl;
}

//-------------------------------------------------------------------------------------------------------------
void UTIL_Sift::saveDescriptorsSiftPCADir(std::vector<std::string> &files,
                                          int keyPointSize,
                                          int density,
                                          bool rSift,
                                          bool filterDescriptors,
                                          float thresholdFilter)
{
    //Se generan los sift de cada imagen y se concatenan todos para generar su pca
    cv::Mat allDescriptors;
    UTIL_Sift::sifts(files,
                     keyPointSize,
                     density,
                     rSift,
                     filterDescriptors,
                     thresholdFilter,
                     allDescriptors);

    //Calculamos la matriz pca con todos los sift almacenados
    int num_components = 64; //La mitad ya uqe la dimension del sift es 128
    cv::PCA pca(allDescriptors, cv::Mat(), CV_PCA_DATA_AS_ROW, num_components);
    allDescriptors.release();
    std::cout << "generate PCA-sift descriptors" << std::endl;

    //Generamos los sift de cada imagen, pryectamos PCA y grabamos un fichero paralelo con los sift
    std::vector < std::string>::iterator it_files = files.begin();
    for (; it_files != files.end(); it_files++)
    {
        std::string file = *it_files;
        cv::Mat image = cv::imread(file,cv::IMREAD_GRAYSCALE);

        cv::Mat descriptors;
        std::vector <cv::KeyPoint> keyPoints;
        UTIL_Sift::sifts(image,
                         keyPointSize,
                         density,
                         rSift,
                         filterDescriptors,
                         thresholdFilter,
                         descriptors,
                         keyPoints);

        pca.project(descriptors,descriptors);

        std::stringstream descriptorsFileName;
        descriptorsFileName << file.substr(0,file.find_last_of('.')) << "." << Constants::DESCRIPTOR_EXT;

        cv::FileStorage fileDescriptors(descriptorsFileName.str(), cv::FileStorage::WRITE);
        fileDescriptors << Constants::DESCRIPTORS_LABEL << descriptors;
        fileDescriptors.release();

        std::cout << descriptorsFileName.str() << std::endl;

        descriptors.release();
        image.release();
    }
}

//-------------------------------------------------------------------------------------------------------------
void UTIL_Sift::saveDescriptorsSiftDir(std::vector<std::string> &files,
                                       int keyPointSize,
                                       int density,
                                       bool rSift,
                                       bool filterDescriptors,
                                       float thresholdFilter)
{
    //Generamos los sif de cada imagen y grabamos un fichero paralelo con los sift
    std::vector < std::string>::iterator it_files = files.begin();
    for (; it_files != files.end(); it_files++)
    {
        std::string file = *it_files;
        cv::Mat image = cv::imread(file,cv::IMREAD_GRAYSCALE);

        cv::Mat descriptors;
        std::vector <cv::KeyPoint> keyPoints;
        UTIL_Sift::sifts(image,
                         keyPointSize,
                         density,
                         rSift,
                         filterDescriptors,
                         thresholdFilter,
                         descriptors,
                         keyPoints);

        std::stringstream descriptorsFileName;
        descriptorsFileName << file.substr(0,file.find_last_of('.')) << "." << Constants::DESCRIPTOR_EXT;

        cv::FileStorage fileDescriptors(descriptorsFileName.str(), cv::FileStorage::WRITE);
        fileDescriptors << Constants::DESCRIPTORS_LABEL << descriptors;
        fileDescriptors.release();

        std::cout << descriptorsFileName.str() << std::endl;

        descriptors.release();
        image.release();
    }
}


