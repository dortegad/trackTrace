#include "mwtracktrace.h"
#include "ui_mwtracktrace.h"

#include "dialogshowbow.h"
#include "dialogshowsift.h"

#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

#include "util_sift.h"
#include "util_bow.h"
#include "constants.h"
//#include <util_files.h>

#include <dirent.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;

//-------------------------------------------------------------------------------------------------------------
MWTrackTrace::MWTrackTrace(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MWTrackTrace)
{
    ui->setupUi(this);

    generateDB = new DialogGenerateDB(this);

    this->bow = NULL;

    this->initGenerateDescriptorsBOW = false;
}

//-------------------------------------------------------------------------------------------------------------
MWTrackTrace::~MWTrackTrace()
{
    delete generateDB;
    delete ui;
}


//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::filesDir(const std::string &dirName,
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
void MWTrackTrace::filesFilter(std::vector< std::string > &files,
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
void MWTrackTrace::readRelationsLabels(const std::string &fileRelationsLabels,
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
void MWTrackTrace::readLabels(const std::string &fileLabels,
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
void MWTrackTrace::on_pbDictionary_clicked()
{
    //Directorio donde estan los ficheros de descriptores para crear el diccionario
    std::stringstream dirDiccionary;
    dirDiccionary << ui->lprincipalDir->text().toStdString() << this->ui->lDiccionaryDir->text().toStdString();

    //Se leen los nombres de todos los ficheros de descriptores sift del directorio
    std::vector < std::string> files;
    filesDir(dirDiccionary.str(), Constants::DESCRIPTOR_EXT, files);

    //Leemos los estiquetas que tienen que tener los ficheros con los que vamos a entrenar
    std::stringstream fileLabelsToDicctionary;
    fileLabelsToDicctionary << this->ui->lprincipalDir->text().toStdString() << ui->lLabelsToDicctionary->text().toStdString();
    std::vector <std::string> labelsToDicctionary;
    readLabels(fileLabelsToDicctionary.str(),labelsToDicctionary);

    //Dejamos solo los ficheros que se indican en labelsForBow (Esto permite que se entrene el bow solo
    //con un determinado tipo)
    filesFilter(files,labelsToDicctionary);

    //Ruta y nombre del fichero diccionario
    std::stringstream fileDictName;
    fileDictName << dirDiccionary.str() << "/" << "dictionary" << "." << Constants::DICTIONARY_EXT;
    std::cout << "generated dictionary : " << fileDictName.str() <<  std::endl;

    //Generamos el diccionario
    UTIL_Bow::generateDictionaryDir(files,
                          fileDictName.str(),
                          this->ui->ldictionarySize->text().toInt());
}


//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::readDictionary(const std::string &fileNameDictionary)
{
    if (this->bow != NULL)
        delete this->bow;

    // declare BOWImgDescriptorExtractor
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::SIFT::create();
    this->bow = new cv::BOWImgDescriptorExtractor( extractor, matcher );

    // load vocabulary data
    cv::Mat vocabulary;
    cv::FileStorage fs( fileNameDictionary, cv::FileStorage::READ);
    fs[Constants::DICTIONARY_LABEL ] >> vocabulary;
    fs.release();
    if (vocabulary.empty())
        return;

    // Set the vocabulary
    this->bow->setVocabulary( vocabulary );
}


//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::saveBow(const std::string &fileName,
                           cv::Mat &img,
                           std::vector<cv::KeyPoint> &keypoints,
                           std::vector< std::vector< int > > &pointIdxsOfClusters)
{
    cv::Mat imgBow = cv::Mat_<uchar>::zeros(img.rows,img.cols);
    int numCluster = pointIdxsOfClusters.size();
    for (int i=0; i<numCluster; i++)
    {
        int numKeypointType = pointIdxsOfClusters[i].size();
        if (numKeypointType > 0)
        {
            for (int j=0; j<numKeypointType; j++)
            {
                cv::KeyPoint keyPoint = keypoints[pointIdxsOfClusters[i][j]];
                cv::Point point = keyPoint.pt;
                cv::Point point_1 = cv::Point(point.x-1,point.y-1);
                cv::Point point_2 = cv::Point(point.x+1,point.y+1);
                cv::rectangle(imgBow,point_1,point_2,cv::Scalar(i),-1,1);
            }
        }
    }

    std::stringstream dirBows;
    dirBows << this->ui->lprincipalDir->text().toStdString() << "/Bows";
    DIR* dir = opendir(dirBows.str().c_str());
    if (dir)
        closedir(dir);
    else
        mkdir(dirBows.str().c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    dirBows << fileName.substr(fileName.find_last_of("/"),fileName.length());
    std::cout << dirBows.str().c_str() << std::endl;
    cv::imwrite(dirBows.str().c_str(),imgBow);
    //    cv::imshow("Prueba",imgBow);
    //    cv::waitKey();
}

//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::saveSamples(const std::string &fileName,
                               cv::Mat &img,
                               std::vector<cv::KeyPoint> &keypoints,
                               std::vector< std::vector< int > > &pointIdxsOfClusters)
{
    int numCluster = pointIdxsOfClusters.size();
    for (int i=0; i<numCluster; i++)
    {
        cv::Mat imgBow = img.clone();
        int numKeypointType = pointIdxsOfClusters[i].size();
        if (numKeypointType > 0)
        {
            for (int j=0; j<numKeypointType; j++)
            {
                cv::KeyPoint keyPoint = keypoints[pointIdxsOfClusters[i][j]];
                cv::Point point = keyPoint.pt;
                cv::Point point_1 = cv::Point(point.x-1,point.y-1);
                cv::Point point_2 = cv::Point(point.x+1,point.y+1);
                cv::rectangle(imgBow,point_1,point_2,cv::Scalar(255,255,255),-1,1);
            }
        }
        std::stringstream dirSamples;
        dirSamples << this->ui->lprincipalDir->text().toStdString() << "/Samples/";
        DIR* dir = opendir(dirSamples.str().c_str());
        if (dir)
            closedir(dir);
        else
            mkdir(dirSamples.str().c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        std::stringstream dirSamplesNum;
        dirSamplesNum << dirSamples.str().c_str() << i;
        dir = opendir(dirSamplesNum.str().c_str());
        if (dir)
            closedir(dir);
        else
            mkdir(dirSamplesNum.str().c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        dirSamplesNum << fileName.substr(fileName.find_last_of("/"),fileName.length());
        std::cout << dirSamplesNum.str().c_str() << std::endl;
        cv::imwrite(dirSamplesNum.str().c_str(),imgBow);
        //        cv::imshow("Prueba",imgBow);
        //        cv::waitKey();
    }

}

//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::bowCompute(const string &fileName)
{
    cv::Mat descBOW;
    descriptorBow(fileName, descBOW);

    std::stringstream fileNameDescriptorBow;
    fileNameDescriptorBow << fileName.substr(0,fileName.find_first_of(".")+1) << Constants::DESCRIPTOR_BOW_EXT;
    cv::FileStorage fileDescriptors(fileNameDescriptorBow.str(), cv::FileStorage::WRITE);
    fileDescriptors << Constants::DESCRIPTORS_BOW_LABEL << descBOW;
    fileDescriptors.release();

    std::cout <<  "generate bow histogram " << fileNameDescriptorBow.str() << std::endl;

    descBOW.release();

    //    if (vervoseGenerateDesBOW)
    //    {
    //        DialogShowBOW *dialogBOWW = new DialogShowBOW(this);
    //        dialogBOWW->showBOW(img,
    //                            keypoints,
    //                            pointIdxsOfClusters);
    //    }
    //    else
    //    {
    //        if (this->ui->cBGenerateBowSamples->isChecked())
    //        {
    //            saveSamples(fileName,
    //                        img,
    //                        keypoints,
    //                        pointIdxsOfClusters);
    //        }
    //        if (this->ui->cBGenerateBows->isChecked())
    //        {
    //            saveBow(fileName,
    //                    img,
    //                    keypoints,
    //                    pointIdxsOfClusters);
    //        }
    //        else
    //        {

    //        }
    //    }
}


//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::on_pBGenerateDescriptorsBOW_clicked()
{
    if (!this->initGenerateDescriptorsBOW)
    {
        //Se lee el diccionario
        std::stringstream diccionaryFile;
        diccionaryFile << ui->lprincipalDir->text().toStdString() << ui->lDiccionaryFile->text().toStdString();
        readDictionary(diccionaryFile.str());

        //Se leen lo ficheros de imagen que se van a traducir con el diccionario
        std::stringstream dirForBOW;
        dirForBOW << ui->lprincipalDir->text().toStdString() << ui->limgsDirTest->text().toStdString();
        filesDir(dirForBOW.str(), Constants::IMAGE_EXT , this->filesForBOW);
        this->it_filesForBow = this->filesForBOW.begin();

        this->initGenerateDescriptorsBOW = true;
    }

    this->vervoseGenerateDesBOW = false;

    this->ui->pBVerboseGenerateDescriptorsBOW->setEnabled(false);

    //Se reccorren los fichero de imagen que se van a traduccir con el BOW
    for (; it_filesForBow != this->filesForBOW.end(); it_filesForBow++)
        bowCompute(*it_filesForBow);

    this->ui->pBVerboseGenerateDescriptorsBOW->setEnabled(true);
}

//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::on_pBVerboseGenerateDescriptorsBOW_clicked()
{
    //    if (!this->initGenerateDescriptorsBOW)
    //    {
    //        std::stringstream diccionaryFile;
    //        diccionaryFile << ui->lprincipalDir->text().toStdString() << ui->lDiccionaryFile->text().toStdString();

    //        readDictionary(diccionaryFile.str());

    //        std::stringstream dirForBOW;
    //        dirForBOW << ui->lprincipalDir->text().toStdString() << ui->limgsDirTest->text().toStdString();
    //        filesDir(dirForBOW.str(), MWTrackTrace::IMAGE_EXT , this->filesForBOW);
    //        this->it_filesForBow = this->filesForBOW.begin();

    //        this->initGenerateDescriptorsBOW = true;
    //    }

    //    this->vervoseGenerateDesBOW = true;

    //    bowCompute();

    //    this->it_filesForBow++;
}


//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::readDataTrainSVMBow(const std::string &dirData,
                                       const std::string &fileRelationsLabels,
                                       cv::Mat &labelMat,
                                       cv::Mat &dataMat)
{
    //Leemos el fichero con la realacion entre nombre del fichero y numero de etiqueta
    std::map < std::string, int> relationsLabel;
    readRelationsLabels(fileRelationsLabels, relationsLabel);

    //Leemos todos los ficheros de descriptores BOW y los indexamos por etiquetas (la primera cadena hasta "_")
    std::vector < std::string> filesDescriptorsBOW;
    filesDir(dirData, Constants::DESCRIPTOR_BOW_EXT, filesDescriptorsBOW);
    int numFilesDescriptorsBOW = filesDescriptorsBOW.size();

    std::map < string, std::vector <std::string> > filesByLabel;
    std::vector < std::string>::iterator it_filesDescriptorsBOW = filesDescriptorsBOW.begin();
    for (; it_filesDescriptorsBOW != filesDescriptorsBOW.end(); it_filesDescriptorsBOW++)
    {
        std::string fileName = *it_filesDescriptorsBOW;
        std::string labelDataFile = fileName.substr(fileName.find_last_of("/")+1,fileName.size());
        labelDataFile = labelDataFile.substr(0,labelDataFile.find_first_of("_"));

        //Cada fichero es de la clase que indica la cadena previa de su nombre, para juntar clases habria que meter aqui
        //un diccionatio y juntar los ficheros que queramos que tengan la misma etiqueta
        filesByLabel[labelDataFile].push_back(fileName);
    }

    //Generamos la matriz de etiquetas y la de datos
    labelMat = cv::Mat(numFilesDescriptorsBOW, 1, CV_32SC1, cv::Scalar(0));
    dataMat = cv::Mat(numFilesDescriptorsBOW, this->ui->ldictionarySize->text().toInt(), CV_32FC1);
    std::map < string, std::vector <std::string> >::iterator it_filesByLabel = filesByLabel.begin();
    int pos = 0;
    int idOfLabel = 0;
    for (; it_filesByLabel != filesByLabel.end(); it_filesByLabel++)
    {
        std::string label = it_filesByLabel->first;
        int numDataOfLabel = it_filesByLabel->second.size();
        for (int i=0; i<numDataOfLabel; i++)
        {
            //            std::cout << pos << std::endl;

            //si entre lasrelacion de etiqeutas de encontramos la etiqueta actual podremos el id de
            //la clase si no dejaremos un 0 para las clases que aparezcan en el fch de relaciones
            if (relationsLabel.find(label) != relationsLabel.end())
                labelMat.at<__int32_t>(pos,0) = relationsLabel[label];

            cv::FileStorage file( it_filesByLabel->second[i] ,cv::FileStorage::READ);
            cv::Mat descriptor;
            file[Constants::DESCRIPTORS_BOW_LABEL] >> descriptor;
            descriptor.copyTo(dataMat.row(pos));
            descriptor.release();

            //            std::cout << dataMat.row(pos) << std::endl;

            pos++;
        }
        //        std::cout << idOfLabel << std::endl;

        idOfLabel++;
    }
}

//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::generateSVM()
{
    cv::Mat labelMat;
    cv::Mat dataMat;

    std::stringstream dirDescriptorBow;
    dirDescriptorBow << this->ui->lprincipalDir->text().toStdString() << ui->ldescriptorsBowDir->text().toStdString();
    std::stringstream labelsFileName;
    labelsFileName << this->ui->lprincipalDir->text().toStdString() << ui->lLabelsFileName->text().toStdString();
    if (this->ui->cBGrayImgs->isChecked())
    {
        readDataTrainSVMGray(dirDescriptorBow.str(),
                             labelsFileName.str(),
                             labelMat,
                             dataMat);
    }
    else
    {
        readDataTrainSVMBow(dirDescriptorBow.str(),
                            labelsFileName.str(),
                            labelMat,
                            dataMat);
    }


    generatePCA(labelMat,dataMat);

    // Set up training data
    //    int labels[4] = {1, -1, -1, -1};
    //    cv::Mat labelsMat(4, 1, CV_32SC1, labels);

    //    float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
    //    cv::Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::LINEAR);
    svm->setGamma(3);

    cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(dataMat, cv::ml::ROW_SAMPLE, labelMat);

    svm->train(tData);

    std::stringstream svmFileName;
    svmFileName << dirDescriptorBow.str().c_str()
                << "/svm." << Constants::DESCRIPTOR_SVM_EXT;

    std::string fileNameSVM = svmFileName.str();
    svm->save(fileNameSVM);

    svm.release();

    std::cout << "tran svm " << fileNameSVM << std::endl;


    //    svm->train( trainingDataMat , cv::ml::ROW_SAMPLE , labelsMat );

    //    // ...
    //    Mat query; // input, 1channel, 1 row (apply reshape(1,1) if nessecary)
    //    Mat res;   // output
    //    svm->predict(query, res);


    //    // Set up SVM's parameters
    //    cv::ml::SVM svm;
    //    svm.t
    //    cv::ml::SVM::Params::Params p params;
    //    params.svmType    = cv::ml::SVM::C_SVC;
    //    params.kernelType = cv::ml::SVM::LINEAR;
    //    params.termCrit   = cv::TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6);

    //    // Train the SVM
    //    cv::Ptr<cv::ml::SVM> svm = cv::ml::StatModel::train<cv::ml::SVM>(trainingDataMat, ROW_SAMPLE, labelsMat, params);
}


//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::generatePCA(cv::Mat &labelMat,
                               cv::Mat &dataMat)
{
    int num_components = 3;
    cv::PCA pca(dataMat, cv::Mat(), CV_PCA_DATA_AS_ROW, num_components);

    //Nos recorremos todas las filas de la matric calculando su pca
    std::stringstream pcaFileName;
    pcaFileName << ui->lprincipalDir->text().toStdString()
                << ui->lfilePCA->text().toStdString()
                << "." << Constants::PCA_EXT;
    std::ofstream filePCA(pcaFileName.str().c_str(), std::ofstream::out);
    for (int i=0; i<dataMat.rows; i++)
    {
        cv::Mat row = dataMat.row(i);
        cv::Mat pcaData;
        pca.project(row, pcaData);
        std::stringstream pcaValues;
        pcaValues << pcaData.at<float>(0,0) << ","
                  << pcaData.at<float>(0,1) << ","
                  << pcaData.at<float>(0,2) << ","
                  << labelMat.at<int>(0,i);
        //std::cout << pcaValues.str().c_str() << std::endl;
        filePCA << pcaValues.str().c_str() << std::endl;
    }
    filePCA.close();
}

//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::readDataTrainSVMGray(const std::string &dirData,
                                        const std::string &fileRelationsLabels,
                                        cv::Mat &labelMat,
                                        cv::Mat &dataMat)
{
    //Leemos el fichero con la realacion entre nombre del fichero y numero de etiqueta
    std::map < std::string, int> relationsLabel;
    readRelationsLabels(fileRelationsLabels, relationsLabel);

    //Leemos todos los ficheros de descriptores BOW y los indexamos por etiquetas (la primera cadena hasta "_")
    std::vector < std::string> filesImgs;
    filesDir(dirData, Constants::IMAGE_EXT, filesImgs);
    int numFiles= filesImgs.size();
    std::cout << numFiles << " files for train." << std::endl;

    std::map < string, std::vector <std::string> > filesByLabel;
    std::vector < std::string>::iterator it_filesImgs = filesImgs.begin();
    for (; it_filesImgs != filesImgs.end(); it_filesImgs++)
    {
        std::string fileName = *it_filesImgs;
        std::string labelDataFile = fileName.substr(fileName.find_last_of("/")+1,fileName.size());
        labelDataFile = labelDataFile.substr(0,labelDataFile.find_first_of("_"));

        //Cada fichero es de la clase que indica la cadena previa de su nombre, para juntar clases habria que meter aqui
        //un diccionatio y juntar los ficheros que queramos que tengan la misma etiqueta
        filesByLabel[labelDataFile].push_back(fileName);
    }

    //Generamos la matriz de etiquetas y la de datos pa
    labelMat = cv::Mat(numFiles, 1, CV_32SC1, cv::Scalar(0));
    dataMat = cv::Mat(numFiles, 100*100, CV_32FC1);
    std::map < string, std::vector <std::string> >::iterator it_filesByLabel = filesByLabel.begin();
    int pos = 0;
    int idOfLabel = 0;
    for (; it_filesByLabel != filesByLabel.end(); it_filesByLabel++)
    {
        std::string label = it_filesByLabel->first;
        int numDataOfLabel = it_filesByLabel->second.size();
        for (int i=0; i<numDataOfLabel; i++)
        {
            //            std::cout << pos << std::endl;

            //si entre las relacion de etiqeutas de encontramos la etiqueta actual podremos el id de
            //la clase si no dejaremos un 0 para las clases que aparezcan en el fch de relaciones
            if (relationsLabel.find(label) != relationsLabel.end())
                labelMat.at<__int32_t>(pos,0) = relationsLabel[label];

            cv::Mat desc;
            descriptorGray(it_filesByLabel->second[i],desc);
            desc.copyTo(dataMat.row(pos));
            //            desc.release();

            //            std::cout << dataMat.row(pos) << std::endl;
            pos++;
        }
        //        std::cout << idOfLabel << std::endl;

        idOfLabel++;
    }
}

//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::on_pBGenerateSVM_clicked()
{
    generateSVM();
}


//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::descriptorBowRegions(const std::string &fileName,
                                        cv::Mat &descBow)
{
    //Generamos los sift de toda la imagen
    cv::Mat img = cv::imread(fileName);
    cv::Mat sifts;
//    UTIL_Sift::siftsRegions(img,
//                            ui->lkpSize->text().toInt(),
//                            ui->lkpDensity->text().toInt(),
//                            ui->cBRootSift->isChecked(),
//                            ui->cBFilterSift->isChecked(),
//                            ui->lthresholdFilterSift->text().toFloat(),
//                            descriptors);
    img.release();

    //Generamos la traduccionBow de cada imagen
    std::vector< std::vector< int > > pointIdxsOfClusters;
    this->bow->compute(sifts,
                       descBow,
                       &pointIdxsOfClusters);

    std::cout << "generate descriptor bow-regions from " << fileName << std::endl;
}

//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::descriptorBow(const std::string &fileName,
                                 cv::Mat &descBow)
{
    cv::Mat img = cv::imread(fileName);

    cv::Mat descriptors;
    std::vector <cv::KeyPoint> keyPoints;
    UTIL_Sift::sifts(img,
                     ui->lkpSize->text().toInt(),
                     ui->lkpDensity->text().toInt(),
                     ui->cBRootSift->isChecked(),
                     ui->cBFilterSift->isChecked(),
                     ui->lthresholdFilterSift->text().toFloat(),
                     descriptors,
                     keyPoints);

    std::vector< std::vector< int > > pointIdxsOfClusters;
    this->bow->compute(descriptors,
                       descBow,
                       &pointIdxsOfClusters);

    img.release();

    std::cout << "generate descriptor bow from " << fileName << std::endl;
}

//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::descriptorGray(const std::string &fileName,
                                  cv::Mat &descGray)
{
    descGray = cv::imread(fileName,cv::IMREAD_GRAYSCALE);
    descGray = descGray.reshape(0,1);
    descGray.convertTo(descGray, CV_32FC1);

    std::cout << "generate descriptor gray from " << fileName << std::endl;
}

//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::on_pBTestSVM_clicked()
{
    //Leemos el fichero con la realacion entre nombre del fichero y numero de etiqueta
    std::map < std::string, int> relationsLabel;
    std::stringstream labelsFileName;
    labelsFileName << ui->lprincipalDir->text().toStdString() << ui->lLabelsFileName->text().toStdString();
    readRelationsLabels(labelsFileName.str(), relationsLabel);
    std::cout << "read relations " << labelsFileName.str() << std::endl;

    //leemos el diccionario
    std::stringstream diccionaryFile;
    diccionaryFile << ui->lprincipalDir->text().toStdString() << ui->lDiccionaryFile->text().toStdString();
    readDictionary(diccionaryFile.str());
    std::cout << "read dicctionay " << diccionaryFile.str() << std::endl;

    //Leemos el svm
    std::stringstream fileNameSVM;
    fileNameSVM << ui->lprincipalDir->text().toStdString() << ui->lFileNameTestSVM->text().toStdString();
    cv::Ptr<cv::ml::SVM> svm = cv::Algorithm::load< cv::ml::SVM >(fileNameSVM.str());
    std::cout << "read svm " << fileNameSVM.str() << std::endl;

    std::stringstream fileResult;
    fileResult << ui->lprincipalDir->text().toStdString()
               << ui->lfileResult->text().toStdString()
               << "." << Constants::RESULT_EXT;
    std::ofstream file(fileResult.str().c_str(), std::ofstream::out);

    //Leemos las images para testear
    std::stringstream dirTestFiles;
    dirTestFiles << ui->lprincipalDir->text().toStdString() << ui->lDirImgsTestSVM->text().toStdString();
    std::vector < std::string> filesTest;
    filesDir(dirTestFiles.str(), Constants::IMAGE_EXT, filesTest);
    std::vector < std::string>::iterator it_filesTest= filesTest.begin();
    for (; it_filesTest != filesTest.end(); it_filesTest++)
    {
        std::string fileName = *it_filesTest;

        cv::Mat desc;

        if (this->ui->cBGrayImgs->isChecked())
            descriptorGray(fileName, desc); //Si vamos a clasficar las imagenes en gris
        else
            descriptorBow(fileName, desc); //Si vamos a clasficar las imagenes traducidas con BOW

        if( desc.empty() )
            return;

        float result = svm->predict(desc,cv::noArray(),cv::ml::StatModel::RAW_OUTPUT);

        std::string name = fileName.substr(fileName.find_last_of("/")+1,fileName.length());
        name = name.substr(0,name.find_first_of("_"));

        file << fileName << "," << relationsLabel[name] << "," <<  result  << std::endl;

        std::cout << result << " - " << relationsLabel[name] << " - " << fileName << std::endl;

        desc.release();
    }
    file.close();

    svm.release();
    delete bow;
    bow = NULL;
    this->initGenerateDescriptorsBOW = false;
}

//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::on_pBGenerateKeyPoints_clicked()
{
    //Leeemos la lista de fichero de imagen de los que vamos a generar todos sus sift
    std::stringstream dirGenerateKeyPoints;
    dirGenerateKeyPoints << ui->lprincipalDir->text().toStdString() << ui->dirKeyGenerate->text().toStdString();
    std::vector < std::string> files;
    filesDir(dirGenerateKeyPoints.str(),
             Constants::IMAGE_EXT,
             files);

    //Generamos los ficheros sift correspondientes a cada imagen
    if (ui->cBPcaSift->isChecked())
    {
        UTIL_Sift::saveDescriptorsSiftPCADir(files,
                                             ui->lkpSize->text().toInt(),
                                             ui->lkpDensity->text().toInt(),
                                             ui->cBRootSift->isChecked(),
                                             ui->cBFilterSift->isChecked(),
                                             ui->lthresholdFilterSift->text().toFloat());
    }
    else
    {
        UTIL_Sift::saveDescriptorsSiftDir(files,
                                          ui->lkpSize->text().toInt(),
                                          ui->lkpDensity->text().toInt(),
                                          ui->cBRootSift->isChecked(),
                                          ui->cBFilterSift->isChecked(),
                                          ui->lthresholdFilterSift->text().toFloat());
    }
}


//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::on_pBThresholdSift_clicked()
{
    std::stringstream dirGenerateKeyPoints;
    dirGenerateKeyPoints << ui->lprincipalDir->text().toStdString() << ui->dirKeyGenerate->text().toStdString();

    std::vector < std::string> files;
    filesDir(dirGenerateKeyPoints.str(), Constants::IMAGE_EXT , files);

    DialogShowSift *dialogSift = new DialogShowSift(this);
    dialogSift->showSift(files,
                         ui->lkpSize->text().toInt(),
                         ui->lkpDensity->text().toInt(),false);
}

//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::on_pBAutoDetectSift_clicked()
{
    std::stringstream dirGenerateKeyPoints;
    dirGenerateKeyPoints << ui->lprincipalDir->text().toStdString() << ui->dirKeyGenerate->text().toStdString();

    std::vector < std::string> files;
    filesDir(dirGenerateKeyPoints.str(), Constants::IMAGE_EXT , files);

    DialogShowSift *dialogSift = new DialogShowSift(this);
    dialogSift->showSift(files,
                         ui->lkpSize->text().toInt(),
                         ui->lkpDensity->text().toInt(),true);
}

//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::all()
{
    this->ui->pbDictionary->click();
    this->ui->pBGenerateDescriptorsBOW->click();
    this->ui->pBGenerateSVM->click();
    this->ui->pBTestSVM->click();
}

//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::on_pBgenerateDB_clicked()
{
    generateDB->show();
}


//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::on_pBAll_clicked()
{
    ui->cBGrayImgs->setChecked(false);
    ui->cBFilterSift->setChecked(false);
    ui->cBRootSift->setChecked(true);

    ui->lkpSize->setText("20");
    ui->lkpDensity->setText("3");
    this->ui->pBGenerateKeyPoints->click();
    /*
    ui->ldictionarySize->setText("200");
    ui->lfilePCA->setText("pca_20_3_rootsift_200");
    ui->lfileResult->setText("result_20_3_rootsift_200");
    all();

    ui->ldictionarySize->setText("500");
    ui->lfilePCA->setText("pca_20_3_rootsift_500");
    ui->lfileResult->setText("result_20_3_rootsift_500");
    all();*/

    ui->ldictionarySize->setText("1200");
    ui->lfilePCA->setText("pca_20_3_rootsift_1200");
    ui->lfileResult->setText("result_20_3_rootsift_1200");
    all();

    ui->ldictionarySize->setText("2000");
    ui->lfilePCA->setText("pca_20_3_rootsift_2000");
    ui->lfileResult->setText("result_20_3_rootsift_2000");
    all();

    //-------------------------------
    ui->lkpSize->setText("10");
    ui->lkpDensity->setText("3");
    this->ui->pBGenerateKeyPoints->click();
    ui->ldictionarySize->setText("200");
    ui->lfilePCA->setText("pca_10_3_rootsift_200");
    ui->lfileResult->setText("result_10_3_rootsift_200");
    all();

    ui->ldictionarySize->setText("500");
    ui->lfilePCA->setText("pca_10_3_rootsift_500");
    ui->lfileResult->setText("result_10_3_rootsift_500");
    all();

    ui->ldictionarySize->setText("1200");
    ui->lfilePCA->setText("pca_10_3_rootsift_1200");
    ui->lfileResult->setText("result_10_3_rootsift_1200");
    all();

    ui->ldictionarySize->setText("2000");
    ui->lfilePCA->setText("pca_10_3_rootsift_2000");
    ui->lfileResult->setText("result_10_3_rootsift_2000");
    all();

    //-------------------------------
    ui->lkpSize->setText("5");
    ui->lkpDensity->setText("3");
    this->ui->pBGenerateKeyPoints->click();
    ui->ldictionarySize->setText("200");
    ui->lfilePCA->setText("pca_5_3_rootsift_200");
    ui->lfileResult->setText("result_5_3_rootsift_200");
    all();

    ui->ldictionarySize->setText("500");
    ui->lfilePCA->setText("pca_5_3_rootsift_500");
    ui->lfileResult->setText("result_5_3_rootsift_500");
    all();

    ui->ldictionarySize->setText("1200");
    ui->lfilePCA->setText("pca_5_3_rootsift_1200");
    ui->lfileResult->setText("result_5_3_rootsift_1200");
    all();

    ui->ldictionarySize->setText("2000");
    ui->lfilePCA->setText("pca_5_3_rootsift_2000");
    ui->lfileResult->setText("result_5_3_rootsift_2000");
    all();

    //-------------------------------
    ui->lkpSize->setText("3");
    ui->lkpDensity->setText("3");
    this->ui->pBGenerateKeyPoints->click();
    ui->ldictionarySize->setText("200");
    ui->lfilePCA->setText("pca_3_3_rootsift_200");
    ui->lfileResult->setText("result_3_3_rootsift_200");
    all();

    ui->ldictionarySize->setText("500");
    ui->lfilePCA->setText("pca_3_3_rootsift_500");
    ui->lfileResult->setText("result_3_3_rootsift_500");
    all();

    ui->ldictionarySize->setText("1200");
    ui->lfilePCA->setText("pca_3_3_rootsift_1200");
    ui->lfileResult->setText("result_3_3_rootsift_1200");
    all();

    ui->ldictionarySize->setText("2000");
    ui->lfilePCA->setText("pca_3_3_rootsift_2000");
    ui->lfileResult->setText("result_3_3_rootsift_2000");
    all();


    //-------------------------------
    //-------------------------------
    ui->lkpSize->setText("20");
    ui->lkpDensity->setText("2");
    this->ui->pBGenerateKeyPoints->click();

    ui->ldictionarySize->setText("200");
    ui->lfilePCA->setText("pca_20_2_rootsift_200");
    ui->lfileResult->setText("result_20_2_rootsift_200");
    all();

    ui->ldictionarySize->setText("500");
    ui->lfilePCA->setText("pca_20_2_rootsift_500");
    ui->lfileResult->setText("result_20_2_rootsift_500");
    all();

    ui->ldictionarySize->setText("1200");
    ui->lfilePCA->setText("pca_20_2_rootsift_1200");
    ui->lfileResult->setText("result_20_2_rootsift_1200");
    all();

    ui->ldictionarySize->setText("2000");
    ui->lfilePCA->setText("pca_20_2_rootsift_2000");
    ui->lfileResult->setText("result_20_2_rootsift_2000");
    all();

    //-------------------------------
    ui->lkpSize->setText("10");
    ui->lkpDensity->setText("2");
    this->ui->pBGenerateKeyPoints->click();

    ui->ldictionarySize->setText("200");
    ui->lfilePCA->setText("pca_10_2_rootsift_200");
    ui->lfileResult->setText("result_10_2_rootsift_200");
    all();

    ui->ldictionarySize->setText("500");
    ui->lfilePCA->setText("pca_10_2_rootsift_500");
    ui->lfileResult->setText("result_10_rootsift_500");
    all();

    ui->ldictionarySize->setText("1200");
    ui->lfilePCA->setText("pca_10_2_rootsift_1200");
    ui->lfileResult->setText("result_10_2_rootsift_1200");
    all();

    ui->ldictionarySize->setText("2000");
    ui->lfilePCA->setText("pca_10_2_rootsift_2000");
    ui->lfileResult->setText("result_10_2_rootsift_2000");
    all();

    //-------------------------------
    ui->lkpSize->setText("5");
    ui->lkpDensity->setText("2");
    this->ui->pBGenerateKeyPoints->click();

    ui->ldictionarySize->setText("200");
    ui->lfilePCA->setText("pca_5_2_rootsift_200");
    ui->lfileResult->setText("result_5_2_rootsift_200");
    all();

    ui->ldictionarySize->setText("500");
    ui->lfilePCA->setText("pca_5_2_rootsift_500");
    ui->lfileResult->setText("result_5_2_rootsift_500");
    all();

    ui->ldictionarySize->setText("1200");
    ui->lfilePCA->setText("pca_5_2_rootsift_1200");
    ui->lfileResult->setText("result_5_2_rootsift_1200");
    all();

    ui->lkpSize->setText("5");
    ui->lkpDensity->setText("2");
    ui->ldictionarySize->setText("2000");
    ui->lfilePCA->setText("pca_5_2_rootsift_2000");
    ui->lfileResult->setText("result_5_2_rootsift_2000");
    all();

    //-------------------------------
    ui->lkpSize->setText("3");
    ui->lkpDensity->setText("2");
    this->ui->pBGenerateKeyPoints->click();

    ui->ldictionarySize->setText("200");
    ui->lfilePCA->setText("pca_3_2_rootsift_200");
    ui->lfileResult->setText("result_3_2_rootsift_200");
    all();

    ui->ldictionarySize->setText("500");
    ui->lfilePCA->setText("pca_3_2_rootsift_500");
    ui->lfileResult->setText("result_3_2_rootsift_500");
    all();

    ui->ldictionarySize->setText("1200");
    ui->lfilePCA->setText("pca_3_2_rootsift_1200");
    ui->lfileResult->setText("result_3_2_rootsift_1200");
    all();

    ui->ldictionarySize->setText("2000");
    ui->lfilePCA->setText("pca_3_2_rootsift_2000");
    ui->lfileResult->setText("result_3_2_rootsift_2000");
    all();

    /*//Gris
    ui->cBGrayImgs->setChecked(true);
    ui->lfilePCA->setText("pca_gray");
    ui->lfileResult->setText("result_gray");
    this->ui->pBGenerateSVM->click();
    this->ui->pBTestSVM->click();
    */

}

