#include "mwtracktrace.h"
#include "ui_mwtracktrace.h"

#include "dialogshowbow.h"
#include "dialogshowsift.h"

#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>


#include "util_files.h"
#include "util_sift.h"
#include "util_bow.h"
#include "constants.h"

#include <iostream>
#include <fstream>
#include <string>

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
void MWTrackTrace::on_pbDictionary_clicked()
{
    //Directorio donde estan los ficheros de descriptores para crear el diccionario
    std::stringstream dirDiccionary;
    dirDiccionary << ui->lprincipalDir->text().toStdString() << this->ui->lDiccionaryDir->text().toStdString();

    //Se leen los nombres de todos los ficheros de descriptores sift del directorio
    std::vector < std::string> files;
    UTIL_Files::filesDir(dirDiccionary.str(), Constants::DESCRIPTOR_EXT, files);

    //Leemos los estiquetas que tienen que tener los ficheros con los que vamos a entrenar
    std::stringstream fileLabelsToDicctionary;
    fileLabelsToDicctionary << this->ui->lprincipalDir->text().toStdString() << ui->lLabelsToDicctionary->text().toStdString();
    std::vector <std::string> labelsToDicctionary;
    UTIL_Files::readLabels(fileLabelsToDicctionary.str(),labelsToDicctionary);

    //Dejamos solo los ficheros que se indican en labelsForBow (Esto permite que se entrene el bow solo
    //con un determinado tipo)
    UTIL_Files::filesFilter(files,labelsToDicctionary);

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
    UTIL_Files::createDir(dirBows.str());
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
        UTIL_Files::createDir(dirSamples.str());
        std::stringstream dirSamplesNum;
        dirSamplesNum << dirSamples.str().c_str() << i;
        UTIL_Files::createDir(dirSamplesNum.str());
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
        UTIL_Files::filesDir(dirForBOW.str(), Constants::IMAGE_EXT , this->filesForBOW);
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
    UTIL_Files::readRelationsLabels(fileRelationsLabels, relationsLabel);

    //Leemos todos los ficheros de descriptores BOW y los indexamos por etiquetas (la primera cadena hasta "_")
    std::vector < std::string> filesDescriptorsBOW;
    UTIL_Files::filesDir(dirData, Constants::DESCRIPTOR_BOW_EXT, filesDescriptorsBOW);
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
    cv::PCA pcaResult(dataMat, cv::Mat(), CV_PCA_DATA_AS_ROW, num_components);

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
        pcaResult.project(row, pcaData);
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
    UTIL_Files::readRelationsLabels(fileRelationsLabels, relationsLabel);

    //Leemos todos los ficheros de descriptores BOW y los indexamos por etiquetas (la primera cadena hasta "_")
    std::vector < std::string> filesImgs;
    UTIL_Files::filesDir(dirData, Constants::IMAGE_EXT, filesImgs);
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
void MWTrackTrace::descriptorBow(const std::string &fileName,
                                 cv::Mat &descBow)
{
    cv::Mat img = cv::imread(fileName);

    cv::Mat descriptors;
    std::vector <cv::KeyPoint> keyPoints;
    UTIL_Sift::descriptorsSift(img,
                               ui->lkpSize->text().toInt(),
                               ui->lkpDensity->text().toInt(),
                               ui->cBRootSift->isChecked(),
                               ui->cBFilterSift->isChecked(),
                               ui->lthresholdFilterSift->text().toFloat(),
                               ui->cBRegionsBow->isChecked(),
                               descriptors,
                               keyPoints);

    if (ui->cBPcaSift->isChecked())
    {
        pca.project(descriptors,descriptors);
        std::cout << "generate project pca-sift from " << fileName << std::endl;
    }

    std::vector< std::vector< int > > pointIdxsOfClusters;
    this->bow->compute(descriptors,
                       descBow,
                       &pointIdxsOfClusters);

    img.release();

    std::cout << "generate descriptor bow from " << fileName << std::endl;
}


//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::cropImage(cv::Mat &img,
                             std::vector<cv::Mat> &subImgs)
{
    subImgs.push_back(img(cv::Rect(0,0,img.cols/2,img.rows/2)));
    subImgs.push_back(img(cv::Rect(img.cols/2,0,img.cols/2,img.rows/2)));
    subImgs.push_back(img(cv::Rect(0,img.rows/2,img.cols/2,img.rows/2)));
    subImgs.push_back(img(cv::Rect(img.cols/2,img.rows/2,img.cols/2,img.rows/2)));
}

/*
//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::descriptorBow(const std::string &fileName,
                                 cv::Mat &descBow)
{
    cv::Mat img = cv::imread(fileName);


    //Tendremos la concatenacion de el hisgrama de la iamgen cmpleta, luego de l4 y luego de 16

    cv::Mat descriptors;
    std::vector <cv::KeyPoint> keyPoints;
    UTIL_Sift::descriptorsSift(img,
                               ui->lkpSize->text().toInt(),
                               ui->lkpDensity->text().toInt(),
                               ui->cBRootSift->isChecked(),
                               ui->cBFilterSift->isChecked(),
                               ui->lthresholdFilterSift->text().toFloat(),
                               false,
                               descriptors,
                               keyPoints);


    //Partimos la imagen en 4 y calculamos los descritores de cada region
    std::vector <cv::Mat> subImgs;
    cropImage(img,subImgs);
    for (int i=0; i<subImgs.size(); i++)
    {
        cv::Mat subDescriptors;
        std::vector <cv::KeyPoint> subKeyPoints;
        UTIL_Sift::descriptorsSift(subImgs[i],
                                   ui->lkpSize->text().toInt(),
                                   ui->lkpDensity->text().toInt()/2,
                                   ui->cBRootSift->isChecked(),
                                   ui->cBFilterSift->isChecked(),
                                   ui->lthresholdFilterSift->text().toFloat(),
                                   false,
                                   subDescriptors,
                                   subKeyPoints);

        //Concatenemos los descriptores de la region
        cv::vconcat(descriptors,subDescriptors,descriptors);


        //Partimos cada subimagen en 4 y volvemos ha hacer los mimos
        std::vector <cv::Mat> subSubImgs;
        cropImage(subImgs[i],subSubImgs);
        for (int i=0; i<subImgs.size(); i++)
        {
            cv::Mat subSubDescriptors;
            std::vector <cv::KeyPoint> subSubKeyPoints;
            UTIL_Sift::descriptorsSift(subImgs[i],
                                       ui->lkpSize->text().toInt(),
                                       2,
                                       ui->cBRootSift->isChecked(),
                                       ui->cBFilterSift->isChecked(),
                                       ui->lthresholdFilterSift->text().toFloat(),
                                       false,
                                       subSubDescriptors,
                                       subSubKeyPoints);

            //Concatensmo los descriptores de la subsurecion
            cv::vconcat(descriptors,subSubDescriptors,descriptors);
        }
    }


    if (ui->cBPcaSift->isChecked())
    {
        pca.project(descriptors,descriptors);
        std::cout << "generate project pca-sift from " << fileName << std::endl;
    }

    std::vector< std::vector< int > > pointIdxsOfClusters;
    this->bow->compute(descriptors,
                       descBow,
                       &pointIdxsOfClusters);

    img.release();

    std::cout << "generate descriptor bow from " << fileName << std::endl;
}
*/

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
void MWTrackTrace::descriptor(const std::string &fileName,
                              cv::Mat &desc)
{
    if (this->ui->cBGrayImgs->isChecked())
        descriptorGray(fileName, desc); //Si vamos a clasficar las imagenes en gris
    else
        descriptorBow(fileName, desc); //Si vamos a clasficar las imagenes traducidas con BOW
}


//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::readTrainData(cv::Ptr<cv::ml::SVM> &svm,
                                 std::map < std::string, int> &relationsLabel)
{
    //Leemos el fichero con la realacion entre nombre del fichero y numero de etiqueta
    std::stringstream labelsFileName;
    labelsFileName << ui->lprincipalDir->text().toStdString() << ui->lLabelsFileName->text().toStdString();
    UTIL_Files::readRelationsLabels(labelsFileName.str(), relationsLabel);
    std::cout << "read relations " << labelsFileName.str() << std::endl;

    //leemos el diccionario
    std::stringstream diccionaryFile;
    diccionaryFile << ui->lprincipalDir->text().toStdString() << ui->lDiccionaryFile->text().toStdString();
    readDictionary(diccionaryFile.str());
    std::cout << "read dicctionay " << diccionaryFile.str() << std::endl;

    //Leemos el svm
    std::stringstream fileNameSVM;
    fileNameSVM << ui->lprincipalDir->text().toStdString() << ui->lFileNameTestSVM->text().toStdString();
    svm = cv::Algorithm::load< cv::ml::SVM >(fileNameSVM.str());
    std::cout << "read svm " << fileNameSVM.str() << std::endl;
}

//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::on_pBTestSVM_clicked()
{
    //Leemos los datos de entrenamiento
    cv::Ptr<cv::ml::SVM> svm;
    std::map < std::string, int> relationsLabel;
    readTrainData(svm,relationsLabel);

    //Se abre el fichero de resultados
    std::stringstream fileResult;
    fileResult << ui->lprincipalDir->text().toStdString()
               << ui->lfileResult->text().toStdString()
               << "." << Constants::RESULT_EXT;
    std::ofstream file(fileResult.str().c_str(), std::ofstream::out);

    //Leemos las images para testear
    std::stringstream dirTestFiles;
    dirTestFiles << ui->lprincipalDir->text().toStdString() << ui->lDirImgsTestSVM->text().toStdString();
    std::vector < std::string> filesTest;
    UTIL_Files::filesDir(dirTestFiles.str(), Constants::IMAGE_EXT, filesTest);
    std::vector < std::string>::iterator it_filesTest= filesTest.begin();
    for (; it_filesTest != filesTest.end(); it_filesTest++)
    {
        std::string fileName = *it_filesTest;

        cv::Mat desc;

        this->descriptor(fileName,desc);

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
void MWTrackTrace::on_pBTestSVMTrack_clicked()
{
    ui->cBGrayImgs->setChecked(false);
    ui->cBFilterSift->setChecked(true);
    ui->lthresholdFilterSift->setText("0.85");
    ui->cBAutoDetectSift->setChecked(false);
    ui->cBRootSift->setChecked(true);
    ui->cBPcaSift->setChecked(false);
    ui->cBRegionsBow->setChecked(false);
    ui->lDirImgsTestSVM->setText("faces_test");
    ui->lkpSize->setText("3");
    ui->lkpDensity->setText("3");
    ui->ldictionarySize->setText("2000");
    ui->lfilePCA->setText("pca_kp3_den3_2000w_filter_0.85_rootsift_region_2_Sheldon");
    ui->lfileResult->setText("result_tracks");

    //Leemos los datos de entrenamiento
    std::map < std::string, int> relationsLabel;
    //Leemos el fichero con la realacion entre nombre del fichero y numero de etiqueta
    std::stringstream labelsFileName;
    labelsFileName << ui->lprincipalDir->text().toStdString() << ui->lLabelsFileName->text().toStdString();
    UTIL_Files::readRelationsLabels(labelsFileName.str(), relationsLabel);
    std::cout << "read relations " << labelsFileName.str() << std::endl;

    //leemos el diccionario
    std::stringstream diccionaryFile;
    diccionaryFile << ui->lprincipalDir->text().toStdString() << ui->lDiccionaryFile->text().toStdString();
    readDictionary(diccionaryFile.str());
    std::cout << "read dicctionay " << diccionaryFile.str() << std::endl;

    //Leemos los svm de cada clase
    cv::Ptr<cv::ml::SVM> svm_Sheldon;
    std::stringstream fileNameSVM_Sheldon;
    fileNameSVM_Sheldon << ui->lprincipalDir->text().toStdString() << "svm_Sheldon.svm";
    svm_Sheldon = cv::Algorithm::load< cv::ml::SVM >(fileNameSVM_Sheldon.str());
    std::cout << "read svm " << fileNameSVM_Sheldon.str() << std::endl;

    cv::Ptr<cv::ml::SVM> svm_Leonard;
    std::stringstream fileNameSVM_Leonard;
    fileNameSVM_Leonard << ui->lprincipalDir->text().toStdString() << "svm_Leonard.svm";
    svm_Leonard = cv::Algorithm::load< cv::ml::SVM >(fileNameSVM_Leonard.str());
    std::cout << "read svm " << fileNameSVM_Leonard.str() << std::endl;

    cv::Ptr<cv::ml::SVM> svm_Penny;
    std::stringstream fileNameSVM_Penny;
    fileNameSVM_Penny << ui->lprincipalDir->text().toStdString() << "svm_Penny.svm";
    svm_Penny = cv::Algorithm::load< cv::ml::SVM >(fileNameSVM_Penny.str());
    std::cout << "read svm " << fileNameSVM_Penny.str() << std::endl;

    cv::Ptr<cv::ml::SVM> svm_Howard;
    std::stringstream fileNameSVM_Howard;
    fileNameSVM_Howard << ui->lprincipalDir->text().toStdString() << "svm_Howard.svm";
    svm_Howard = cv::Algorithm::load< cv::ml::SVM >(fileNameSVM_Howard.str());
    std::cout << "read svm " << fileNameSVM_Howard.str() << std::endl;

    cv::Ptr<cv::ml::SVM> svm_Raj;
    std::stringstream fileNameSVM_Raj;
    fileNameSVM_Raj << ui->lprincipalDir->text().toStdString() << "svm_Raj.svm";
    svm_Raj = cv::Algorithm::load< cv::ml::SVM >(fileNameSVM_Raj.str());
    std::cout << "read svm " << fileNameSVM_Raj.str() << std::endl;


    //Se abre el fichero de resultados
    std::stringstream fileResult;
    fileResult << ui->lprincipalDir->text().toStdString()
               << ui->lfileResult->text().toStdString()
               << "." << Constants::RESULT_EXT;
    std::ofstream file(fileResult.str().c_str(), std::ofstream::out);

    //Leemos los track para testear en lso qeu viene el nombre del fichero precedido del track al que pertenece
    std::stringstream fileFilesTracks;
    fileFilesTracks << ui->lprincipalDir->text().toStdString() << "files_tracks_01.txt";
    std::map < std::string, std::vector <std::string> > filesTrack;
    UTIL_Files::readFilesTracks(fileFilesTracks.str(),filesTrack);

     std::map < std::string, std::vector <std::string> >::iterator it_filesTrack = filesTrack.begin();
     for (; it_filesTrack != filesTrack.end(); it_filesTrack++)
     {
         std::string trackName = it_filesTrack->first;
         std::vector <std::string> files = it_filesTrack->second;
         std::cout << "procesing track name " << trackName << std::endl;

         cv::Mat_<float> count = cv::Mat_<float>::zeros(1,5);


         std::vector <std::string>::iterator it_files = files.begin();
         for (; it_files != files.end(); it_files++)
         {
             std::string fileName = *it_files;
             std::cout << "procesing file " << fileName << std::endl;

             cv::Mat desc;
             this->descriptor(fileName,desc);

             if( desc.empty() )
                 return;

             cv::Mat_<float> scores;
             float scoreSheldon = svm_Sheldon->predict(desc,cv::noArray(),cv::ml::StatModel::RAW_OUTPUT);
             float scoreLeonard = svm_Leonard->predict(desc,cv::noArray(),cv::ml::StatModel::RAW_OUTPUT);
             float scorePenny = svm_Penny->predict(desc,cv::noArray(),cv::ml::StatModel::RAW_OUTPUT);
             float scoreHoward = svm_Howard->predict(desc,cv::noArray(),cv::ml::StatModel::RAW_OUTPUT);
             float scoreRaj = svm_Raj->predict(desc,cv::noArray(),cv::ml::StatModel::RAW_OUTPUT);

             scores.push_back(scoreSheldon);
             scores.push_back(scoreLeonard);
             scores.push_back(scorePenny);
             scores.push_back(scoreHoward);
             scores.push_back(scoreRaj);

             double min,max;
             cv::Point  minpos,maxpos;
             cv::minMaxLoc(scores,&min,&max,&minpos,&maxpos);

             std::string name = fileName.substr(fileName.find_last_of("/")+1,fileName.length());
             name = name.substr(0,name.find_first_of("_"));

             //file << fileName << "," <<  min  << ',' << minpos.y+1 << std::endl;

             std::cout << scores << std::endl;
             //std::cout << min << " - " << minpos.y+1 << " - " << fileName << std::endl;

             count.at<int>(0,minpos.y) = count.at<int>(0,minpos.y) + 1;

             /*
             count.at<float>(0,0) = count.at<float>(0,0) + scoreSheldon;
             count.at<float>(0,1) = count.at<float>(1,0) + scoreLeonard;
             count.at<float>(0,2) = count.at<float>(2,0) + scorePenny;
             count.at<float>(0,3) = count.at<float>(3,0) + scoreHoward;
             count.at<float>(0,4) = count.at<float>(4,0) + scoreRaj;
             */

             desc.release();
         }

         //Se seleciona el mas vodato el que tenia mas puntacion acumualda
         double min,max;
         cv::Point  minpos,maxpos;
         cv::minMaxLoc(count,&min,&max,&minpos,&maxpos);

         file << trackName << ","  << maxpos.x+1 << std::endl;
         std::cout << trackName << ","  << maxpos.x+1 << std::endl;
         //file << trackName << "  " << count << ","  << minpos.x+1 << std::endl;
         //std::cout << trackName << ","  << minpos.x+1 << std::endl;
     }
     file.close();

    /*
    //Leemos las images para testear
    std::stringstream dirTestFiles;
    dirTestFiles << ui->lprincipalDir->text().toStdString() << ui->lDirImgsTestSVM->text().toStdString();
    std::vector < std::string> filesTest;
    UTIL_Files::filesDir(dirTestFiles.str(), Constants::IMAGE_EXT, filesTest);
    std::vector < std::string>::iterator it_filesTest= filesTest.begin();
    for (; it_filesTest != filesTest.end(); it_filesTest++)
    {
        std::string fileName = *it_filesTest;

        cv::Mat desc;

        this->descriptor(fileName,desc);

        if( desc.empty() )
            return;


        cv::Mat_<float> scores;
        scores.push_back(svm_Sheldon->predict(desc,cv::noArray(),cv::ml::StatModel::RAW_OUTPUT));
        scores.push_back(svm_Leonard->predict(desc,cv::noArray(),cv::ml::StatModel::RAW_OUTPUT));
        scores.push_back(svm_Penny->predict(desc,cv::noArray(),cv::ml::StatModel::RAW_OUTPUT));
        scores.push_back(svm_Howard->predict(desc,cv::noArray(),cv::ml::StatModel::RAW_OUTPUT));
        scores.push_back(svm_Raj->predict(desc,cv::noArray(),cv::ml::StatModel::RAW_OUTPUT));

        double min,max;
        cv::Point  minpos,maxpos;
        cv::minMaxLoc(scores,&min,&max,&minpos,&maxpos);

        std::string name = fileName.substr(fileName.find_last_of("/")+1,fileName.length());
        name = name.substr(0,name.find_first_of("_"));

        file << fileName << "," <<  min  << ',' << minpos.y+1 << std::endl;

        std::cout << scores << std::endl;
        std::cout << min << " - " << minpos.y+1 << " - " << fileName << std::endl;

        desc.release();
    }
    file.close();
    */

    svm_Sheldon.release();
    svm_Leonard.release();
    svm_Penny.release();
    svm_Howard.release();
    svm_Raj.release();
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
    UTIL_Files::filesDir(dirGenerateKeyPoints.str(),
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
                                             ui->lthresholdFilterSift->text().toFloat(),
                                             ui->cBRegionsBow->isChecked(),
                                             pca);
    }
    else
    {
        bool prueba = ui->cBRegionsBow->isChecked();
        UTIL_Sift::saveDescriptorsSiftDir(files,
                                          ui->lkpSize->text().toInt(),
                                          ui->lkpDensity->text().toInt(),
                                          ui->cBRootSift->isChecked(),
                                          ui->cBFilterSift->isChecked(),
                                          ui->cBRegionsBow->isChecked(),
                                          ui->lthresholdFilterSift->text().toFloat());
    }
}


//-------------------------------------------------------------------------------------------------------------
void MWTrackTrace::on_pBThresholdSift_clicked()
{
    std::stringstream dirGenerateKeyPoints;
    dirGenerateKeyPoints << ui->lprincipalDir->text().toStdString() << ui->dirKeyGenerate->text().toStdString();

    std::vector < std::string> files;
    UTIL_Files::filesDir(dirGenerateKeyPoints.str(), Constants::IMAGE_EXT , files);

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
    UTIL_Files::filesDir(dirGenerateKeyPoints.str(), Constants::IMAGE_EXT , files);

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
    //this->ui->pBTestSVMTrack->click();
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

    ui->cBFilterSift->setChecked(true);
    ui->lthresholdFilterSift->setText("0.85");

    ui->cBAutoDetectSift->setChecked(false);

    ui->cBRootSift->setChecked(true);
    ui->cBPcaSift->setChecked(false);
    ui->cBRegionsBow->setChecked(true);
    ui->lDirImgsTestSVM->setText("faces_test");


    //-------------------------------
    //ui->lthresholdFilterSift->setText("0.35");
    ui->lkpSize->setText("3");
    ui->lkpDensity->setText("3");
    this->ui->pBGenerateKeyPoints->click();

    ui->ldictionarySize->setText("200");
    ui->lfilePCA->setText("pca_kp3_den3_200w_filter_0.85_rootsift");
    ui->lfileResult->setText("result_kp3_den3_200w_filter_0.85_rootsift");
    all();


    //ui->lthresholdFilterSift->setText("0.35");
    //ui->lkpSize->setText("3");
    //ui->lkpDensity->setText("3");
    //this->ui->pBGenerateKeyPoints->click();
    ui->ldictionarySize->setText("500");
    ui->lfilePCA->setText("pca_kp3_den3_500w_filter_0.85_rootsift");
    ui->lfileResult->setText("result_kp3_den3_500w_filter_0.85_rootsift");
    all();

    //ui->lthresholdFilterSift->setText("0.65");
    //ui->lkpSize->setText("3");
    //ui->lkpDensity->setText("3");
    //this->ui->pBGenerateKeyPoints->click();
    ui->ldictionarySize->setText("1200");
    ui->lfilePCA->setText("pca_kp3_den3_1200w_filter_0.85_rootsift");
    ui->lfileResult->setText("result_kp3_den3_1200w_filter_0.85_rootsift");
    all();

    //ui->lthresholdFilterSift->setText("0.65");
    //ui->lkpSize->setText("3");
    //ui->lkpDensity->setText("3");
    //this->ui->pBGenerateKeyPoints->click();
    ui->ldictionarySize->setText("2000");
    ui->lfilePCA->setText("pca_kp3_den3_2000w_filter_0.85_rootsift");
    ui->lfileResult->setText("result_kp3_den3_2000w_filter_0.85_rootsift");
    all();

    //ui->lthresholdFilterSift->setText("0.65");
    //ui->lkpSize->setText("3");
    //ui->lkpDensity->setText("3");
    //this->ui->pBGenerateKeyPoints->click();
    ui->ldictionarySize->setText("3000");
    ui->lfilePCA->setText("pca_kp3_den3_3000w_filter_0.85_rootsift");
    ui->lfileResult->setText("result_kp3_den3_3000w_filter_0.85_rootsift");
    all();

    //ui->lthresholdFilterSift->setText("0.65");
    //ui->lkpSize->setText("3");
    //ui->lkpDensity->setText("3");
    //this->ui->pBGenerateKeyPoints->click();
    //ui->ldictionarySize->setText("500");
    //ui->lfilePCA->setText("pca_kp3_den3_500w_filter_0.85_rootsift_pca_region");
    //ui->lfileResult->setText("result_kp3_den3_500w_filter_0.85_rootsift_pca_region");
    //all();

    //ui->lthresholdFilterSift->setText("0.85");
    //ui->lkpSize->setText("3");
    //ui->lkpDensity->setText("3");
    //this->ui->pBGenerateKeyPoints->click();
    //ui->ldictionarySize->setText("2000");
    //ui->lfilePCA->setText("pca_kp3_den5_2000w_filter_0.85");
    //ui->lfileResult->setText("result_kp3_den5_2000w_filter_0.85");
    //all();

    /*
    //ui->lkpSize->setText("3");
    //ui->lkpDensity->setText("10");
    //this->ui->pBGenerateKeyPoints->click();
    ui->ldictionarySize->setText("2000");
    ui->lfilePCA->setText("pca_kp3_den5_2000w");
    ui->lfileResult->setText("result_kp3_den5_2000w");
    all();
    */

    /*
    //-------------------------------
    ui->lkpSize->setText("3");
    ui->lkpDensity->setText("20");
    this->ui->pBGenerateKeyPoints->click();
    this->ui->pBGenerateKeyPoints->click();
    ui->ldictionarySize->setText("2000");
    ui->lfilePCA->setText("pca_kp3_den20_2000w");
    ui->lfileResult->setText("result_kp3_den20_2000w");
    all();


    ui->lkpSize->setText("3");
    ui->lkpDensity->setText("3");
    this->ui->pBGenerateKeyPoints->click();
    ui->ldictionarySize->setText("2000");
    ui->lfilePCA->setText("pca_kp3_den3_2000w");
    ui->lfileResult->setText("result_kp3_den3_2000w");
    all();

    ui->lkpSize->setText("3");
    ui->lkpDensity->setText("5");
    this->ui->pBGenerateKeyPoints->click();
    ui->ldictionarySize->setText("2000");
    ui->lfilePCA->setText("pca_kp3_den5_2000w");
    ui->lfileResult->setText("result_kp3_den5_2000w");
    all();

    ui->lkpSize->setText("3");
    ui->lkpDensity->setText("10");
    this->ui->pBGenerateKeyPoints->click();
    ui->ldictionarySize->setText("2000");
    ui->lfilePCA->setText("pca_kp3_den10_2000w");
    ui->lfileResult->setText("result_kp3_den10_2000w");
    all();
    */


    //-------------------------------
//    ui->lkpSize->setText("20");
//    ui->lkpDensity->setText("3");
//    this->ui->pBGenerateKeyPoints->click();
//    ui->ldictionarySize->setText("200");
//    ui->lfilePCA->setText("pca_20_3_rsift_region_3_200");
//    ui->lfileResult->setText("result_20_3_rsift_region_3_200");
//    all();

//    ui->ldictionarySize->setText("500");
//    ui->lfilePCA->setText("pca_20_3_rsift_region_3_500");
//    ui->lfileResult->setText("result_20_3_rsift_region_3_500");
//    all();

//    ui->ldictionarySize->setText("1200");
//    ui->lfilePCA->setText("pca_20_3_rsift_region_3_1200");
//    ui->lfileResult->setText("result_20_3_rsift_region_3_1200");
//    all();

//    ui->ldictionarySize->setText("2000");
//    ui->lfilePCA->setText("pca_20_3_rsift_region_3_2000");
//    ui->lfileResult->setText("result_20_3_rsift_region_3_2000");
//    all();


    //-------------------------------
//    ui->lkpSize->setText("10");
//    ui->lkpDensity->setText("3");
//    this->ui->pBGenerateKeyPoints->click();
//    ui->ldictionarySize->setText("200");
//    ui->lfilePCA->setText("pca_10_3_rsift_region_200");
//    ui->lfileResult->setText("result_10_3_rsift_region_200");
//    all();

//    ui->ldictionarySize->setText("500");
//    ui->lfilePCA->setText("pca_10_3_rsift_region_500");
//    ui->lfileResult->setText("result_10_3_rsift_region_500");
//    all();

//    ui->ldictionarySize->setText("1200");
//    ui->lfilePCA->setText("pca_10_3_rsift_region_1200");
//    ui->lfileResult->setText("result_10_3_rsift_region_1200");
//    all();

//    ui->ldictionarySize->setText("2000");
//    ui->lfilePCA->setText("pca_10_3_rsift_region_2000");
//    ui->lfileResult->setText("result_10_3_rsift_region_2000");
//    all();

    //-------------------------------
//    ui->lkpSize->setText("5");
//    ui->lkpDensity->setText("3");
//    this->ui->pBGenerateKeyPoints->click();
//    ui->ldictionarySize->setText("200");
//    ui->lfilePCA->setText("pca_5_3_rsift_region_200");
//    ui->lfileResult->setText("result_5_3_rsift_region_200");
//    all();

//    ui->ldictionarySize->setText("500");
//    ui->lfilePCA->setText("pca_5_3_rsift_region_500");
//    ui->lfileResult->setText("result_5_3_rsift_region_500");
//    all();

//    ui->ldictionarySize->setText("1200");
//    ui->lfilePCA->setText("pca_5_3_rsift_region_1200");
//    ui->lfileResult->setText("result_5_3_rsift_region_1200");
//    all();

//    ui->ldictionarySize->setText("2000");
//    ui->lfilePCA->setText("pca_5_3_rsift_region_2000");
//    ui->lfileResult->setText("result_5_3_rsift_region_2000");
//    all();


    //-------------------------------
    //ui->lkpSize->setText("3");
    //ui->lkpDensity->setText("3");
    //this->ui->pBGenerateKeyPoints->click();
    //ui->ldictionarySize->setText("200");
    /*
    ui->lfilePCA->setText("pca_3_3_rsift_only_region_200_modified_Move_2");
    ui->lfileResult->setText("result_3_3_rsift_only_region_200_track");
    all();

    ui->ldictionarySize->setText("500");
    ui->lfilePCA->setText("pca_3_3_rsift_onlu_region_500_modified_Move_2");
    ui->lfileResult->setText("result_3_3_rsift_only_region_500_track");
    all();

    ui->ldictionarySize->setText("1200");
    ui->lfilePCA->setText("pca_3_3_rsift_only_region_1200_modified_Move");
    ui->lfileResult->setText("result_3_3_rsift_only_region_1200_track");
    all();
    */

    //ui->ldictionarySize->setText("2000");
    //ui->lfilePCA->setText("pca_3_3_rsift_only_region_2000_modified_Move_2");
    //ui->lfileResult->setText("result_3_3_rsift_only_region_2000_track");
    //all();

    //-------------------------------
    //-------------------------------
//    ui->lkpSize->setText("20");
//    ui->lkpDensity->setText("2");
//    this->ui->pBGenerateKeyPoints->click();

//    ui->ldictionarySize->setText("200");
//    ui->lfilePCA->setText("pca_20_2_rsift_region_200");
//    ui->lfileResult->setText("result_20_2_rsift_region_200");
//    all();

//    ui->ldictionarySize->setText("500");
//    ui->lfilePCA->setText("pca_20_2_rsift_region_500");
//    ui->lfileResult->setText("result_20_2_rsift_region_500");
//    all();

//    ui->ldictionarySize->setText("1200");
//    ui->lfilePCA->setText("pca_20_2_rsift_region_1200");
//    ui->lfileResult->setText("result_20_2_rsift_region_1200");
//    all();

//    ui->ldictionarySize->setText("2000");
//    ui->lfilePCA->setText("pca_20_2_rsift_region_2000");
//    ui->lfileResult->setText("result_20_2_rsift_region_2000");
//    all();

    //-------------------------------
//    ui->lkpSize->setText("10");
//    ui->lkpDensity->setText("2");
//    this->ui->pBGenerateKeyPoints->click();

//    ui->ldictionarySize->setText("200");
//    ui->lfilePCA->setText("pca_10_2_rsift_region_200");
//    ui->lfileResult->setText("result_10_2_rsift_region_200");
//    all();

//    ui->ldictionarySize->setText("500");
//    ui->lfilePCA->setText("pca_10_2_rsift_region_500");
//    ui->lfileResult->setText("result_10_2_rsift_region_500");
//    all();

//    ui->ldictionarySize->setText("1200");
//    ui->lfilePCA->setText("pca_10_2_rsift_region_1200");
//    ui->lfileResult->setText("result_10_2_rsift_region_1200");
//    all();

//    ui->ldictionarySize->setText("2000");
//    ui->lfilePCA->setText("pca_10_2_rsift_region_2000");
//    ui->lfileResult->setText("result_10_2_rsift_region_2000");
//    all();

    //-------------------------------
//    ui->lkpSize->setText("5");
//    ui->lkpDensity->setText("2");
//    this->ui->pBGenerateKeyPoints->click();

//    ui->ldictionarySize->setText("200");
//    ui->lfilePCA->setText("pca_5_2_rsift_region_200");
//    ui->lfileResult->setText("result_5_2_rsift_region_200");
//    all();

//    ui->ldictionarySize->setText("500");
//    ui->lfilePCA->setText("pca_5_2_rsift_region_500");
//    ui->lfileResult->setText("result_5_2_rsift_region_500");
//    all();

//    ui->ldictionarySize->setText("1200");
//    ui->lfilePCA->setText("pca_5_2_rsift_region_1200");
//    ui->lfileResult->setText("result_5_2_rsift_region_1200");
//    all();

//    ui->lkpSize->setText("5");
//    ui->lkpDensity->setText("2");
//    ui->ldictionarySize->setText("2000");
//    ui->lfilePCA->setText("pca_5_2_rsift_region_2000");
//    ui->lfileResult->setText("result_5_2_rsift_region_2000");
//    all();

    //-------------------------------
//    ui->lkpSize->setText("3");
//    ui->lkpDensity->setText("2");
//    this->ui->pBGenerateKeyPoints->click();

//    ui->ldictionarySize->setText("200");
//    ui->lfilePCA->setText("pca_3_2_rsift_region_200");
//    ui->lfileResult->setText("result_3_2_rsift_region_200");
//    all();

//    ui->ldictionarySize->setText("500");
//    ui->lfilePCA->setText("pca_3_2_rsift_region_500");
//    ui->lfileResult->setText("result_3_2_rsift_region_500");
//    all();

//    ui->ldictionarySize->setText("1200");
//    ui->lfilePCA->setText("pca_3_2_rsift_region_1200");
//    ui->lfileResult->setText("result_3_2_rsift_region_1200");
//    all();

//    ui->ldictionarySize->setText("2000");
//    ui->lfilePCA->setText("pca_3_2_rsift_region_2000");
//    ui->lfileResult->setText("result_3_2_rsift_region_2000");
//    all();

    //Gris
//    ui->cBGrayImgs->setChecked(true);
//    ui->lfilePCA->setText("pca_gray_modified_Move");
//    ui->lfileResult->setText("result_gray_modified_Move");
//    this->ui->pBGenerateSVM->click();
//    this->ui->pBTestSVM->click();
}


