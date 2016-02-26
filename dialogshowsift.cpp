#include "dialogshowsift.h"
#include "ui_dialogshowsift.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <mwtracktrace.h>

#include <util_sift.h>

//------------------------------------------------------------------------------------------
DialogShowSift::DialogShowSift(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DialogShowSift)
{
    ui->setupUi(this);
}

//------------------------------------------------------------------------------------------
DialogShowSift::~DialogShowSift()
{
    delete ui;
}

//------------------------------------------------------------------------------------------
void DialogShowSift::showImage(cv::Mat &img)
{
    cv::Mat imgMostrar;
    if (img.channels() == 1)
        cv::cvtColor(img,imgMostrar,CV_GRAY2RGB);
    else
    {
        cv::cvtColor(img,img,CV_RGB2BGR);
        imgMostrar = img.clone();
    }
    QImage imagenQT = QImage((const unsigned char *)imgMostrar.data, imgMostrar.cols, imgMostrar.rows, imgMostrar.step, QImage::Format_RGB888);
    ui->lImagenSift->setScaledContents(true);
    ui->lImagenSift->setPixmap(QPixmap::fromImage(imagenQT));
    ui->lImagenSift->setGeometry(0,0,imagenQT.width(),imagenQT.height());
}

//------------------------------------------------------------------------------------------
void DialogShowSift::generateSift()
{
    cv::Mat imgAux = imgSift.clone();
    int numPoints = keyPoints.size();
    for (int i=0; i<numPoints; i++)
    {
        float score = scores[i];
        float umbral = ui->lEThresholdSift->text().toFloat();
        if (score > umbral)
        {
            cv::KeyPoint kp = keyPoints[i];
            cv::Point2f p = kp.pt;
            cv::Rect rect(p.x,p.y,1,1);
            cv::rectangle(imgAux,rect,cv::Scalar(0,0,255),-1,1);
        }
    }
    showImage(imgAux);
}

//------------------------------------------------------------------------------------------
void DialogShowSift::showSift(std::vector < std::string> &files,
                              int sizeSift,
                              int DensitySift,
                              bool detect)
{
   this->sizeSift = sizeSift;
   this->DensitySift = DensitySift;
   this->files = files;
   this->detect = detect;

   numfiles = files.size();
   actualFile = 0;

   showSiftActualFile();
}

//------------------------------------------------------------------------------------------
void DialogShowSift::showSiftActualFile()
{
    cv::Mat img = cv::imread(files[actualFile],cv::IMREAD_GRAYSCALE);

    cv::Mat descriptors;
    if(detect)
    {
        UTIL_Sift::computeAndDetectSift(img,
                                        this->keyPoints,descriptors,
                                        this->scores);
    }
    else
    {
        UTIL_Sift::createKeyPoints(img,
                                   this->sizeSift,
                                   this->DensitySift,
                                   this->keyPoints);

        UTIL_Sift::computeSift(img,this->keyPoints,descriptors);

        UTIL_Sift::scoresDescriptorsSift(descriptors,this->scores);
    }

    float max = *(std::max_element(this->scores.begin(),this->scores.end()));
    min = *(std::min_element(this->scores.begin(),this->scores.end()));
    float dif = max - min;
    paso = dif / 100;


    cv::cvtColor(img,this->imgSift,CV_GRAY2BGR);

    this->showSift();
}

//------------------------------------------------------------------------------------------
void DialogShowSift::showSift()
{
    th = ui->lEThresholdSift->text().toFloat();

    generateSift();
    this->show();
}

//------------------------------------------------------------------------------------------
void DialogShowSift::on_vSThresholdSift_sliderMoved(int position)
{
    int value = ui->vSThresholdSift->value();
    th = value;
    ui->lEThresholdSift->setText(QString::number((th*paso)+min));

    generateSift();
}

//------------------------------------------------------------------------------------------
void DialogShowSift::on_lEThresholdSift_editingFinished()
{
    th = ui->lEThresholdSift->text().toFloat();

    ui->vSThresholdSift->setValue((th/paso)+min);

    generateSift();
    this->show();
}

//------------------------------------------------------------------------------------------
void DialogShowSift::on_pBPreFile_clicked()
{
    actualFile = actualFile <= 0? 0 : --actualFile;
    showSiftActualFile();
}

//------------------------------------------------------------------------------------------
void DialogShowSift::on_pBSigFile_clicked()
{
    actualFile = actualFile >= numfiles? numfiles : ++actualFile;
    showSiftActualFile();
}
