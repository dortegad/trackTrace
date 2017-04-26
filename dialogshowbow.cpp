#include "dialogshowbow.h"
#include "ui_dialogshowbow.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

DialogShowBOW::DialogShowBOW(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DialogShowBOW)
{
    ui->setupUi(this);
}

//------------------------------------------------------------------------------------------
DialogShowBOW::~DialogShowBOW()
{
    delete ui;
}


//------------------------------------------------------------------------------------------
void DialogShowBOW::showImage(cv::Mat &img)
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
    ui->lImagen->setScaledContents(true);
    ui->lImagen->setPixmap(QPixmap::fromImage(imagenQT));
    ui->lImagen->setGeometry(0,0,imagenQT.width(),imagenQT.height());
}

//------------------------------------------------------------------------------------------
void DialogShowBOW::generateImageBOW(cv::Mat &img,
                                     std::vector<cv::KeyPoint> &keypoints,
                                     std::vector< std::vector< int > > &pointIdxsOfClusters)
{
    this->imgBow = cv::Mat_<uchar>::zeros(img.rows,img.cols);
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
}

//------------------------------------------------------------------------------------------
void DialogShowBOW::showBOW(cv::Mat &img,
                            std::vector<cv::KeyPoint> &keypoints,
                            std::vector< std::vector< int > > &pointIdxsOfClusters)
{
    generateImageBOW(img,
                     keypoints,
                     pointIdxsOfClusters);

    this->image = img.clone();

    ui->vSNumBow->setMaximum(pointIdxsOfClusters.size());
    int value = ui->vSNumBow->maximum()/2;
    ui->vSNumBow->setValue(ui->vSNumBow->maximum()/2);
    ui->lENumBow->setText(QString::number(value));

    cv::Mat unBow = cv::Mat(this->image.rows,this->image.cols,this->image.type(),cv::Scalar(255,255,255));
    this->image.copyTo(unBow,(this->imgBow != ui->vSNumBow->value()));
    showImage(unBow);
    this->show();
}

//------------------------------------------------------------------------------------------
void DialogShowBOW::on_vSNumBow_sliderMoved(int position)
{
    int value = ui->vSNumBow->value();
    ui->lENumBow->setText(QString::number(value));

    cv::Mat unBow = cv::Mat(this->image.rows,this->image.cols,this->image.type(),cv::Scalar(255,255,255));
    this->image.copyTo(unBow,(this->imgBow != value));
    showImage(unBow);
}

//------------------------------------------------------------------------------------------
void DialogShowBOW::on_lENumBow_returnPressed()
{
    int value = ui->lENumBow->text().toInt();
    ui->vSNumBow->setValue(value);

    cv::Mat unBow = cv::Mat(this->image.rows,this->image.cols,this->image.type(),cv::Scalar(255,255,255));
    this->image.copyTo(unBow,(this->imgBow != value));
    showImage(unBow);
}
