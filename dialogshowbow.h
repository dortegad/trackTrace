#ifndef DIALOGSHOWBOW_H
#define DIALOGSHOWBOW_H

#include <QDialog>

#include <opencv2/core/core.hpp>

namespace Ui {
class DialogShowBOW;
}

class DialogShowBOW : public QDialog
{
    Q_OBJECT

public:
    explicit DialogShowBOW(QWidget *parent = 0);
    ~DialogShowBOW();

    void showBOW(cv::Mat &img,
                 std::vector<cv::KeyPoint> &keypoints,
                 std::vector< std::vector< int > > &pointIdxsOfClusters);


    void generateImageBOW(cv::Mat &img,
                         std::vector<cv::KeyPoint> &keypoints,
                         std::vector< std::vector< int > > &pointIdxsOfClusters);

private slots:

    void on_vSNumBow_sliderMoved(int position);

    void on_lENumBow_returnPressed();

private:
    Ui::DialogShowBOW *ui;
    cv::Mat_<uchar> imgBow;
    cv::Mat image;

    void showImage(cv::Mat &img);
};

#endif // DIALOGSHOWBOW_H
