#ifndef DIALOGSHOWSIFT_H
#define DIALOGSHOWSIFT_H

#include <QDialog>

#include <opencv2/core/core.hpp>

namespace Ui {
class DialogShowSift;
}

class DialogShowSift : public QDialog
{
    Q_OBJECT

public:
    std::vector < cv::KeyPoint > keyPoints;
    std::vector<float> scores;
    cv::Mat imgSift;
    float th;
    std::vector <std::string> files;
    int sizeSift;
    int DensitySift;
    int numfiles;
    int actualFile;
    bool detect;
    float paso;
    float min;

    explicit DialogShowSift(QWidget *parent = 0);
    ~DialogShowSift();

    void showImage(cv::Mat &img);

    void showSift();
    void showSift(std::vector < std::string> &files,
                  int sizeSift,
                  int DensitySift,
                  bool detect = false);
    void showSiftActualFile();
    void generateSift();

private slots:
    void on_vSThresholdSift_sliderMoved(int position);

    void on_lEThresholdSift_editingFinished();

    void on_pBPreFile_clicked();

    void on_pBSigFile_clicked();

private:
    Ui::DialogShowSift *ui;
};

#endif // DIALOGSHOWSIFT_H
