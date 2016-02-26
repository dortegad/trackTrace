#ifndef DIALOGGENERATEDB_H
#define DIALOGGENERATEDB_H

#include <QDialog>


#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

class TracksFame
{
public:
    std::string type;
    cv::Rect rect;
};

class Shot
{
public:
    int initFrame;
    float initTime;
    std::string shotType;
    int endFrame;
    float endTime;
};


namespace Ui {
class DialogGenerateDB;
}

class DialogGenerateDB : public QDialog
{
    Q_OBJECT

public:
    std::map <int, std::string> person;
    std::map <std::string, std::vector <int> > categoryPerson;
    std::vector < std::vector<TracksFame*> > tracksFrames;
    std::vector < Shot* > shots;

    std::map < std::string, int > countNumFacesOfPerson;

    explicit DialogGenerateDB(QWidget *parent = 0);
    ~DialogGenerateDB();

    void saveTracks(std::map<std::string, std::vector<cv::Mat> > &facesTrack, int numTrack, int imgSize);

    void readTracks(const std::string &idsFileName, const std::string &videvenstsFileName, const std::string &facetrackFileName);

    void readShot(const std::string &videvensFileName);

    void readPersons(const std::string &idsFileName);

    void viewTracks(std::map<std::string, std::vector<cv::Mat> > &facesTrack);

    void readFiltersId(const std::string &fileNameFilterIds, std::vector<std::string> &filterIds);
private slots:
    void on_pBGenerateImages_clicked();

private:
    Ui::DialogGenerateDB *ui;
};

#endif // DIALOGGENERATEDB_H
