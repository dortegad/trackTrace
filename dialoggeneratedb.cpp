#include "dialoggeneratedb.h"
#include "ui_dialoggeneratedb.h"

#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

#include "constants.h"

#include <dirent.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;

//-------------------------------------------------------------------------------------------------------------
DialogGenerateDB::DialogGenerateDB(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DialogGenerateDB)
{
    ui->setupUi(this);
}

//-------------------------------------------------------------------------------------------------------------
DialogGenerateDB::~DialogGenerateDB()
{
    delete ui;
}

//-------------------------------------------------------------------------------------------------------------
void DialogGenerateDB::readPersons(const std::string &idsFileName)
{
    std::fstream  myfile (idsFileName.c_str(),std::ios_base::in);

    //Leer cabezera
    std::string a;
    std::getline(myfile, a);

    //Leer frames
    person.clear();
    while (std::getline(myfile, a))
    {
        std::stringstream cadena;
        cadena.str(a);

        int idPerson;
        cadena >> idPerson;
        string namePerson;
        cadena >> namePerson;

        //cout << idPerson << "-" << namePerson <<  std::endl;

        categoryPerson[namePerson].push_back(idPerson);
        person[idPerson] = namePerson;

    }
}


//-------------------------------------------------------------------------------------------------------------
void DialogGenerateDB::readShot(const std::string &videvensFileName)
{
    std::fstream  myfile (videvensFileName.c_str(),std::ios_base::in);

    //Leer cabezera
    std::string a;
    std::getline(myfile, a);

    //Leer frames
    shots.clear();
    while (std::getline(myfile, a)) {
        std::stringstream cadena;
        cadena.str(a);

        int initFrame;
        cadena >> initFrame;
        float initTime;
        cadena >> initTime;
        string shotType;
        cadena >> shotType;

        int endFrame = initFrame;
        float endTime = initTime;
        if (shotType != "CUT")
        {
            cadena >> endFrame;
            cadena >> endTime;
        }

        Shot *shot = new Shot();
        shot->initFrame = initFrame;
        shot->initTime = initTime;
        shot->shotType = shotType;
        shot->endFrame = endFrame;
        shot->endTime = endTime;

        shots.push_back(shot);
        //std::cout << initFrame << "-" << initTime << "-" << shotType << "-" << endFrame << "-" << endTime << std::endl;
    }
    myfile.close();
}


//-------------------------------------------------------------------------------------------------------------
void DialogGenerateDB::viewTracks(std::map <std::string, std::vector<cv::Mat> > &facesTrack)
{
    int faceSize = 64;
    int numFacesCols = 10;

    std::cout << "numPerson" << facesTrack.size() << std::endl;

    std::map <std::string, std::vector< cv::Mat> >::iterator itPersons = facesTrack.begin();
    for ( ; itPersons != facesTrack.end(); itPersons++)
    {
        std::vector<cv::Mat> faces = (*itPersons).second;
        std::string name = (*itPersons).first;
        int numFaces = faces.size();

        std::cout << "numFaces" << numFaces  << std::endl;

        int numFacesRows = numFaces / numFacesCols;
        std::cout << "numFacesRows " << numFacesRows  << std::endl;
        std::cout << "numFacesCols " << numFacesCols  << std::endl;
        if (numFacesRows == 0)
        {
            numFacesRows = 1;
        }
        else
        {
            if (numFaces % numFacesCols)
            {
                numFacesRows++;
            }
        }

        std::cout << "numFacesRows " << numFacesRows  << std::endl;
        std::cout << "numFacesCols " << numFacesCols  << std::endl;

        int w = numFacesCols*faceSize;
        int h = numFacesRows*faceSize;

        cv::Mat imgFaces(h,w,CV_8UC3,cv::Scalar(0,0,0));

        int posFace= 0;
        for (int j=0; j<numFacesRows; j++)
        {
            for (int i=0; i<numFacesCols; i++)
            {

                if (posFace < faces.size())
                {
                    cv::Mat face = faces[posFace];

                    cv::resize(face,face,cv::Size(faceSize,faceSize));

                    face.copyTo(imgFaces.colRange(i*faceSize,(i*faceSize)+faceSize).rowRange(j*faceSize,(j*faceSize)+faceSize));

                    posFace++;
                }
            }
        }

        cv::imshow(name,imgFaces);
    }
}

//-------------------------------------------------------------------------------------------------------------
void DialogGenerateDB::readTracks(const std::string &idsFileName,
                              const std::string &videvenstsFileName,
                              const std::string &facetrackFileName)
{
    readShot(videvenstsFileName);

    readPersons(idsFileName);

    std::fstream  myfile (facetrackFileName.c_str(),std::ios_base::in);

    tracksFrames.clear();

    //Leer cabezera
    std::string a;
    std::getline(myfile, a);
    std::getline(myfile, a);

    //Leer frames
    while (std::getline(myfile, a)) {
        std::stringstream cadena;
        cadena.str(a);

        int numFrame;
        cadena >> numFrame;
        float time;
        cadena >> time;
        int numTracks;
        cadena >> numTracks;

        vector < TracksFame * > tracks;
        if (numTracks > 0)
        {
            for (int i=0; i<numTracks; i++)
            {
                int p, x, y, w, h;
                cadena >> p;
                cadena >> x;
                cadena >> y;
                cadena >> w;
                cadena >> h;

                //cout << p << "-" << x << "-" << y << "-" << w << "-" << h << std::endl;

                TracksFame *track = new TracksFame();
                track->rect = cv::Rect(x,y,h,w);
                track->type = person[p];

                tracks.push_back(track);

                //lectura datos ley rey
                int leyex, leyey, reyex, reyey;
                cadena >> leyex;
                cadena >> leyey;
                cadena >> reyex;
                cadena >> reyey;
            }
        }
        tracksFrames.push_back(tracks);
    }
    myfile.close();
}

//-------------------------------------------------------------------------------------------------------------
void DialogGenerateDB::saveTracks(std::map <std::string, std::vector<cv::Mat> > &facesTrack,
                              int numTrack,
                              int imgSize)
{
    std::map <std::string, std::vector< cv::Mat> >::iterator itPersons = facesTrack.begin();
    for ( ; itPersons != facesTrack.end(); itPersons++)
    {
        std::vector<cv::Mat> faces = (*itPersons).second;
        std::string personName = (*itPersons).first;

        std::stringstream dirImages;
        dirImages << ui->dirImgGenerate->text().toStdString();

        DIR* dir = opendir(dirImages.str().c_str());
        if (dir)
            closedir(dir);
        else
            mkdir(dirImages.str().c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);


        if (ui->cBSeparateImgs->isChecked())
        {
            dirImages << "/" << personName;
            dir = opendir(dirImages.str().c_str());
            if (dir)
                closedir(dir);
            else
                mkdir(dirImages.str().c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        }
        dirImages << "/";

        int numFaces = faces.size();
        std::cout << "write " << numFaces << " faces in directory " << dirImages.str() << std::endl;

        for (int i=0; i<numFaces; i++)
        {
            //contabilizamos cuanto hemos añadido de cada clase para no superar el maximo
            countNumFacesOfPerson[personName]++;
            if ((countNumFacesOfPerson[personName] < ui->lMaxFilesOfClass->text().toInt()) ||
                (ui->lMaxFilesOfClass->text().toInt() == 0))
            {
                std::stringstream fileFaceName;
                fileFaceName << dirImages.str() << personName << "_" << numTrack << "_" << i << "." << Constants::IMAGE_EXT;

                cv::Mat face = faces[i];
                if ((face.cols > 0) && (face.rows >0))
                {
                    cv::resize(face,face,cv::Size(imgSize,imgSize));
                    cv::imwrite(fileFaceName.str(),face);
                }
            }
        }
    }
}

//-------------------------------------------------------------------------------------------------------------
void DialogGenerateDB::readFiltersId(const std::string &fileNameFilterIds,
                                 std::vector < std::string > &filterIds)
{
    std::ifstream file(fileNameFilterIds.c_str());
    if (file.is_open())
    {
        std::string line;
        while ( getline (file,line) )
        {
            filterIds.push_back(line);
        }
        file.close();
    }
}


//-------------------------------------------------------------------------------------------------------------
void DialogGenerateDB::on_pBGenerateImages_clicked()
{
    cv::CascadeClassifier haar_cascade;
    haar_cascade.load("haarcascade_frontalface_default.xml");

    //Leemos toda la informacion del video (Saltos, y caras por frame)
    std::string fileNameVideo = ui->lvideoFileName->text().toStdString();
    std::string onlyFileNameVideo = fileNameVideo.substr(0,fileNameVideo.find_last_of("."));
    std::stringstream fileNameEvents;
    fileNameEvents << onlyFileNameVideo << "." << Constants::EVENTS_FILE_EXT;
    std::stringstream fileNameIDs;
    fileNameIDs << onlyFileNameVideo << "." << Constants::IDS_FILE_EXT;
    std::stringstream fileNameFaceTrack;
    fileNameFaceTrack << onlyFileNameVideo << "." << Constants::FACE_TRACK_FILE_EXT;

    readTracks(fileNameEvents.str(),
               fileNameIDs.str(),
               fileNameFaceTrack.str());

    //Si no queremos que se genere determiandas caras (no cara o desconodios por ejemplo)
    std::vector < std::string > filterIds;
    readFiltersId(ui->dirFileFilterId->text().toStdString(),filterIds);

    //Tendremos en cuenta cuenta cuantas caras hemos añadido de cada tipo
    countNumFacesOfPerson.clear();

    cv::VideoCapture vid;
    vid.open(fileNameVideo);

    int numShot = 0;
    Shot *shot = shots[numShot];

    std::map <std::string, std::vector< cv::Mat> > facesTrack;

    // LOS VIDEOS Y LA INFORMACION LEIDA TIENE DESFASE DE 25 CAPITULO_2 int desfase = 25 - 2;
    int numFrame = 0;
    int desfase = ui->lEDesfase->text().toInt();

    int total_frame = vid.get(cv::CAP_PROP_FRAME_COUNT);


    while((vid.isOpened()) && (numFrame < total_frame-1))
    {
        cv::Mat frame;
        vid >> frame;

        cv::Mat auxFrame;
        cv::cvtColor(frame,auxFrame,CV_BGR2BGRA);

        if (tracksFrames.size() > numFrame-desfase)
        {
            int numTrack = tracksFrames[numFrame-desfase].size();
            for (int i=0; i<numTrack; i++)
            {
                //Si el id de la cara no es ninguno de los que nos interesan no recortamos esa region ni la presentamos
                std::string person = tracksFrames[numFrame-desfase][i]->type;
                if (std::find(filterIds.begin(),filterIds.end(),person) == filterIds.end())
                    continue;

                int pixelExpand = ui->lPixelsExpand->text().toInt();
                cv::Mat face;
                cv::Rect rect = tracksFrames[numFrame-desfase][i]->rect;
                if (!ui->cBViolaVerify->isChecked())
                {
                    rect.x = rect.x-pixelExpand;
                    rect.y = rect.y-pixelExpand;
                    rect.width = rect.width+(pixelExpand*2);
                    rect.height = rect.height+(pixelExpand*3);
                    rect &= cv::Rect(0,0,frame.cols,frame.rows); //ajuste para no salirse de la iamgen
                    face = auxFrame(rect);
                    facesTrack[person].push_back(face);
                }
                else
                {
                    //Recortamos la region y las almacenamos ordenadas por personaje
                    rect.x = rect.x-50;
                    rect.y = rect.y-50;
                    rect.width = rect.width+(50*2);
                    rect.height = rect.height+(50*3);
                    rect &= cv::Rect(0,0,frame.cols,frame.rows); //ajuste para no salirse de la iamgen
                    face = auxFrame(rect);

                    std::vector< cv::Rect_<int> > faces;
                    haar_cascade.detectMultiScale(face, faces);
                    int numRects = faces.size();
                    if (numRects > 0)
                    {
                        cv::Rect reg = faces[0];
                        for (int i=1; i<numRects; i++)
                            reg &= faces[i];

                        if (ui->cBExpandImgs->isChecked())
                        {
                            reg.x = reg.x-pixelExpand;
                            reg.y = reg.y-pixelExpand;
                            reg.width = reg.width+(pixelExpand*2);
                            reg.height = reg.height+(pixelExpand*2);
                            reg &= cv::Rect(0,0,face.cols,face.rows);
                        }

                        facesTrack[person].push_back(face(reg));
                    }
                }

                //Presentamos la región
                cv::rectangle(frame,rect,cv::Scalar(255,0,255));
                int fontFace = cv::FONT_HERSHEY_SIMPLEX;
                double fontScale = 0.5;
                int thickness = 2;
                cv::putText(frame, person, cv::Point(rect.x,rect.y), fontFace, fontScale, cv::Scalar::all(255), thickness,8);

            }
        }

        if (numFrame-desfase == shot->initFrame)
        {
            numShot++;
            if (numShot < shots.size())
            {
                shot = shots[numShot];

                if (!ui->cBOnlyVisualTraking->isChecked())
                    saveTracks(facesTrack,numShot,ui->lImgSize->text().toInt());

                facesTrack.clear();
            }
        }

        if (ui->cBOnlyVisualTraking->isChecked())
        {
            cv::imshow("video",frame);
            cv::waitKey(3);
        }

        numFrame++;
    }

    vid.release();
}
