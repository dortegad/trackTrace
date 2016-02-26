#include <QApplication>
#include "mwtracktrace.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MWTrackTrace w;
    w.show();
    
    return a.exec();
}
