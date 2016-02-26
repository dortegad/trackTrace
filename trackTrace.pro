#-------------------------------------------------
#
# Project created by QtCreator 2015-06-23T18:26:14
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = trackTrace
TEMPLATE = app

SOURCES += main.cpp\
        mwtracktrace.cpp \
    dialogshowbow.cpp \
    dialogshowsift.cpp \
    util_sift.cpp \
    dialoggeneratedb.cpp \
    util_bow.cpp \
    constants.cpp

HEADERS  += mwtracktrace.h \
    dialogshowbow.h \
    dialogshowsift.h \
    util_sift.h \
    dialoggeneratedb.h \
    util_bow.h \
    constants.h

FORMS    += mwtracktrace.ui \
    dialogshowbow.ui \
    dialogshowsift.ui \
    dialoggeneratedb.ui

QT_CONFIG -= no-pkg-config
CONFIG += link_pkgconfig
PKGCONFIG += opencv
