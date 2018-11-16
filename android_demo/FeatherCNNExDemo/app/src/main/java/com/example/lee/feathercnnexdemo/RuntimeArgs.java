package com.example.lee.feathercnnexdemo;

/*
typedef struct runtime_args
{
    int32_t outLoopCnt;
    int32_t loopCnt;
    int32_t num_threads;
    int32_t bSameMean;
    int32_t bGray;
    int32_t b1x1Sgemm;
    int32_t bLowPrecision;
    int32_t viewType;
    char pFname[1024];
    char pModel[1024];
    char pBlob[1024];
    char pSerialFile[1024];
}RUNTIME_ARGS_S;
 */

import java.util.Arrays;

public class RuntimeArgs {
    public int outLoopCnt;
    public int loopCnt;
    public int num_threads;
    public int bSameMean;
    public int bGray;
    public int b1x1Sgemm;
    public int bLowPrecision;
    public int viewType;
    public String pFname;
    public String pModel;
    public String pBlob;
    public String pSerialFile;

    public int getOutLoopCnt() {
        return outLoopCnt;
    }

    public void setOutLoopCnt(int outLoopCnt) {
        this.outLoopCnt = outLoopCnt;
    }

    public int getLoopCnt() {
        return loopCnt;
    }

    public void setLoopCnt(int loopCnt) {
        this.loopCnt = loopCnt;
    }

    public int getNum_threads() {
        return num_threads;
    }

    public void setNum_threads(int num_threads) {
        this.num_threads = num_threads;
    }

    public int getbSameMean() {
        return bSameMean;
    }

    public void setbSameMean(int bSameMean) {
        this.bSameMean = bSameMean;
    }

    public int getbGray() {
        return bGray;
    }

    public void setbGray(int bGray) {
        this.bGray = bGray;
    }

    public int getB1x1Sgemm() {
        return b1x1Sgemm;
    }

    public void setB1x1Sgemm(int b1x1Sgemm) {
        this.b1x1Sgemm = b1x1Sgemm;
    }

    public int getbLowPrecision() {
        return bLowPrecision;
    }

    public void setbLowPrecision(int bLowPrecision) {
        this.bLowPrecision = bLowPrecision;
    }

    public int getViewType() {
        return viewType;
    }

    public void setViewType(int viewType) {
        this.viewType = viewType;
    }

    public String getpFname() {
        return pFname;
    }

    public void setpFname(String pFname) {
        this.pFname = pFname;
    }

    public String getpModel() {
        return pModel;
    }

    public void setpModel(String pModel) {
        this.pModel = pModel;
    }

    public String getpBlob() {
        return pBlob;
    }

    public void setpBlob(String pBlob) {
        this.pBlob = pBlob;
    }

    public String getpSerialFile() {
        return pSerialFile;
    }

    public void setpSerialFile(String pSerialFile) {
        this.pSerialFile = pSerialFile;
    }

    @Override
    public String toString() {
        return "RuntimeArgs{" +
                "outLoopCnt=" + outLoopCnt +
                ", loopCnt=" + loopCnt +
                ", num_threads=" + num_threads +
                ", bSameMean=" + bSameMean +
                ", bGray=" + bGray +
                ", b1x1Sgemm=" + b1x1Sgemm +
                ", bLowPrecision=" + bLowPrecision +
                ", viewType=" + viewType +
                ", pFname='" + pFname + '\'' +
                ", pModel='" + pModel + '\'' +
                ", pBlob='" + pBlob + '\'' +
                ", pSerialFile='" + pSerialFile + '\'' +
                '}';
    }
}

