package com.example.lee.feathercnnexdemo;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;

public class JniActivity extends AppCompatActivity implements View.OnClickListener {
    private ImageView imageView;
    private RuntimeArgs args = new RuntimeArgs();

    static {
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_jni);
        imageView = findViewById(R.id.imageView);
        findViewById(R.id.show).setOnClickListener(this);
        findViewById(R.id.process).setOnClickListener(this);
        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.test);
        imageView.setImageBitmap(bitmap);
    }

    @Override
    public void onClick(View v) {
        if (v.getId() == R.id.show) {
            Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.test);
            imageView.setImageBitmap(bitmap);
            args.outLoopCnt = 1;
            args.loopCnt = 50;
            args.num_threads = 4;
            args.bSameMean = 1;
            args.bGray = 0;
            args.b1x1Sgemm = 1;
            args.bLowPrecision = 0;
            args.viewType = 3;
            args.pFname = "/sdcard/lj/74.png";
            args.pModel = "/sdcard/lj/rokid_detection_out_0.feathermodel";
            args.pBlob = "detection_out";
            args.pSerialFile = "/sdcard/lj/feather.key";

            FeatherCNNExTest(args);
        } else {
            Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.test);
            getEdge(bitmap);
            imageView.setImageBitmap(bitmap);
        }
    }

    public native void getEdge(Object bitmap);
    public native int FeatherCNNExTest(RuntimeArgs obj);
}