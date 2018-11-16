package com.example.lee.feathercnnexdemo;

import android.net.Uri;
import android.util.Log;
import java.io.FileNotFoundException;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.content.ContentResolver;
import android.widget.TextView;

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
    }

    @Override
    public void onClick(View v) {
        if (v.getId() == R.id.show) {
            EditText imgEdit = (EditText)findViewById(R.id.editImg);
            String url = "file:///sdcard/lj/"+imgEdit.getText().toString();
            Log.d("lee-jni-test", url);
            ContentResolver cr = this.getContentResolver();
            try {
                Bitmap bitmap = BitmapFactory.decodeStream(cr.openInputStream(Uri.parse(url)));
                imageView.setImageBitmap(bitmap);
            } catch (FileNotFoundException e) {
                Log.e("Exception", e.getMessage(),e);
            }
        } else {
            Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.test);
            //getEdge(bitmap);
            EditText loopEdit = (EditText)findViewById(R.id.editLoop);
            EditText threadsEdit = (EditText)findViewById(R.id.editThreads);
            EditText imgEdit = (EditText)findViewById(R.id.editImg);
            EditText modelEdit = (EditText)findViewById(R.id.editModel);
            EditText blobEdit = (EditText)findViewById(R.id.editBlob);
            args.outLoopCnt = 1;
            args.loopCnt = Integer.parseInt(loopEdit.getText().toString());
            args.num_threads = Integer.parseInt(threadsEdit.getText().toString());;
            args.bSameMean = 1;
            args.bGray = 0;
            args.b1x1Sgemm = 1;
            args.bLowPrecision = 0;
            args.viewType = 3;
            args.pFname = "/sdcard/lj/"+imgEdit.getText().toString();
            args.pModel = "/sdcard/lj/"+modelEdit.getText().toString();
            args.pBlob = blobEdit.getText().toString();
            args.pSerialFile = "/sdcard/lj/feather.key";

            int avgTime = FeatherCNNExTest(args, bitmap);
            TextView avgTimeText = (TextView)findViewById(R.id.textViewAvgTime);
            avgTimeText.setText("AvgTime: "+avgTime+" ms");
            imageView.setImageBitmap(bitmap);
        }
    }

    public native void getEdge(Object bitmap);
    public native int FeatherCNNExTest(RuntimeArgs obj, Object bitmap);
}