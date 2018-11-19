package com.example.lee.feathercnnexdemo;

import java.util.ArrayList;
import java.util.List;

import android.content.Context;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Handler;
import android.util.Log;
import java.io.FileNotFoundException;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.ImageView;
import android.content.ContentResolver;
import android.widget.TextView;
import android.widget.Spinner;
import android.widget.ArrayAdapter;

public class JniActivity extends AppCompatActivity implements View.OnClickListener {
    class TestAsyncTask extends AsyncTask<Void, Integer, Boolean> {
        private Context mContext = null;
        int avgTime = 0;
        RuntimeArgs args = new RuntimeArgs();
        EditText imgEdit = findViewById(R.id.editImg);
        ImageView imageView = findViewById(R.id.imageView);
        Bitmap bitmap;
        public TestAsyncTask(Context context) {
            mContext = context;
        }

        protected void onPreExecute() {
            super.onPreExecute();
        }

        protected Boolean doInBackground(Void... params) {
            String url = "file:///sdcard/lj/"+imgEdit.getText().toString();
            ContentResolver cr = mContext.getContentResolver();
            try {
                bitmap = BitmapFactory.decodeStream(cr.openInputStream(Uri.parse(url)));
                EditText loopEdit = findViewById(R.id.editLoop);
                EditText threadsEdit = findViewById(R.id.editThreads);
                EditText modelEdit = findViewById(R.id.editModel);
                EditText blobEdit = findViewById(R.id.editBlob);
                CheckBox grayCheckBox = findViewById(R.id.checkBoxGray);
                CheckBox lowPCheckBox = findViewById(R.id.checkBoxLowP);
                CheckBox sameMCheckBox = findViewById(R.id.checkBoxSameMean);
                Spinner viewType = findViewById(R.id.spinnerViewType);
                args.outLoopCnt = 1;
                args.loopCnt = Integer.parseInt(loopEdit.getText().toString());
                args.num_threads = Integer.parseInt(threadsEdit.getText().toString());;
                args.bSameMean = sameMCheckBox.isChecked()?1:0;
                args.bGray = grayCheckBox.isChecked()?1:0;
                args.b1x1Sgemm = 1;
                args.bLowPrecision = lowPCheckBox.isChecked()?1:0;
                args.viewType = viewType.getSelectedItemPosition();
                args.pFname = "/sdcard/lj/"+imgEdit.getText().toString();
                args.pModel = "/sdcard/lj/"+modelEdit.getText().toString();
                args.pBlob = blobEdit.getText().toString();
                args.pSerialFile = "/sdcard/lj/feather.key";
                avgTime = FeatherCNNExTest(args, bitmap);
            } catch (FileNotFoundException e) {
                Log.e("Exception", e.getMessage(),e);
            }
            return true;
        }

        protected void onProgressUpdate(Integer... values) {
            super.onProgressUpdate(values);
        }

        protected void onPostExecute(Boolean aBoolean) {
            super.onPostExecute(aBoolean);
            imageView.setImageBitmap(bitmap);
            TextView avgTimeText = findViewById(R.id.textViewAvgTime);
            avgTimeText.setText("AvgTime: "+avgTime+" ms");
        }
    }

    private List<String> getData() {
        List<String> dataList = new ArrayList<String>();
        dataList.add("Empty");
        dataList.add("Raw");
        dataList.add("Label");
        dataList.add("Rect");
        return dataList;
    }

    static {
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_jni);
        findViewById(R.id.show).setOnClickListener(this);
        findViewById(R.id.process).setOnClickListener(this);
        Spinner viewType = findViewById(R.id.spinnerViewType);
        ArrayAdapter<String> adapter = new ArrayAdapter<String>(
                this, android.R.layout.simple_spinner_item,
                getData());
        viewType.setAdapter(adapter);
        viewType.setSelection(3, true);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

    @Override
    public void onClick(View v) {
        if (v.getId() == R.id.show) {
            ImageView imageView = findViewById(R.id.imageView);
            EditText imgEdit = (EditText)findViewById(R.id.editImg);
            String url = "file:///sdcard/lj/"+imgEdit.getText().toString();
            ContentResolver cr = this.getContentResolver();
            try {
                Bitmap bitmap = BitmapFactory.decodeStream(cr.openInputStream(Uri.parse(url)));
                imageView.setImageBitmap(bitmap);
            } catch (FileNotFoundException e) {
                Log.e("Exception", e.getMessage(),e);
            }
        } else {
            new TestAsyncTask(this).execute();
        }
    }

    public native int FeatherCNNExTest(RuntimeArgs obj, Object bitmap);
}