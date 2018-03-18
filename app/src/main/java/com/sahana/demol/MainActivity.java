package com.sahana.demol;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.Editable;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    private TextView textview;
    private EditText edittext;
    private Button button;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textview = findViewById(R.id.textView2);

        edittext = findViewById(R.id.editText);

        button = findViewById(R.id.button);

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Editable e = edittext.getText();
                String s = e.toString();
                if(e.charAt(e.length()-1) == 'o' && e.charAt(e.length()-2) == 'c') {
                    textview.setText("that's fake news!!!");
                }

            }
        });
    }
}
