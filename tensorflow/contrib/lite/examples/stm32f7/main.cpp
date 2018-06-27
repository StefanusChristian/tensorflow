#include "mbed.h"
#include "TS_DISCO_F746NG.h"
#include "LCD_DISCO_F746NG.h" #include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/interpreter.h"

LCD_DISCO_F746NG lcd;
TS_DISCO_F746NG ts;

static char error_buf[1024];

class Reporter:public tflite::ErrorReporter {
 public:
  virtual ~Reporter(){}
  int Report(const char* format, va_list args) {
    int len = vsprintf(error_buf, format, args);
    ptr += len;
  }

  char* ptr = error_buf;
};

Reporter reporter;

int main()
{
    TS_StateTypeDef TS_State;
    uint16_t x, y;
    uint8_t text[30];
    uint8_t status;
    uint8_t idx;
    uint8_t cleared = 0;
    uint8_t prev_nb_touches = 0;

    lcd.DisplayStringAt(0, LINE(5), (uint8_t *)"TOUCHSCREEN DEMO", CENTER_MODE);
    wait(1);

    status = ts.Init(lcd.GetXSize(), lcd.GetYSize());
#if 0 
    if (status != TS_OK) {
        lcd.Clear(LCD_COLOR_RED);
        lcd.SetBackColor(LCD_COLOR_RED);
        lcd.SetTextColor(LCD_COLOR_WHITE);
        lcd.DisplayStringAt(0, LINE(5), (uint8_t *)"TOUCHSCREEN INIT FAIL", CENTER_MODE);
    } else {
        lcd.Clear(LCD_COLOR_GREEN);
        lcd.SetBackColor(LCD_COLOR_GREEN);
        lcd.SetTextColor(LCD_COLOR_WHITE);
        lcd.DisplayStringAt(0, LINE(5), (uint8_t *)"TOUCHSCREEN INIT OK", CENTER_MODE);
    }

    wait(1);
#endif
    lcd.SetFont(&Font12);
    lcd.SetBackColor(LCD_COLOR_BLACK);
    lcd.SetTextColor(LCD_COLOR_WHITE);




    TfLiteRegistration reg_add = {nullptr, nullptr, nullptr, nullptr};
    reg_add.prepare = [](TfLiteContext* context, TfLiteNode* node) {
      TfLiteTensor* tensorIn0 = &context->tensors[node->inputs->data[0]];
      // TODO(aselle): Check if tensorIn1 is the same size as tensorOut
      // and that tensorIn0 and tensorIn1 and tensorOut are all float32 type.
      TfLiteTensor* tensorOut = &context->tensors[node->outputs->data[0]];
      TfLiteIntArray* newSize = TfLiteIntArrayCopy(tensorIn0->dims);
      TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, tensorOut, newSize));
      return kTfLiteOk;
    };
    reg_add.invoke = [](TfLiteContext* context, TfLiteNode* node) {
      TfLiteTensor* a0 = &context->tensors[node->inputs->data[0]];
      TfLiteTensor* a1 = &context->tensors[node->inputs->data[1]];
      TfLiteTensor* a2 = &context->tensors[node->outputs->data[0]];
      int count = a0->bytes / sizeof(float);
      float* a = a0->data.f;
      float* b = a1->data.f;
      float* c = a2->data.f;
      float* c_end = c + count;
      for(; c != c_end; c++, a++, b++)
        *c = *a + *b;
      return kTfLiteOk;
    };

    tflite::Interpreter interpreter(&reporter);
    int base;
    interpreter.AddTensors(4, &base);
    interpreter.AddNodeWithParameters({0, 1}, {2}, nullptr, 0, nullptr,
                                                &reg_add);
    interpreter.AddNodeWithParameters({2, 2}, {3}, nullptr, 0, nullptr,
                                                &reg_add);
    TfLiteQuantizationParams quantized;
    interpreter.SetTensorParametersReadWrite(0, kTfLiteFloat32, "", {3},
                                             quantized);
    interpreter.SetTensorParametersReadWrite(1, kTfLiteFloat32, "", {3},
                                             quantized);
    interpreter.SetTensorParametersReadWrite(2, kTfLiteFloat32, "", {3},
                                             quantized);
    interpreter.SetTensorParametersReadWrite(3, kTfLiteFloat32, "", {3},
                                             quantized);
    interpreter.SetInputs({0, 1});
    interpreter.SetOutputs({3});
    TfLiteStatus allocateStatus = interpreter.AllocateTensors();
    float* aIn = interpreter.typed_tensor<float>(0);
    float* bIn = interpreter.typed_tensor<float>(1);
    aIn[0] = 1.f;
    aIn[1] = 2.f;
    aIn[2] = 3.f;
    bIn[0] = -3.f;
    bIn[1] = -2.f;
    bIn[2] = 11.f;
    interpreter.Invoke();
    float* cOut = interpreter.typed_tensor<float>(3);



    while(1) {

        ts.GetState(&TS_State);
        if (TS_State.touchDetected) {
            // Clear lines corresponding to old touches coordinates
            if (TS_State.touchDetected < prev_nb_touches) {
                for (idx = (TS_State.touchDetected + 1); idx <= 5; idx++) {
                    lcd.ClearStringLine(idx);
                }
            }
            prev_nb_touches = TS_State.touchDetected;

            cleared = 0;
            
  
            sprintf((char*)text, "Touches: %d", TS_State.touchDetected);
            lcd.DisplayStringAt(0, LINE(0), (uint8_t *)&text, LEFT_MODE);

            for (idx = 0; idx < TS_State.touchDetected; idx++) {
                x = TS_State.touchX[idx];
                y = TS_State.touchY[idx];
                sprintf((char*)text, "Touch %d: x=%d y=%d    ", idx+1, x, y);
                lcd.DisplayStringAt(0, LINE(idx+1), (uint8_t *)&text, LEFT_MODE);
            }

            lcd.DrawPixel(TS_State.touchX[0], TS_State.touchY[0], LCD_COLOR_ORANGE);
            if(TS_State.touchDetected >= 2){
                float diffx = (TS_State.touchX[1] - TS_State.touchX[0]);
                float diffy = (TS_State.touchY[1] - TS_State.touchY[0]);
                float diameter = sqrt(diffx*diffx+diffy*diffy);
                int rint = (int)(diameter * .5f);
                lcd.DrawCircle((TS_State.touchX[0] + TS_State.touchX[1])/2,
                            (TS_State.touchY[0] + TS_State.touchY[1])/2, rint);
            }
        } else {
            if (!cleared) {
                lcd.Clear(LCD_COLOR_BLUE);
                sprintf((char*)text, "Touches: 0");
                lcd.DisplayStringAt(0, LINE(0), (uint8_t *)&text, LEFT_MODE);
                cleared = 1;
            }
        }
        sprintf((char*)text, "tflite input: 2 * ([1, 2, 3] + [-3,-2,11]");
        lcd.DisplayStringAt(0, LINE(17), (uint8_t *)&text, LEFT_MODE);
        if (cOut) {
        sprintf((char*)text, "tflite output: %f %f %f", cOut[0], cOut[1], cOut[2]);
        lcd.DisplayStringAt(0, LINE(18), (uint8_t *)&text, LEFT_MODE);
        }
        sprintf((char*)text, "Status: %d Compiler: %s", allocateStatus, __VERSION__);
        lcd.DisplayStringAt(0, LINE(20), (uint8_t *)&error_buf, LEFT_MODE);
        
    }
}
