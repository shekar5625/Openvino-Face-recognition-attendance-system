@echo off
python app\face_recognition_demo.py ^
  -i 0 ^
  -m_fd models\models\face-detection-retail-0004\FP32\face-detection-retail-0004.xml ^
  -m_lm models\models\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009.xml ^
  -m_reid models\models\face-reidentification-retail-0095\FP32\face-reidentification-retail-0095.xml ^
  -m_as models\models\anti_spoof\2.7_80x80_MiniFASNetV2.xml ^
  --as_scale 2.7 ^
  -fg my_gallery\my_gallery ^
  --run_detector
