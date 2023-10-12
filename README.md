# ANPR_Model
<h1>Automatic Number Plate Reconition Model</h1>
The ANPR works in four steps:

<b>Real-time object detection:<b> Uses deep learning techniques to recognize the vehicle through the webcam monitoring. Object detection algorithms-Harcascade detection model is used.

<b>Image processing:<b> The image obtained is processed through EasyOCR algorithm, which includes normalizing the image by sharpening it to reduce noise, performing edge detection, converting it to grayscale to improve the results of the subsequent algorithm.

<b>Optical Character Recognition:<b> Using Tesseract OCR model, the system reads the license plate in the processed image. This involves detecting individual characters, verifying the sequence and then converting the number plate image to text/string format.
<p>
The recognised number plate is stored in a .csv file with initial time-stamp, marking the entry of the vehicle in a smart-parking based system. This model is used in a Smart-Parking System to automate the parking process and reduce customer wait time.
</p>
