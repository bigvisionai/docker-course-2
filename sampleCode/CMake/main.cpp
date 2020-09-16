#include <iostream>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>
#include "../Extras/renderFace.hpp"

using namespace std;
using namespace cv;
using namespace tesseract;
using namespace dlib;


// Helper function for dlib
void writeLandmarksToFile(full_object_detection &landmarks, const string &filename)
{
    std::ofstream ofs;
    ofs.open(filename);

    for (int i = 0; i < landmarks.num_parts(); i++)
    ofs << landmarks.part(i).x() << " " << landmarks.part(i).y() << endl;
    ofs.close();
}


int main(){
    
    // Testing the OpenCV installation
    cout << "Hello, World!" << endl;
    cout << "OpenCV version : " << CV_VERSION << endl;
    cout << "Major version : " << CV_MAJOR_VERSION << endl;
    cout << "Minor version : " << CV_MINOR_VERSION << endl;
    cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;

    Mat testImage = imread("../Extras/Quick_brown_fox.png");
    
    
    // Testing the Tesseract-ocr installation
    string outText;
    TessBaseAPI *ocr = new TessBaseAPI();

    ocr->Init(NULL, "eng", OEM_LSTM_ONLY);
    ocr->SetPageSegMode(PSM_AUTO);
    ocr->SetImage(testImage.data, testImage.cols, testImage.rows, 3, testImage.step);

    outText = string(ocr->GetUTF8Text());
    cout << outText << endl;
    
    
    // Testing the Libtorch installation
    at::Tensor a = at::ones({2, 2}, at::kInt);
    at::Tensor b = at::randn({2, 2});
    auto c = a + b.to(at::kInt);
    cout << "a: " << a << endl;
    cout << endl;
    cout << "b: " << b << endl;
    cout << endl;
    cout << "c: " << c << endl;
    
    
    // Testing the dlib installation
    frontal_face_detector faceDetector = get_frontal_face_detector();
    shape_predictor landmarkDetector;

    deserialize("../Extras/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

    string imageFilename("../Extras/family.jpg");
    Mat img = imread(imageFilename);

    string landmarksBasename("results/family");
    cv_image<bgr_pixel> dlibIm(img);

    std::vector<dlib::rectangle> faceRects = faceDetector(dlibIm);
    cout << "Number of faces detected: " << faceRects.size() << endl;
    std::vector<full_object_detection> landmarksAll;

    for (int i = 0; i < faceRects.size(); i++) {
        full_object_detection landmarks = landmarkDetector(dlibIm, faceRects[i]);

        if (i == 0) cout << "Number of landmarks : " << landmarks.num_parts() << endl;

        landmarksAll.push_back(landmarks);
        renderFace(img, landmarks);

        stringstream landmarksFilename;
        landmarksFilename << landmarksBasename <<  "_"  << i << ".txt";
        cout << "Saving landmarks to " << landmarksFilename.str() << endl;
        writeLandmarksToFile(landmarks, landmarksFilename.str());
    }

    string outputFilename("results/familyLandmarks.jpg");
    cout << "Saving output image to " << outputFilename << endl;
    imwrite(outputFilename, img);
    
    return 0; 
}
