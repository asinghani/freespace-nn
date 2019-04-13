#include <iostream>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

/**
 * Keybindings:
 * Next   = L     = 108
 * Prev   = H     = 104
 * Reset  = R     = 114
 * Create = Space = 32
 * 
 * Autosave on create
 */
#define NEXT_KEY 108
#define PREV_KEY 104
#define RESET_KEY 114
#define CREATE_KEY 32

vector<Point> points;

bool down = false;

void mouseCb(int event, int x, int y, int flags, void* args) {
    if (event == EVENT_LBUTTONDOWN) {
        down = true;
        if(points.size() == 0 && x != 0) {
            Point pt(0, y);
            points.push_back(pt);
        }
        Point pt(x, y);
        points.push_back(pt);
    } else if (event == EVENT_LBUTTONUP) {
        down = false;
    }

    return; // temp fix for mac
    if (event == EVENT_MOUSEMOVE && down) {
        if(points.size() == 0 && x != 0) {
            Point pt(0, y);
            points.push_back(pt);
        }
        Point pt(x, y);
        points.push_back(pt);
    }
}

string getFileName(const std::string& path) {
    return path.substr(path.find_last_of("/\\") + 1);
}

int main(int argc, char* argv[]) {
    if(argc < 3) {
		cout << "Usage: LabellingTool <output directory> <images (as glob)>" << endl;
		return -1;
    }

    string outputPath = argv[1];
    outputPath.append("/");

    vector<string> imageFiles;

    for(int i = 2; i < argc; i++) {
        imageFiles.push_back(argv[i]);
    }

    namedWindow("Labelling", WINDOW_NORMAL);
    setMouseCallback("Labelling", mouseCb);

    bool drawn = false;
    int n = 0;
    Mat image;
    image = imread(imageFiles[n], CV_LOAD_IMAGE_COLOR);
    cout << "Opening image " << n + 1 << " / " << imageFiles.size() << endl;
    while(true) {
        if(points.size() > 1) {
            if(drawn) {
                image = imread(imageFiles[n], CV_LOAD_IMAGE_COLOR);
                drawn = false;
            }
            for(int i = 0; i < points.size() - 1; i++) {
                line(image, points[i], points[i + 1], Scalar(0, 255, 0), 2);
            }
        }
        imshow("Labelling", image);

        int key = waitKey(30);

        if(key == NEXT_KEY) {
            n++;
            if(n >= imageFiles.size()) {
                break;
            }
            points.clear();
            image = imread(imageFiles[n], CV_LOAD_IMAGE_COLOR);
            cout << "Going forward to image " << n + 1 << " / " << imageFiles.size() << endl;
            drawn = false;
        }
        if(key == PREV_KEY) {
            n--;
            if(n < 0) {
                n = 0;
            }
            points.clear();
            image = imread(imageFiles[n], CV_LOAD_IMAGE_COLOR);
            cout << "Going back to image " << n + 1 << " / " << imageFiles.size() << endl;
            drawn = false;
        }
        if(key == RESET_KEY) {
            points.clear();
            image = imread(imageFiles[n], CV_LOAD_IMAGE_COLOR);
            cout << "Resetting label, image " << n + 1 << " / " << imageFiles.size() << endl;
            drawn = false;
        }
        if(key == CREATE_KEY) {
            if(points.size() > 1) {
                vector<Point> maskPts;
                maskPts.push_back(Point(0, image.rows - 1));
                maskPts.insert(maskPts.end(), points.begin(), points.end());
                maskPts.push_back(Point(image.cols - 1, points[points.size() - 1].y));
                maskPts.push_back(Point(image.cols - 1, image.rows - 1));

                Mat overlay;
                image.copyTo(overlay);

                const Point* drawPts[1] = { &maskPts[0] };
                int numPts = (int) maskPts.size();
                fillPoly(overlay, drawPts, &numPts, 1, Scalar(0, 255, 0));
                
                addWeighted(image, 0.6, overlay, 0.4, 0.0, image);
                drawn = true;

                for(int i = 0; i < maskPts.size() - 1; i++) {
                    line(image, maskPts[i], maskPts[i + 1], Scalar(0, 255, 0), 2);
                }
                line(image, maskPts[maskPts.size() - 1], maskPts[0], Scalar(0, 255, 0), 2);

                points.clear();

                // Save to file
                string outFile = outputPath + getFileName(imageFiles[n]);
                Mat img(image.rows, image.cols, CV_8UC3, Scalar(0, 0, 0));
                fillPoly(img, drawPts, &numPts, 1, Scalar(255, 255, 255));
                imwrite(outFile, img);
                cout << "Wrote label to " << outFile << endl;
            }
        }
    }
    return 0;
}




