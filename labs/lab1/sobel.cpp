#include <cstdio>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <math.h>

#define itera 1000

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {
    int dimx, dimy, depth, size;
    string line; 

    if (argc < 3) {
        cout << "Usage: " << argv[0] << " infile.pnm outfile.pgm" << endl;
        return EXIT_FAILURE; 
    }

    ifstream ifs(argv[1], ios_base::in | ios_base::binary);

    ifs >> line;
    if (line != "P6") {
        cout << argv[1] << " is not a valid P6 file!" << endl;
        return EXIT_FAILURE;
    }
    ifs >> dimx >> dimy >> depth;
    ifs.ignore();

    ofstream ofs(argv[2], ios_base::out | ios_base::binary);
    ofs << "P5" << endl << dimx << ' ' << dimy << endl << "255" << endl;

    size = dimx*dimy;
    char *rgb = new char[size*3];
    char *gray = new char[size];
    char *sobel = new char[size];

    unsigned char r, g, b;

    auto len = ifs.read(rgb, size*3+7).gcount();
    if (len != size*3) {
        cout << "Error reading file!" << endl;
        return EXIT_FAILURE;
    }
    ifs.close();

    auto start = high_resolution_clock::now();
    for (auto it = 0u; it < itera; it++) {
        for (auto ii = 0u, oi = 0u; ii < size*3; ii=ii+3, oi++) {
            r = (char)rgb[ii];
            g = (char)rgb[ii+1];
            b = (char)rgb[ii+2];
            int gs = (r+(g<<1)+b)>>2;
            gray[oi] = (char)gs;
        }

        int pixel_x;
        int pixel_y;

        float sobel_x[3][3] =
        {{ -1, 0, 1 },
         { -2, 0, 2 },
         { -1, 0, 1 }};

        float sobel_y[3][3] =
        {{ -1, -2, -1 },
         {  0,  0,  0 },
         {  1,  2,  1 }};

        for (int x=1; x < dimx-1; x++) {
            for (int y=1; y < dimy-1; y++) {
                pixel_x = (sobel_x[0][0] * gray[dimx * (y-1) + (x-1)])
                        + (sobel_x[0][1] * gray[dimx * (y-1) +  x   ])
                        + (sobel_x[0][2] * gray[dimx * (y-1) + (x+1)])
                        + (sobel_x[1][0] * gray[dimx *  y    + (x-1)])
                        + (sobel_x[1][1] * gray[dimx *  y    +  x   ])
                        + (sobel_x[1][2] * gray[dimx *  y    + (x+1)])
                        + (sobel_x[2][0] * gray[dimx * (y+1) + (x-1)])
                        + (sobel_x[2][1] * gray[dimx * (y+1) +  x   ])
                        + (sobel_x[2][2] * gray[dimx * (y+1) + (x+1)]);
                pixel_y = (sobel_y[0][0] * gray[dimx * (y-1) + (x-1)])
                        + (sobel_y[0][1] * gray[dimx * (y-1) +  x   ])
                        + (sobel_y[0][2] * gray[dimx * (y-1) + (x+1)])
                        + (sobel_y[1][0] * gray[dimx *  y    + (x-1)])
                        + (sobel_y[1][1] * gray[dimx *  y    +  x   ])
                        + (sobel_y[1][2] * gray[dimx *  y    + (x+1)])
                        + (sobel_y[2][0] * gray[dimx * (y+1) + (x-1)])
                        + (sobel_y[2][1] * gray[dimx * (y+1) +  x   ])
                        + (sobel_y[2][2] * gray[dimx * (y+1) + (x+1)]);
                int val = (int)sqrt((pixel_x * pixel_x) + (pixel_y * pixel_y));
                if(val < 0) val = 0;
                if(val > 255) val = 255;
                sobel[dimx * y + x] = (unsigned char)val;
            }
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << argv[1] << " with " << dimx << "x" << dimy << " in " << duration.count() << " ms" << endl;
    ofs.write(sobel, size);
    ofs.close();
    return EXIT_SUCCESS;
}
