#include <cstdio>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <math.h>
#include <CL/sycl.hpp>

#define itera 1000

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {
    //sycl::cpu_selector selector;
    sycl::gpu_selector selector;
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

    try {
        sycl::queue q(selector);
        size = dimx*dimy;
        char *rgb = sycl::malloc_shared<char>(size*3, q);
        char *gray = sycl::malloc_shared<char>(size, q);
        char *sobel = sycl::malloc_shared<char>(size, q);

        auto len = ifs.read(rgb, size*3).gcount();
        if (len != size*3) {
            cout << "Error reading file!" << endl;
            return EXIT_FAILURE;
        }
        ifs.close();

        std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

        auto start = high_resolution_clock::now();
        for (auto it = 0u; it < itera; it++) {
            sycl::range<1> rsize{static_cast<size_t>(size)};
            q.submit([&](sycl::handler &h) {
                h.parallel_for(rsize, [=](auto i) {
                    unsigned char r, g, b;
                    r = (char)rgb[i*3];
                    g = (char)rgb[i*3+1];
                    b = (char)rgb[i*3+2];
                    int gs = (r+(g<<1)+b)>>2;
                    gray[i] = (char)gs;
                });
            }).wait();

            auto global_range = sycl::range<2>(dimx, dimy);
            q.submit([&](sycl::handler &h) {
                h.parallel_for<class ndim>(global_range, [=](auto it) {
                    auto x = it[0]; 
                    auto y = it[1];
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

                    if (x > 0 && x < dimx && y > 0 && y < dimy) {
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
                });
            }).wait();
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << argv[1] << " with " << dimx << "x" << dimy << " in " << duration.count() << " ms" << endl;
        ofs.write(sobel, size);
        ofs.close();
    } catch (exception const &e) {
        cout << "An exception is caught!" << endl << e.what() << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
