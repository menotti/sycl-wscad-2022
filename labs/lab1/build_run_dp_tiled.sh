dpcpp -ffast-math -fsycl solutions/sobel_dp_tiled.cpp -o sobel_dp_tiled &&
for f in `ls *.pnm`; do ./solutions/sobel_dp_tiled $f tiled/${f%.*}.pgm; done
