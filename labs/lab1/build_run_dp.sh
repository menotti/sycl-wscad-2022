dpcpp -ffast-math -fsycl sobel_dp.cpp -o sobel_dp &&
for f in `ls *.pnm`; do ./sobel_dp $f dp/${f%.*}.pgm; done
