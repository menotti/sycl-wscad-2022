dpcpp -ffast-math -fsycl sobel_dp_usm.cpp -o sobel_dp_usm &&
for f in `ls *.pnm`; do ./sobel_dp_usm $f usm/${f%.*}.pgm; done
