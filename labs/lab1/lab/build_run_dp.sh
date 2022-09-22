#export OverrideDefaultFP64Settings=1
#export IGC_EnableDPEmulation=1
dpcpp -ffast-math -fsycl sobel_dp.cpp -o sobel_dp &&
for f in `ls *.pnm`; do ./sobel_dp $f dp/${f%.*}.pgm; done
