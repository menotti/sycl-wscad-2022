dpcpp -ffast-math sobel.cpp -o sobel && 
for f in `ls *.pnm`; do ./sobel $f serial/${f%.*}.pgm; done
