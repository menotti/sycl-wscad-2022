/bin/echo "##" $(whoami) is working oneAPI_Toolkits Hough_transform lab WSCAD2022, dir $(pwd) 
#source /opt/intel/inteloneapi/setvars.sh
dpcpp -fintelfpga -DFPGA_EMULATOR hough_transform.cpp -o fpga.emu
./fpga.emu
dpcpp -fintelfpga -c hough_transform.cpp -o fpga.o
dpcpp -fintelfpga -fsycl-link -Xshardware fpga.o
#source ./clean.sh
