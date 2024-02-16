# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: clean build

build: 
	$(CXX) convertRGBToGrey.cu --std=c++14 `pkg-config opencv4 --cflags --libs` -o convertRGBToGrey -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda -I/usr/local/cuda/samples/common/inc/

run:
	./convertRGBToGrey $(ARGS)

clean:
	rm -f convertRGBToGrey