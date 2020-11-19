FROM opencvcourses/course-1:4.1.0

MAINTAINER Labhesh Valechha <labheshvalechha@gmail.com>

ARG lept_version=1.80.0
ARG tess_version=4.1.1
ARG dlib=19.21
ARG dlib_version=v$dlib

WORKDIR /home

RUN apt-get update && apt-get install -y --no-install-recommends libwebp-dev \
	&& rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/DanBloomberg/leptonica.git \
	&& cd leptonica/ \
	&& git checkout $lept_version \
	&& mkdir build && cd build \
	&& cmake \
	-DBUILD_SHARED_LIBS=1 \
	.. \
	&& make \
	&& make install \
	&& cd /home \
	&& rm -rf leptonica

RUN git clone https://github.com/tesseract-ocr/tesseract.git \
	&& cd tesseract \
	&& git checkout $tess_version \
	&& mkdir build && cd build \
	&& cmake \
	-DLeptonica_DIR=/usr/local/cmake/ \
	.. \
	&& make \
	&& make install \
	&& cd /home \
	&& rm -rf tesseract

WORKDIR /usr/local/tessdata

RUN wget --no-verbose -O eng.traineddata https://github.com/tesseract-ocr/tessdata/blob/master/eng.traineddata?raw=true
	
ENV TESSDATA_PREFIX=/usr/local/tessdata

WORKDIR /home

RUN apt-get update && apt-get install -y --no-install-recommends unzip \
	&& conda install -y pytesseract -c conda-forge \
	&& conda install -y pytorch torchvision cpuonly -c pytorch \
	&& wget --no-verbose -O libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcpu.zip \
	&& unzip -q libtorch.zip -d /usr/local/ \
	&& rm -f libtorch.zip \
	&& apt-get remove unzip -y \
	&& conda clean --all --yes \
	&& rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/davisking/dlib.git \
	&& cd dlib \
	&& git checkout $dlib_version \
	&& cd dlib && mkdir build && cd build \
	&& cmake \
	-DBUILD_SHARED_LIBS=1 \
	.. \
	&& make \
	&& make install \
	&& cd ../.. \
	&& python setup.py install \
	&& cd .. \
	&& rm -rf dlib

RUN rm -rf sampleCode

COPY script.sh script.sh
COPY sampleCode /home/sampleCode

RUN chmod u+x script.sh \
	&& ./script.sh $tess_version \
	&& rm -f /home/script.sh

CMD ["bash"]
