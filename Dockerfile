FROM hbredin/pyannote:dev
ADD . /code
WORKDIR /code

RUN cd /code/SubComponent/3_Face/3_extract_flandmark && \
    cmake . && \
    make

ENV PYTHONPATH /code
CMD ["python"]