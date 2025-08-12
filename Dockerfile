FROM python:3.12

# Services and app dependencies
RUN apt-get update
RUN apt-get install -y llvm libv4l-dev libhdf5-dev
RUN apt-get install -y libatlas-base-dev libqtgui4 libqt4-test

WORKDIR /app

RUN git clone https://github.com/jasperproject/jasper-client.git jasper && chmod +x jasper/jasper.py && \
    pip install --upgrade setuptools && \
    pip install -r jasper/client/requirements.txt

ADD requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD [ "python", "test.py" ]
