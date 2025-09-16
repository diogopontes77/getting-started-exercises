FROM python:3.10.7

WORKDIR /src

COPY . .
COPY ../data ../../src/data

RUN pip install -r requirements.txt

CMD ["python", "./node.py"]

# docker run -v $(pwd)/data:/data your_image_name

