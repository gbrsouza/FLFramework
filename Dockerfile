FROM python:3.6-slim-buster

EXPOSE 5454
RUN apt-get update 
RUN mkdir flframework


COPY ./ /flframework


RUN pip3 install --no-cache-dir tensorflow scikit-learn tqdm
