FROM debian

#Install packages
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install python3.7
COPY requirements.txt requirements.txt
RUN apt install python3-pip -y
RUN pip3 install -r requirements.txt 
CMD python3 app.py
EXPOSE 8088
COPY . . 

