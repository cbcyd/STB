FROM python:3.8

WORKDIR /STB

ADD . /STB

RUN pip install -r requirements.txt

CMD ["python3",  "./bot.py"]