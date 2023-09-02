FROM python:3.11.5

ADD clipindex.py .

RUN pip install -r requirements.txt

CMD ["python", "clipindex.py"]
