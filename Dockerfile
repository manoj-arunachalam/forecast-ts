FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN python -m pip install prophet
EXPOSE 5001
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]