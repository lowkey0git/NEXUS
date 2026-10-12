FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
ENV ASSISTANT_DATA_DIR=/data HOST=0.0.0.0 PORT=8000
VOLUME /data
EXPOSE 8000
CMD ["python", "-m", "app"]
