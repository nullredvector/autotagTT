FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir open_clip_torch scikit-learn pillow numpy
WORKDIR /app
COPY autotagtt /app/autotagtt
ENV PYTHONUNBUFFERED=1 HF_HOME=/models ARCHIVE_DIR=/archive
ENTRYPOINT ["python", "-m", "autotagtt"]
CMD ["run"]
