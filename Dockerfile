FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
LABEL org.opencontainers.image.source="https://github.com/nullredvector/autotagTT"
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir open_clip_torch scikit-learn pillow numpy opencv-python-headless "transformers==4.46.3" sentencepiece protobuf
# Fail the build (not the nightly run) if the text tokenizer used for captions cannot load
RUN python -c "import open_clip; t = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP'); assert t(['a caption #tag']).shape[0] == 1"
WORKDIR /app
COPY autotagtt /app/autotagtt
ENV PYTHONUNBUFFERED=1 HF_HOME=/models ARCHIVE_DIR=/archive
ENTRYPOINT ["python", "-m", "autotagtt"]
CMD ["run"]
