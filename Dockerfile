# momentum-after-the-paper: reproduce every number and chart in a clean container.
#   docker build -t momentum-after-the-paper . && docker run --rm -v "$PWD/results:/app/results" momentum-after-the-paper
# The Ken French data is downloaded when the container runs (needs network); it is not in the image.
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends make && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY Makefile momentum.py walk_forward.py data.py costs.py charts.py data.sha256 ./
COPY fonts/ fonts/
ENV MPLCONFIGDIR=/tmp/mpl IN_DOCKER=1
CMD ["make", "reproduce"]
