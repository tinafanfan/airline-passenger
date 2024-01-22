FROM pytorch/pytorch:latest

# mkdir
WORKDIR /app

# 複製本機的 train.py 到容器的 /app 目錄下
COPY . /app/

# pip install matplot
RUN pip install --no-cache-dir --upgrade pip && \
 pip install pandas numpy scikit-learn matplotlib torch

# ENTRYPOINT /bin/bash
CMD ["python", "train.py"]