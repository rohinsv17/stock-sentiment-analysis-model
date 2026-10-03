# Creates a base image with Python for Docker to run backend
FROM python:3.12

# Creates a non-root user to run the app and sets environment user variables
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Sets working directory for the app
WORKDIR /home/user/app

# Copies requirements.txt to directory and installs dependencies
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copies the app code to the working directory
COPY --chown=user . /home/user/app

# Exposes port for HuggingFace spaces
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]