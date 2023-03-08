FROM python:3.7

# Set the current working directory to /app
WORKDIR /app

# Copy requirements.txt to working directory
COPY ./requirements.txt /code/requirements.txt

# Install requirements
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy model file(s)
COPY ./model /app/model

# Copy api.py file
COPY ./api.py /app/

# Expose port
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
