# Gunakan Python 3.9 sebagai base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Salin file requirements.txt terlebih dahulu
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file lainnya
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Expose port
EXPOSE 5000

# Jalankan aplikasi
CMD ["python", "app.py"]

