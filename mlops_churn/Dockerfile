FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy pandas scikit-learn scipy joblib
COPY churn/ churn/
COPY registry/ registry/
EXPOSE 8080
CMD ["python", "-c", "from churn.serve import serve; import http.server; s,_=serve('registry', 8080); s.server_address=('0.0.0.0',8080); s.serve_forever()"]
