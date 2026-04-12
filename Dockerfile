# ── 1. Image de base ──────────────────────────────────────────────────
FROM python:3.13-slim

# ── 2. Répertoire de travail dans le conteneur ────────────────────────
WORKDIR /app

# ── 3. Copier et installer les dépendances ────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 4. Copier tout le projet ──────────────────────────────────────────
COPY main.py .
COPY supervised_model/ ./supervised_model/
COPY unsupervised_model/ ./unsupervised_model/

# ── 5. Port exposé ────────────────────────────────────────────────────
EXPOSE 8000

# ── 6. Commande de démarrage ──────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
