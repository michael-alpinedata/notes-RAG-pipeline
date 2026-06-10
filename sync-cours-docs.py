import time

import hashlib
import json
import logging
import re
from pathlib import Path
import click
import yaml
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuration de l'observabilité (Logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialisation du client Groq
client = Groq()


# --- MODULE 1: STATE MANAGEMENT (Tracking par Hash) ---
def get_file_hash(filepath: Path) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier pour détecter un changement de contenu exact."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_state(dest_dir: Path) -> dict:
    state_file = dest_dir / ".sync_state.json"
    if state_file.exists():
        with open(state_file, "r") as f:
            return json.load(f)
    return {}


def save_state(dest_dir: Path, state: dict):
    state_file = dest_dir / ".sync_state.json"
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


# --- MODULE 2: LLM & DATA QUALITY ---
def extract_yaml_from_text(llm_output: str) -> str:
    """Nettoie la réponse brute du LLM."""
    cleaned = re.sub(r"^```[a-zA-Z]*\n", "", llm_output, flags=re.MULTILINE)
    cleaned = re.sub(r"\n```$", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip().strip("---").strip()


# Le décorateur Retry : S'il y a une erreur API, il va réessayer 3 fois
# en attendant 2s, puis 4s, puis 8s avant d'abandonner.
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def generate_yaml_metadata(content: str, model_name: str, relative_path: str) -> str:
    """Appelle Groq et valide le YAML."""
    prompt = f"""Analyse l'extrait suivant d'un cours technique (Fichier: {relative_path}) 
    et génère UNIQUEMENT un bloc YAML (front-matter) valide.
    
    Champs requis :
    topic: "Nom du sujet principal"
    module: Numéro du module (ex: 1, ou 0 si inconnu)
    source: "official_course"
    difficulty: "Beginner" ou "Intermediate" ou "Advanced"
    tags: [liste de 2 à 4 tags techniques]

    Extrait :
    {content[:3000]}"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "Tu es un agent d'ingestion de données. Tu réponds uniquement en YAML valide.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    yaml_str = extract_yaml_from_text(response.choices[0].message.content)

    # VALIDATION (Data Quality Check)
    try:
        parsed_yaml = yaml.safe_load(yaml_str)
        if not isinstance(parsed_yaml, dict) or "topic" not in parsed_yaml:
            raise ValueError(
                "Le YAML généré ne contient pas la structure minimale requise."
            )
        return yaml_str
    except yaml.YAMLError as e:
        logger.error(f"LLM a généré un YAML invalide: {e}")
        raise ValueError("Invalid YAML")  # Déclenchera le Retry de tenacity


# --- MODULE 3: PIPELINE PRINCIPALE ---
@click.command()
@click.option(
    "--source",
    "-s",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Dossier Git source",
)
@click.option(
    "--dest",
    "-d",
    type=click.Path(path_type=Path),
    required=True,
    help="Dossier RAG destination",
)
@click.option(
    "--model", default="meta-llama/llama-4-scout-17b-16e-instruct", help="Modèle Groq"
)
@click.option("--force", is_flag=True, help="Force la synchro de tous les fichiers")
def main(source: Path, dest: Path, model: str, force: bool):
    """ETL Incrémental robuste pour la documentation RAG."""
    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"Démarrage de la pipeline d'ingestion : {source.name} -> {dest.name}")

    # Chargement de l'état précédent (Le State Backend)
    sync_state = load_state(dest) if not force else {}
    source_files = list(source.rglob("*.md"))

    metrics = {"processed": 0, "skipped": 0, "errors": 0}

    for src_file in source_files:
        rel_path = src_file.relative_to(source)
        dest_file = dest / rel_path

        # 1. Calcul du hash actuel
        current_hash = get_file_hash(src_file)
        path_key = str(rel_path)

        # 2. Logique d'idempotence (Comparaison des hash)
        if (
            path_key in sync_state
            and sync_state[path_key] == current_hash
            and dest_file.exists()
        ):
            logger.debug(f"Skipped (unchanged): {rel_path}")
            metrics["skipped"] += 1
            continue

        logger.info(f"Traitement de : {rel_path}")

        try:
            # Extraction
            with open(src_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Transformation (LLM Enrichment avec Retry)
            yaml_header = generate_yaml_metadata(content, model, str(rel_path))

            # Chargement (Load)
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_file, "w", encoding="utf-8") as f:
                f.write(f"---\n{yaml_header}\n---\n\n{content}")

            # Mise à jour du State
            sync_state[path_key] = current_hash
            metrics["processed"] += 1

            # RATE LIMITING PROACTIF : Pause de 6 secondes entre chaque appel API
            # 6 secondes = ~10 requêtes par minute (sécurise le plafond de 30K TPM)
            logger.info("Temporisation 6s (pour ne pas reach Rate Limit)...")
            time.sleep(6)

        except Exception as e:
            logger.error(f"Échec sur {rel_path} après plusieurs tentatives : {e}")
            metrics["errors"] += 1

    # Sauvegarde de l'état de la pipeline
    save_state(dest, sync_state)
    logger.info(
        f"Pipeline terminée. Succès: {metrics['processed']} | Ignorés: {metrics['skipped']} | Erreurs: {metrics['errors']}"
    )


if __name__ == "__main__":
    main()
