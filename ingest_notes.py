import re
from pathlib import Path
import click
import nbformat
from nbconvert import MarkdownExporter
from groq import Groq

# Initialisation du client Groq
# L'outil dirdotenv que tu as configuré va automatiquement charger GROQ_API_KEY depuis ton .env
client = Groq()


def clean_yaml_response(llm_output: str) -> str:
    """Nettoie la réponse pour extraire uniquement le YAML brut"""
    cleaned = re.sub(r"^```[a-zA-Z]*\n", "", llm_output, flags=re.MULTILINE)
    cleaned = re.sub(r"\n```$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip().strip("---").strip()
    return cleaned


def generate_yaml_metadata(content: str, model_name: str) -> str:
    """Appelle Groq pour catégoriser le contenu textuel du notebook"""
    prompt = f"""Analyse l'extrait suivant d'un notebook de cours et génère UNIQUEMENT un bloc YAML (front-matter) valide.
    Ne mets aucune explication, et pas de balises de code markdown (pas de ```yaml).
    
    Génère exactement ces champs basés sur le contexte :
    topic: "Nom du sujet principal"
    module: Numéro du module (ex: 1)
    difficulty: "Beginner" ou "Intermediate" ou "Advanced"
    tags: [liste de 2 à 4 tags techniques précis]

    Extrait du contenu :
    {content[:3000]}"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un agent de structuration de données. Tu réponds uniquement en YAML pur, sans fioritures.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return clean_yaml_response(response.choices[0].message.content)
    except Exception as e:
        raise click.ClickException(f"Échec de l'appel API Groq : {e}")


@click.command()
@click.argument("filepath", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--model",
    default="meta-llama/llama-4-scout-17b-16e-instruct",
    help="Modèle Groq à utiliser.",
)
def main(filepath: Path, model: str):
    """Convertit un notebook .ipynb en .md enrichi en front-matter YAML pour ton RAG via Groq."""
    if filepath.suffix != ".ipynb":
        raise click.BadParameter(
            "Le fichier fourni doit être un notebook avec l'extension .ipynb"
        )

    click.echo(f"📦 Extraction du notebook : {filepath.name}")

    # 1. Extraction et conversion via nbconvert
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        exporter = MarkdownExporter()
        body, _ = exporter.from_notebook_node(nb)
    except Exception as e:
        raise click.ClickException(f"Erreur lors de la conversion du notebook : {e}")

    # 2. Génération du YAML via Groq
    click.echo(f"🤖 Génération des métadonnées via Groq ({model})...")
    yaml_header = generate_yaml_metadata(body, model)

    # 3. Assemblage et écriture du fichier final
    output_path = filepath.with_suffix(".md")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"---\n{yaml_header}\n---\n\n{body}")

        click.secho(
            f"🚀 Succès ! Fichier sauvegardé ici : {output_path}", fg="green", bold=True
        )
    except Exception as e:
        raise click.ClickException(f"Impossible d'écrire le fichier Markdown : {e}")


if __name__ == "__main__":
    main()
