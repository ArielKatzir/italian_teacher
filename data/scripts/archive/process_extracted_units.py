#!/usr/bin/env python3
"""
General-purpose script to process manually extracted textbook exercises.
Takes a JSON file with extracted exercises and generates training data.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Unit metadata
UNIT_METADATA = {
    1: {"title": "Nouns: gender and number", "grammar_focus": "nouns", "cefr": "A1"},
    2: {"title": "Definite and indefinite articles", "grammar_focus": "articles", "cefr": "A1"},
    3: {
        "title": "Adjectives; possessive and demonstrative pronouns",
        "grammar_focus": "adjectives",
        "cefr": "A1",
    },
    4: {
        "title": "The present tense of essere and avere",
        "grammar_focus": "present_tense",
        "cefr": "A1",
    },
    5: {
        "title": "The present tense of regular (and some irregular) verbs",
        "grammar_focus": "present_tense",
        "cefr": "A1",
    },
    6: {"title": "Adverbs", "grammar_focus": "adverbs", "cefr": "A2"},
    7: {"title": "Direct object pronouns (1)", "grammar_focus": "pronouns", "cefr": "A2"},
    8: {"title": "Prepositions", "grammar_focus": "prepositions", "cefr": "A2"},
    9: {"title": "Questions", "grammar_focus": "questions", "cefr": "A2"},
    10: {"title": "Indirect object pronouns (1)", "grammar_focus": "pronouns", "cefr": "A2"},
    11: {"title": "Piacere and similar verbs", "grammar_focus": "verbs", "cefr": "A2"},
    12: {"title": "The present perfect tense", "grammar_focus": "past_tense", "cefr": "A2"},
    13: {
        "title": "Direct and indirect object pronouns (2)",
        "grammar_focus": "pronouns",
        "cefr": "A2",
    },
    14: {
        "title": "Direct and indirect object pronouns (3: stressed forms)",
        "grammar_focus": "pronouns",
        "cefr": "A2",
    },
    15: {"title": "Relative pronouns", "grammar_focus": "pronouns", "cefr": "B1"},
    16: {"title": "The imperfect tense", "grammar_focus": "imperfect_tense", "cefr": "B1"},
    17: {"title": "The pronouns ne and ci", "grammar_focus": "pronouns", "cefr": "B1"},
    18: {"title": "The future tense", "grammar_focus": "future_tense", "cefr": "B1"},
    19: {"title": "The past perfect tense", "grammar_focus": "past_tense", "cefr": "B1"},
    20: {"title": "The conditional", "grammar_focus": "conditional", "cefr": "B1"},
    21: {"title": "The imperative", "grammar_focus": "imperative", "cefr": "A2"},
    22: {"title": "The present conditional", "grammar_focus": "conditional", "cefr": "B1"},
    23: {"title": "The subjunctive", "grammar_focus": "subjunctive", "cefr": "B1"},
}


# Topic inference keywords - EXPANDED WITH MORE WORDS
TOPIC_KEYWORDS = {
    "daily_life": [
        "giorno",
        "sera",
        "mattina",
        "notte",
        "pomeriggio",
        "caffè",
        "colazione",
        "pranzo",
        "cena",
        "dormire",
        "alzarsi",
        "svegliarsi",
        "lavarsi",
        "routine",
        "quotidiano",
        "giornata",
        "mezzogiorno",
    ],
    "people": [
        "uomo",
        "donna",
        "ragazzo",
        "ragazza",
        "bambino",
        "bambini",
        "persona",
        "persone",
        "gente",
        "infermiera",
        "infermiere",
        "dottore",
        "dottoressa",
        "cliente",
        "clienti",
        "amico",
        "amica",
        "amici",
        "signore",
        "signora",
        "signorina",
        "uomini",
        "donne",
        "ragazzi",
        "ragazze",
    ],
    "places": [
        "strada",
        "vie",
        "piazza",
        "piazze",
        "stazione",
        "aeroporto",
        "porto",
        "ristorante",
        "bar",
        "caffè",
        "osteria",
        "trattoria",
        "banca",
        "posta",
        "ufficio postale",
        "fiume",
        "lago",
        "mare",
        "spiaggia",
        "montagna",
        "città",
        "paese",
        "villaggio",
        "centro",
        "casa",
        "appartamento",
        "abitazione",
        "scuola",
        "università",
        "liceo",
        "ufficio",
        "fabbrica",
        "negozio",
        "biblioteca",
        "museo",
        "teatro",
        "cinema",
        "ospedale",
        "farmacia",
        "clinica",
        "chiesa",
        "cattedrale",
        "basilica",
        "supermercato",
        "mercato",
        "giardino",
        "parco",
        "villa",
    ],
    "body": [
        "occhio",
        "occhi",
        "orecchio",
        "orecchie",
        "mano",
        "mani",
        "dito",
        "dita",
        "ginocchio",
        "ginocchia",
        "capelli",
        "testa",
        "faccia",
        "viso",
        "braccio",
        "braccia",
        "gamba",
        "gambe",
        "piede",
        "piedi",
        "corpo",
        "naso",
        "bocca",
        "denti",
        "lingua",
    ],
    "food_drink": [
        "caffè",
        "tè",
        "latte",
        "acqua",
        "vino",
        "birra",
        "zucchero",
        "sale",
        "pepe",
        "pane",
        "pasta",
        "pizza",
        "riso",
        "carne",
        "pesce",
        "pollo",
        "arancia",
        "arance",
        "mela",
        "mele",
        "ciliegia",
        "ciliegie",
        "uva",
        "pomodoro",
        "pomodori",
        "insalata",
        "formaggio",
        "burro",
        "uovo",
        "uova",
        "yogurt",
        "gelato",
        "dolce",
        "mangiare",
        "bere",
        "cibo",
        "bevanda",
        "colazione",
        "pranzo",
        "cena",
        "spuntino",
        "ristorante",
        "cucina",
    ],
    "objects": [
        "televisione",
        "tv",
        "radio",
        "calendario",
        "orologio",
        "sveglia",
        "camicia",
        "vestito",
        "abito",
        "libro",
        "libri",
        "giornale",
        "rivista",
        "tavolo",
        "sedia",
        "poltrona",
        "divano",
        "computer",
        "telefono",
        "cellulare",
        "porta",
        "finestra",
        "finestre",
        "macchina",
        "auto",
        "automobile",
        "bicicletta",
        "chiave",
        "chiavi",
        "penna",
        "matita",
        "quaderno",
        "borsa",
        "valigia",
        "zaino",
        "specchio",
        "foto",
        "fotografia",
    ],
    "family": [
        "famiglia",
        "famiglie",
        "padre",
        "papà",
        "madre",
        "mamma",
        "figlio",
        "figlia",
        "figli",
        "figlie",
        "fratello",
        "sorella",
        "fratelli",
        "sorelle",
        "zio",
        "zia",
        "zii",
        "zie",
        "nonno",
        "nonna",
        "nonni",
        "nonne",
        "cugino",
        "cugina",
        "cugini",
        "nipote",
        "nipoti",
        "marito",
        "moglie",
        "sposo",
        "sposa",
        "genitori",
        "parenti",
    ],
    "work_education": [
        "lavoro",
        "lavorare",
        "lavoratore",
        "professore",
        "professoressa",
        "insegnante",
        "studente",
        "studenti",
        "studentessa",
        "scuola",
        "università",
        "liceo",
        "istituto",
        "esame",
        "esami",
        "compito",
        "compiti",
        "lezione",
        "lezioni",
        "corso",
        "biblioteca",
        "aula",
        "classe",
        "ufficio",
        "uffici",
        "scrivania",
        "collega",
        "colleghi",
        "capo",
        "direttore",
        "impiegato",
        "impiegata",
        "medico",
        "avvocato",
        "ingegnere",
    ],
    "emotions_states": [
        "felice",
        "contento",
        "allegro",
        "gioioso",
        "triste",
        "infelice",
        "depresso",
        "stanco",
        "stanca",
        "stanchi",
        "arrabbiato",
        "nervoso",
        "furioso",
        "simpatico",
        "antipatico",
        "gentile",
        "cortese",
        "educato",
        "preoccupato",
        "ansioso",
        "sorpreso",
        "stupito",
        "paura",
        "coraggio",
        "timido",
    ],
    "time": [
        "ora",
        "ore",
        "tempo",
        "volta",
        "anno",
        "anni",
        "mese",
        "mesi",
        "settimana",
        "settimane",
        "giorno",
        "giorni",
        "giornata",
        "ieri",
        "oggi",
        "domani",
        "lunedì",
        "martedì",
        "mercoledì",
        "giovedì",
        "venerdì",
        "sabato",
        "domenica",
        "mattina",
        "pomeriggio",
        "sera",
        "notte",
        "gennaio",
        "febbraio",
        "marzo",
        "aprile",
        "maggio",
        "giugno",
        "luglio",
        "agosto",
        "settembre",
        "ottobre",
        "novembre",
        "dicembre",
        "passato",
        "presente",
        "futuro",
        "presto",
        "tardi",
        "sempre",
        "mai",
        "spesso",
        "raramente",
    ],
    "travel": [
        "viaggio",
        "viaggiare",
        "viaggiatore",
        "treno",
        "treni",
        "stazione",
        "binario",
        "aereo",
        "aeroporto",
        "volo",
        "nave",
        "porto",
        "traghetto",
        "macchina",
        "automobile",
        "guidare",
        "autobus",
        "pullman",
        "tram",
        "metro",
        "bicicletta",
        "moto",
        "partire",
        "partenza",
        "arrivare",
        "arrivo",
        "biglietto",
        "biglietti",
        "prenotazione",
        "valigia",
        "bagaglio",
        "zaino",
        "hotel",
        "albergo",
        "ostello",
        "vacanza",
        "vacanze",
        "turismo",
        "turista",
    ],
    "weather": [
        "tempo",
        "meteo",
        "clima",
        "sole",
        "soleggiato",
        "sereno",
        "pioggia",
        "piovere",
        "piovoso",
        "neve",
        "nevicare",
        "nevoso",
        "vento",
        "ventoso",
        "ventilato",
        "nuvola",
        "nuvoloso",
        "coperto",
        "caldo",
        "freddo",
        "fresco",
        "temperatura",
        "grado",
        "gradi",
        "umido",
        "secco",
        "tempesta",
        "temporale",
        "tuono",
        "fulmine",
        "nebbia",
        "nebbioso",
    ],
    "clothing": [
        "vestito",
        "vestiti",
        "abito",
        "abiti",
        "camicia",
        "camicie",
        "maglia",
        "maglie",
        "gonna",
        "gonne",
        "pantalone",
        "pantaloni",
        "giacca",
        "cappotto",
        "giubbotto",
        "scarpe",
        "scarpa",
        "stivali",
        "sandali",
        "cappello",
        "cappelli",
        "berretto",
        "calzini",
        "calze",
        "collant",
        "cravatta",
        "cintura",
        "maglione",
        "felpa",
        "t-shirt",
        "jeans",
        "shorts",
    ],
    "colors": [
        "colore",
        "colori",
        "rosso",
        "rossa",
        "rossi",
        "rosse",
        "blu",
        "azzurro",
        "celeste",
        "bianco",
        "bianca",
        "bianchi",
        "bianche",
        "nero",
        "nera",
        "neri",
        "nere",
        "verde",
        "verdi",
        "giallo",
        "gialla",
        "gialli",
        "gialle",
        "arancione",
        "arancioni",
        "rosa",
        "marrone",
        "viola",
        "grigio",
        "grigi",
        "grigia",
        "grigie",
    ],
    "hobbies_sports": [
        "calcio",
        "tennis",
        "nuoto",
        "pallavolo",
        "basket",
        "correre",
        "camminare",
        "sciare",
        "giocare",
        "gioco",
        "sport",
        "musica",
        "suonare",
        "cantare",
        "canzone",
        "leggere",
        "lettura",
        "libro",
        "cinema",
        "film",
        "teatro",
        "spettacolo",
        "pittura",
        "dipingere",
        "disegnare",
        "fotografia",
        "fotografare",
        "foto",
    ],
    "technology": [
        "computer",
        "pc",
        "laptop",
        "telefono",
        "cellulare",
        "smartphone",
        "internet",
        "email",
        "sito",
        "web",
        "tv",
        "televisione",
        "radio",
        "stampante",
        "fotocopiatrice",
        "app",
        "applicazione",
        "programma",
        "tecnologia",
        "digitale",
    ],
    "nature": [
        "natura",
        "naturale",
        "ambiente",
        "albero",
        "alberi",
        "pianta",
        "piante",
        "fiore",
        "fiori",
        "erba",
        "foglia",
        "foglie",
        "fiume",
        "lago",
        "mare",
        "oceano",
        "montagna",
        "montagne",
        "collina",
        "bosco",
        "foresta",
        "giardino",
        "parco",
        "animale",
        "animali",
        "cane",
        "gatto",
        "uccello",
        "uccelli",
        "pesce",
        "pesci",
    ],
    "house_home": [
        "casa",
        "abitazione",
        "appartamento",
        "stanza",
        "stanze",
        "camera",
        "camere",
        "cucina",
        "bagno",
        "salotto",
        "soggiorno",
        "camera da letto",
        "letto",
        "porta",
        "finestra",
        "muro",
        "parete",
        "pavimento",
        "soffitto",
        "tetto",
        "giardino",
        "terrazzo",
        "balcone",
        "garage",
        "cantina",
        "soffitta",
    ],
    "shopping": [
        "negozio",
        "negozi",
        "shop",
        "supermercato",
        "mercato",
        "comprare",
        "acquistare",
        "vendere",
        "prezzo",
        "costo",
        "costare",
        "euro",
        "soldi",
        "denaro",
        "contanti",
        "carta",
        "pagare",
        "pagamento",
        "scontrino",
        "ricevuta",
        "fattura",
        "offerta",
        "sconto",
        "saldi",
    ],
    "health": [
        "salute",
        "sano",
        "malato",
        "malattia",
        "medico",
        "dottore",
        "infermiere",
        "ospedale",
        "clinica",
        "ambulanza",
        "farmacia",
        "medicina",
        "medicinale",
        "dolore",
        "male",
        "febbre",
        "tosse",
        "raffreddore",
        "influenza",
        "visita",
        "cura",
        "guarire",
    ],
}


def infer_topic(questions, answers):
    """Infer contextual topic from vocabulary."""
    all_text = " ".join(questions + answers).lower()

    scores = defaultdict(int)
    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in all_text:
                scores[topic] += 1

    if scores:
        best = max(scores.items(), key=lambda x: x[1])
        if best[1] >= 2:  # At least 2 keyword matches
            return best[0]

    return None


def chunk_exercises(questions, answers, chunk_size=8):
    """Split exercises into chunks of max size."""
    chunks = []
    for i in range(0, len(questions), chunk_size):
        chunk_q = questions[i : i + chunk_size]
        chunk_a = answers[i : i + chunk_size] if answers else []
        chunks.append((chunk_q, chunk_a))
    return chunks


def create_training_examples(units_data):
    """Convert extracted units data to training format."""
    all_examples = []

    for unit_num, unit_data in units_data.items():
        unit_meta = UNIT_METADATA.get(
            unit_num, {"title": f"Unit {unit_num}", "grammar_focus": "grammar", "cefr": "A2"}
        )

        for exercise in unit_data["exercises"]:
            chunks = chunk_exercises(exercise["questions"], exercise["answers"], chunk_size=8)

            for chunk_idx, (chunk_q, chunk_a) in enumerate(chunks):
                # Infer topic
                topic = infer_topic(chunk_q, chunk_a)
                topic_str = topic if topic else "vocabulary"

                # Create JSON exercises
                json_exercises = []
                for q, a in zip(chunk_q, chunk_a):
                    json_exercises.append(
                        {
                            "type": exercise["exercise_type"],
                            "question": q,
                            "answer": a,
                            "hint": unit_meta["grammar_focus"],
                        }
                    )

                # Create training example
                user_prompt = (
                    f"Generate {len(json_exercises)} exercises:\n"
                    f"CEFR Level: {unit_meta['cefr']}\n"
                    f"Grammar Focus: {unit_meta['grammar_focus']}\n"
                    f"Topic: {topic_str}\n"
                    f"Exercise Types: {exercise['exercise_type']}"
                )

                example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert Italian language teacher. Generate high-quality exercises based on the assignment specification. Output exercises in JSON format.",
                        },
                        {"role": "user", "content": user_prompt},
                        {
                            "role": "assistant",
                            "content": json.dumps(json_exercises, ensure_ascii=False, indent=2),
                        },
                    ],
                    "metadata": {
                        "cefr_level": unit_meta["cefr"],
                        "grammar_focus": unit_meta["grammar_focus"],
                        "topic": topic_str,
                        "num_exercises": len(json_exercises),
                        "source": "Basic Italian Grammar Workbook",
                        "unit": unit_num,
                        "exercise": exercise["number"],
                        "exercise_type": exercise["exercise_type"],
                        "chunk": chunk_idx + 1 if len(chunks) > 1 else None,
                    },
                }
                all_examples.append(example)

    return all_examples


def main():
    parser = argparse.ArgumentParser(description="Process extracted textbook exercises")
    parser.add_argument("input", help="Input JSON file with extracted exercises")
    parser.add_argument(
        "--output", default="data/datasets/v4/textbook_extracted.jsonl", help="Output JSONL file"
    )
    parser.add_argument("--merge", action="store_true", help="Merge with existing v4 training data")
    args = parser.parse_args()

    # Load extracted data
    with open(args.input, "r", encoding="utf-8") as f:
        units_data = json.load(f)

    # Convert to training format
    examples = create_training_examples(units_data)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✅ Processed {len(units_data)} units → {len(examples)} training examples")
    print(f"   Saved to: {output_path}")

    # Merge if requested
    if args.merge:
        v4_path = Path("data/datasets/v4/exercise_generation_train.jsonl")
        if v4_path.exists():
            # Read existing
            existing = []
            with open(v4_path, "r", encoding="utf-8") as f:
                existing = f.readlines()

            # Append new data
            with open(output_path, "r", encoding="utf-8") as f:
                new_data = f.readlines()

            # Write combined
            with open(v4_path, "w", encoding="utf-8") as f:
                f.writelines(existing + new_data)

            print(
                f"✅ Merged with v4: {len(existing)} + {len(new_data)} = {len(existing) + len(new_data)} total"
            )

    # Statistics
    exercise_types = defaultdict(int)
    topics = defaultdict(int)
    for ex in examples:
        exercise_types[ex["metadata"]["exercise_type"]] += 1
        topics[ex["metadata"]["topic"]] += 1

    print(f"\nExercise Types:")
    for et, count in sorted(exercise_types.items()):
        print(f"  {et}: {count}")

    print(f"\nTop 10 Topics:")
    for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {topic}: {count}")


if __name__ == "__main__":
    main()
