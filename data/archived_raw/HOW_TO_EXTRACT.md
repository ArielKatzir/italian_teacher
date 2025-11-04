# How to Extract Textbook Exercises

## 📖 **What This Workflow Does**

Converts textbook exercises (from screenshots) into training data for your Italian exercise generation model.

## 🔄 **The Two-Step Process**

### **Step 1: Claude Reads Screenshots & Extracts Data**
Claude reads the images in `data/raw/textbook_screenshots/unitX/` and manually adds exercises to `data/extracted_exercises.json` in this format:

```json
{
  "5": {
    "exercises": [
      {
        "number": 1,
        "instruction": "Complete the sentences using the correct verb form.",
        "exercise_type": "conjugation",
        "questions": ["Io (parlare) ___ italiano.", "Tu (mangiare) ___ la pizza."],
        "answers": ["parlo", "mangi"]
      }
    ]
  }
}
```

### **Step 2: You Run the Processing Script**
```bash
python data/scripts/process_extracted_units.py data/extracted_exercises.json --merge
```

This automatically:
- ✅ Infers topics from vocabulary (e.g., "pizza, pasta, ristorante" → food_drink)
- ✅ Chunks exercises into groups of ≤8
- ✅ Generates proper training format (system → user → assistant)
- ✅ Merges with existing v4 data

## 📂 **Current Status**

**Extracted:** Unit 1 only (in extracted_exercises.json)
**Already in v4:** Units 1-4 (from old hardcoded script)
**Remaining:** Units 5-23

## 🚀 **How to Continue**

### Option A: Claude Extracts Next Units
Claude reads screenshots for Units 5-10 and updates `extracted_exercises.json`, then you run:
```bash
python data/scripts/process_extracted_units.py data/extracted_exercises.json --merge
```

### Option B: You Manually Add to JSON
You can add units to `extracted_exercises.json` yourself by looking at screenshots and following the format.

## 📊 **Topic Inference**

The script automatically infers topics from vocabulary using 22 topic categories with 400+ keywords:

- **daily_life**: giorno, sera, caffè, colazione, dormire...
- **people**: uomo, donna, amico, bambino, persona...
- **places**: casa, città, ristorante, scuola, ospedale...
- **family**: madre, padre, figlio, fratello, nonno...
- **food_drink**: pizza, pasta, vino, pane, mangiare...
- **work_education**: professore, studente, lavoro, ufficio...
- **travel**: treno, aereo, viaggio, hotel, biglietto...
- **time**: ora, giorno, anno, ieri, domani...
- **emotions_states**: felice, stanco, triste, gentile...
- **body**: mano, occhio, testa, piede, corpo...
- **objects**: libro, tavolo, computer, telefono, auto...
- **clothing**: vestito, camicia, scarpe, gonna, cappello...
- **colors**: rosso, blu, bianco, nero, verde...
- **weather**: sole, pioggia, neve, vento, caldo...
- **hobbies_sports**: calcio, tennis, musica, cinema, leggere...
- **technology**: computer, internet, smartphone, email...
- **nature**: albero, fiore, fiume, montagna, animale...
- **house_home**: stanza, cucina, bagno, letto, porta...
- **shopping**: negozio, comprare, prezzo, pagare, euro...
- **health**: medico, ospedale, malattia, medicina, febbre...

If no topic matches (< 2 keywords), it defaults to "vocabulary".

## 🎯 **Exercise Types**

The script recognizes these exercise types:
- `gender_identification` - Identify m/f/m-f for nouns
- `plural_singular` - Convert between singular/plural
- `fill_in_blank` - Complete sentences with blanks
- `conjugation` - Verb conjugation exercises
- `adjective_agreement` - Match adjectives to nouns
- `transformation` - Transform sentences (affirmative/negative)
- `multiple_choice` - Choose correct option
- `question_answer` - Answer questions using specific grammar

## 📈 **Expected Output**

For Units 1-4 (already done):
- 17 exercises → 41 training examples
- Average: ~2.4x expansion (due to chunking)

For all 23 units:
- Estimated: ~100 exercises → ~240 training examples
