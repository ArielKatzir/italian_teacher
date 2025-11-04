"""
Homework Generator

Uses the fine-tuned Italian teaching model to generate structured exercises based on
teacher specifications and assignment parameters.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ...fine_tuning.inference import ItalianTeacherInference
from .command_processor import CEFRLevel, ExerciseType, GrammarFocus, HomeworkAssignment

logger = logging.getLogger(__name__)


@dataclass
class Exercise:
    """Individual homework exercise."""

    exercise_type: ExerciseType
    question: str
    answer: Optional[str] = None
    choices: Optional[List[str]] = None  # For multiple choice
    hint: Optional[str] = None
    difficulty: int = 1  # 1-5 scale within CEFR level


@dataclass
class HomeworkSet:
    """Complete set of homework exercises."""

    assignment: HomeworkAssignment
    exercises: List[Exercise]
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {
                "generated_at": None,
                "model_version": "italian_exercise_generator",
                "total_exercises": len(self.exercises),
            }


class HomeworkGenerator:
    """Generates homework exercises using the fine-tuned Italian teaching model."""

    def __init__(self, model_path: str = "models/minerva_marco_v3_merged", device: str = "auto"):
        """
        Initialize homework generator with fine-tuned model.

        Args:
            model_path: Path to the fine-tuned Marco model
            device: Device to run inference on
        """
        self.model_path = Path(model_path)
        self.device = device
        self.marco_model = None
        self._exercise_templates = self._load_exercise_templates()

        logger.info(f"Initialized HomeworkGenerator with model: {model_path}")

    def _load_model(self):
        """Lazy load the Marco model for inference."""
        if self.marco_model is None:
            try:
                # For merged models, pass as base_model_name (not lora_adapter_path)
                self.marco_model = MarcoInference(
                    base_model_name=str(self.model_path),  # Merged model path
                    lora_adapter_path=None,  # No separate LoRA adapter
                    device=self.device,
                )
                logger.info("Marco model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Marco model: {e}")
                raise

    def generate_homework(self, assignment: HomeworkAssignment) -> HomeworkSet:
        """
        Generate a complete homework set based on assignment parameters.

        Args:
            assignment: Homework assignment specification

        Returns:
            HomeworkSet with generated exercises

        Example:
            >>> generator = HomeworkGenerator()
            >>> assignment = HomeworkAssignment(
            ...     cefr_level=CEFRLevel.A2,
            ...     grammar_focus=GrammarFocus.PAST_TENSE,
            ...     topic="history of Milan",
            ...     quantity=3
            ... )
            >>> homework = generator.generate_homework(assignment)
            >>> len(homework.exercises)
            3
        """
        logger.info(f"Generating homework for assignment: {assignment}")

        self._load_model()

        exercises = []
        for i in range(assignment.quantity):
            try:
                exercise = self._generate_single_exercise(assignment, i + 1)
                exercises.append(exercise)
            except Exception as e:
                logger.error(f"Failed to generate exercise {i + 1}: {e}")
                # Continue with remaining exercises

        homework_set = HomeworkSet(assignment=assignment, exercises=exercises)

        logger.info(f"Generated {len(exercises)} exercises")
        return homework_set

    def _generate_single_exercise(
        self, assignment: HomeworkAssignment, exercise_number: int
    ) -> Exercise:
        """Generate a single exercise based on assignment parameters."""

        # Select exercise type for this exercise
        exercise_type = self._select_exercise_type(assignment, exercise_number)

        # Create prompt for the Marco model
        prompt = self._create_exercise_prompt(assignment, exercise_type, exercise_number)

        # Generate exercise using Marco model
        response = self.marco_model.generate_response(prompt, max_tokens=300, temperature=0.7)

        # Parse the model response into structured exercise
        exercise = self._parse_exercise_response(response, exercise_type, assignment)

        return exercise

    def _select_exercise_type(
        self, assignment: HomeworkAssignment, exercise_number: int
    ) -> ExerciseType:
        """Select appropriate exercise type based on assignment and variety."""

        if assignment.exercise_types:
            # Cycle through specified exercise types
            type_index = (exercise_number - 1) % len(assignment.exercise_types)
            return assignment.exercise_types[type_index]

        # Default selection based on CEFR level
        level_defaults = {
            CEFRLevel.A1: [ExerciseType.FILL_IN_BLANK, ExerciseType.MULTIPLE_CHOICE],
            CEFRLevel.A2: [ExerciseType.FILL_IN_BLANK, ExerciseType.TRANSLATION],
            CEFRLevel.B1: [ExerciseType.SENTENCE_COMPLETION, ExerciseType.TRANSLATION],
            CEFRLevel.B2: [ExerciseType.TRANSLATION, ExerciseType.ESSAY],
            CEFRLevel.C1: [ExerciseType.ESSAY, ExerciseType.CONVERSATION],
            CEFRLevel.C2: [ExerciseType.ESSAY, ExerciseType.CONVERSATION],
        }

        default_types = level_defaults.get(assignment.cefr_level, [ExerciseType.FILL_IN_BLANK])
        type_index = (exercise_number - 1) % len(default_types)
        return default_types[type_index]

    def _create_exercise_prompt(
        self, assignment: HomeworkAssignment, exercise_type: ExerciseType, exercise_number: int
    ) -> str:
        """Create a prompt for the Marco model to generate an exercise."""

        # Base prompt template
        base_prompt = f"""Sei Marco, un insegnante di italiano. Crea un esercizio per studenti di livello {assignment.cefr_level.value}.

Parametri dell'esercizio:
- Livello: {assignment.cefr_level.value}
- Tipo di esercizio: {exercise_type.value}"""

        # Add grammar focus if specified
        if assignment.grammar_focus:
            grammar_italian = self._get_italian_grammar_name(assignment.grammar_focus)
            base_prompt += f"\n- Focus grammaticale: {grammar_italian}"

        # Add topic if specified
        if assignment.topic:
            base_prompt += f"\n- Argomento: {assignment.topic}"

        # Add exercise type specific instructions
        type_instructions = self._get_exercise_type_instructions(exercise_type)
        base_prompt += f"\n\n{type_instructions}"

        # Add format requirements
        base_prompt += "\n\nFormato della risposta:"

        if exercise_type == ExerciseType.MULTIPLE_CHOICE:
            base_prompt += """
DOMANDA: [La domanda]
A) [Opzione A]
B) [Opzione B]
C) [Opzione C]
D) [Opzione D]
RISPOSTA_CORRETTA: [A/B/C/D]
SPIEGAZIONE: [Breve spiegazione]"""

        elif exercise_type == ExerciseType.FILL_IN_BLANK:
            base_prompt += """
FRASE: [Frase con spazio vuoto indicato da ___]
RISPOSTA: [Parola/frase corretta]
SPIEGAZIONE: [Breve spiegazione]"""

        elif exercise_type == ExerciseType.TRANSLATION:
            base_prompt += """
ITALIANO: [Frase in italiano da tradurre]
INGLESE: [Traduzione corretta in inglese]
SPIEGAZIONE: [Note sulla traduzione]"""

        else:
            base_prompt += """
DOMANDA: [La domanda o richiesta]
RISPOSTA_ESEMPIO: [Esempio di risposta corretta]
CRITERI: [Criteri di valutazione]"""

        return base_prompt

    def _get_italian_grammar_name(self, grammar: GrammarFocus) -> str:
        """Get Italian name for grammar focus."""
        grammar_italian = {
            GrammarFocus.PRESENT_TENSE: "presente indicativo",
            GrammarFocus.PAST_TENSE: "passato prossimo/imperfetto",
            GrammarFocus.FUTURE_TENSE: "futuro",
            GrammarFocus.SUBJUNCTIVE: "congiuntivo",
            GrammarFocus.CONDITIONAL: "condizionale",
            GrammarFocus.IMPERATIVE: "imperativo",
            GrammarFocus.PASSIVE_VOICE: "forma passiva",
            GrammarFocus.PRONOUNS: "pronomi",
            GrammarFocus.ARTICLES: "articoli",
            GrammarFocus.PREPOSITIONS: "preposizioni",
            GrammarFocus.CONJUNCTIONS: "congiunzioni",
        }
        return grammar_italian.get(grammar, grammar.value)

    def _get_exercise_type_instructions(self, exercise_type: ExerciseType) -> str:
        """Get specific instructions for each exercise type."""
        instructions = {
            ExerciseType.FILL_IN_BLANK: "Crea una frase con uno spazio vuoto che lo studente deve riempire con la parola o forma corretta.",
            ExerciseType.MULTIPLE_CHOICE: "Crea una domanda a scelta multipla con 4 opzioni, di cui solo una corretta.",
            ExerciseType.TRANSLATION: "Fornisci una frase in italiano che lo studente deve tradurre in inglese.",
            ExerciseType.SENTENCE_COMPLETION: "Inizia una frase che lo studente deve completare in modo grammaticalmente corretto.",
            ExerciseType.ESSAY: "Crea una domanda che richiede una risposta di 50-100 parole.",
            ExerciseType.CONVERSATION: "Crea uno scenario di conversazione con domande guida.",
        }
        return instructions.get(
            exercise_type, "Crea un esercizio appropriato per il livello specificato."
        )

    def _parse_exercise_response(
        self, response: str, exercise_type: ExerciseType, assignment: HomeworkAssignment
    ) -> Exercise:
        """Parse the Marco model response into a structured Exercise object."""

        try:
            # Extract different parts based on exercise type
            if exercise_type == ExerciseType.MULTIPLE_CHOICE:
                return self._parse_multiple_choice(response, assignment)
            elif exercise_type == ExerciseType.FILL_IN_BLANK:
                return self._parse_fill_in_blank(response, assignment)
            elif exercise_type == ExerciseType.TRANSLATION:
                return self._parse_translation(response, assignment)
            else:
                return self._parse_general_exercise(response, exercise_type, assignment)

        except Exception as e:
            logger.error(f"Error parsing exercise response: {e}")
            # Return a fallback exercise
            return Exercise(
                exercise_type=exercise_type,
                question="Errore nella generazione dell'esercizio. Riprova.",
                answer="N/A",
            )

    def _parse_multiple_choice(self, response: str, assignment: HomeworkAssignment) -> Exercise:
        """Parse multiple choice exercise from response."""
        lines = response.strip().split("\n")
        question = ""
        choices = []
        answer = ""
        hint = ""

        for line in lines:
            line = line.strip()
            if line.startswith("DOMANDA:"):
                question = line.replace("DOMANDA:", "").strip()
            elif line.startswith(("A)", "B)", "C)", "D)")):
                choices.append(line[3:].strip())
            elif line.startswith("RISPOSTA_CORRETTA:"):
                answer = line.replace("RISPOSTA_CORRETTA:", "").strip()
            elif line.startswith("SPIEGAZIONE:"):
                hint = line.replace("SPIEGAZIONE:", "").strip()

        return Exercise(
            exercise_type=ExerciseType.MULTIPLE_CHOICE,
            question=question,
            answer=answer,
            choices=choices,
            hint=hint,
        )

    def _parse_fill_in_blank(self, response: str, assignment: HomeworkAssignment) -> Exercise:
        """Parse fill-in-blank exercise from response."""
        lines = response.strip().split("\n")
        question = ""
        answer = ""
        hint = ""

        for line in lines:
            line = line.strip()
            if line.startswith("FRASE:"):
                question = line.replace("FRASE:", "").strip()
            elif line.startswith("RISPOSTA:"):
                answer = line.replace("RISPOSTA:", "").strip()
            elif line.startswith("SPIEGAZIONE:"):
                hint = line.replace("SPIEGAZIONE:", "").strip()

        return Exercise(
            exercise_type=ExerciseType.FILL_IN_BLANK, question=question, answer=answer, hint=hint
        )

    def _parse_translation(self, response: str, assignment: HomeworkAssignment) -> Exercise:
        """Parse translation exercise from response."""
        lines = response.strip().split("\n")
        question = ""
        answer = ""
        hint = ""

        for line in lines:
            line = line.strip()
            if line.startswith("ITALIANO:"):
                question = line.replace("ITALIANO:", "").strip()
            elif line.startswith("INGLESE:"):
                answer = line.replace("INGLESE:", "").strip()
            elif line.startswith("SPIEGAZIONE:"):
                hint = line.replace("SPIEGAZIONE:", "").strip()

        return Exercise(
            exercise_type=ExerciseType.TRANSLATION,
            question=f"Traduci in inglese: {question}",
            answer=answer,
            hint=hint,
        )

    def _parse_general_exercise(
        self, response: str, exercise_type: ExerciseType, assignment: HomeworkAssignment
    ) -> Exercise:
        """Parse general exercise format from response."""
        lines = response.strip().split("\n")
        question = ""
        answer = ""
        hint = ""

        for line in lines:
            line = line.strip()
            if line.startswith("DOMANDA:"):
                question = line.replace("DOMANDA:", "").strip()
            elif line.startswith("RISPOSTA_ESEMPIO:"):
                answer = line.replace("RISPOSTA_ESEMPIO:", "").strip()
            elif line.startswith("CRITERI:"):
                hint = line.replace("CRITERI:", "").strip()

        return Exercise(exercise_type=exercise_type, question=question, answer=answer, hint=hint)

    def _load_exercise_templates(self) -> Dict:
        """Load exercise templates for fallback generation."""
        return {
            "fill_in_blank_templates": [
                "Completa la frase con la forma corretta del verbo: Ieri io ___ (andare) al cinema.",
                "Inserisci l'articolo corretto: ___ libro è molto interessante.",
                "Completa con la preposizione giusta: Vado ___ scuola ogni giorno.",
            ],
            "translation_templates": [
                "Traduci: Buongiorno, come stai?",
                "Traduci: Mi piace molto questo ristorante.",
                "Traduci: Domani andrò al mare con i miei amici.",
            ],
        }

    def save_homework_set(self, homework_set: HomeworkSet, output_path: str) -> str:
        """
        Save homework set to file.

        Args:
            homework_set: The homework set to save
            output_path: Path where to save the homework

        Returns:
            Path to saved file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        data = {
            "assignment": asdict(homework_set.assignment),
            "exercises": [asdict(exercise) for exercise in homework_set.exercises],
            "metadata": homework_set.metadata,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Homework set saved to: {output_file}")
        return str(output_file)
