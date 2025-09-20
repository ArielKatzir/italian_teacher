"""
Inference utilities for the fine-tuned Marco Italian teaching model.

Provides easy interface for using the LoRA-trained model for conversation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class MarcoInference:
    """Inference interface for Marco Italian teaching model."""

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        lora_adapter_path: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Initialize Marco model for inference.

        Args:
            base_model_name: Base model identifier
            lora_adapter_path: Path to trained LoRA adapter (if available)
            device: Device to load model on
        """
        self.base_model_name = base_model_name
        self.lora_adapter_path = lora_adapter_path
        self.device = device

        self.tokenizer = None
        self.model = None

        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer."""

        logger.info(f"Loading tokenizer: {self.base_model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        logger.info(f"Loading base model: {self.base_model_name}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map=self.device if self.device != "auto" else "auto",
            trust_remote_code=True,
        )

        # Load LoRA adapter if available
        if self.lora_adapter_path and Path(self.lora_adapter_path).exists():
            logger.info(f"Loading LoRA adapter: {self.lora_adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.lora_adapter_path)
            logger.info("✅ LoRA adapter loaded successfully")
        else:
            logger.info("Using base model without LoRA fine-tuning")

        self.model.eval()

    def generate_response(
        self,
        conversation: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Generate response from Marco for a given conversation.

        Args:
            conversation: List of message dicts with 'role' and 'content'
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling

        Returns:
            Generated response text
        """

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Extract only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip()

    def chat(self, user_message: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Simple chat interface with Marco.

        Args:
            user_message: User's message
            conversation_history: Previous conversation context

        Returns:
            Marco's response
        """

        # Build conversation
        conversation = conversation_history or []
        conversation.append({"role": "user", "content": user_message})

        # Generate response
        response = self.generate_response(conversation)

        return response

    def explain_italian_grammar(self, italian_sentence: str) -> str:
        """
        Ask Marco to explain Italian grammar for a sentence.

        Args:
            italian_sentence: Italian sentence to analyze

        Returns:
            Grammar explanation
        """

        prompt = f"Can you explain the grammar in this Italian sentence: '{italian_sentence}'?"
        return self.chat(prompt)

    def translate_and_explain(self, italian_sentence: str) -> str:
        """
        Ask Marco to translate and explain an Italian sentence.

        Args:
            italian_sentence: Italian sentence to translate and explain

        Returns:
            Translation and explanation
        """

        prompt = (
            f"Please translate this Italian sentence and explain the grammar: '{italian_sentence}'"
        )
        return self.chat(prompt)

    def generate_practice_question(self, level: str, topic: str, question_type: str) -> str:
        """
        Generate a practice question for Italian learning.

        Args:
            level: CEFR level (A1, A2, B1, B2)
            topic: Topic (e.g., "food", "family", "travel")
            question_type: Type of question (e.g., "fill-in-the-blank", "translation")

        Returns:
            Generated practice question
        """

        prompt = f"Please create a {question_type} practice question for {level} level Italian learners about {topic}."
        return self.chat(prompt)


# Example usage functions
def quick_test_marco(lora_adapter_path: Optional[str] = None):
    """Quick test of Marco model functionality."""

    print("🧑‍🏫 Testing Marco Italian Teacher...")

    # Initialize Marco
    marco = MarcoInference(lora_adapter_path=lora_adapter_path)

    # Test conversations
    test_cases = [
        "What does 'Buongiorno' mean?",
        "Can you explain the grammar in 'Ho mangiato una pizza'?",
        "Help me practice Italian greetings at A1 level",
        "What's the difference between 'sono' and 'sto'?",
    ]

    for i, question in enumerate(test_cases, 1):
        print(f"\n💬 Test {i}: {question}")
        try:
            response = marco.chat(question)
            print(f"🤖 Marco: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n✅ Marco testing complete!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Test with base model (no LoRA)
    quick_test_marco()
