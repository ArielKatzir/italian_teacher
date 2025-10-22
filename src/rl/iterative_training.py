import asyncio
import gc
import json
import re
from typing import Dict, List, Tuple, Union

import json5
import torch
from tqdm.auto import tqdm  # For progress bars

# transformers imports (assumes transformers, bitsandbytes installed)
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the official prompt formatter
try:
    from .prompt_formatter import format_prompt_with_chat_template
except ImportError:
    from src.rl.prompt_formatter import format_prompt_with_chat_template


# keep your create_round2_dataset as-is
def create_round2_dataset(
    low_scoring: List[Dict], high_scoring: List[Dict], low_weight: float = 0.7
) -> List[Dict]:
    import random

    total_size = len(low_scoring) + len(high_scoring)
    num_low = int(total_size * low_weight)
    num_high = total_size - num_low

    selected_low = random.choices(low_scoring, k=min(num_low, len(low_scoring)))
    selected_high = random.choices(high_scoring, k=min(num_high, len(high_scoring)))

    round2_data = selected_low + selected_high
    random.shuffle(round2_data)

    print(f"\n📚 Round 2 dataset created:")
    print(f"  Low-scoring examples: {len(selected_low)} ({low_weight*100:.0f}%)")
    print(f"  High-scoring examples: {len(selected_high)} ({(1-low_weight)*100:.0f}%)")
    print(f"  Total: {len(round2_data)}")

    return [r["request"] for r in round2_data]


def _maybe_move_sentence_transformer_to_cuda(reward_fn):
    """
    Try to move a SentenceTransformer used by TopicScorer to CUDA for speed.
    Silently continue if not present or fails.
    """
    try:
        # many reward_fn implementers have scorers dict with 'topic'
        topic = getattr(reward_fn, "scorers", {}).get("topic", None)
        if topic is not None:
            sim = getattr(topic, "similarity_model", None)
            if sim is not None:
                try:
                    sim.to("cuda")
                    print("✅ SentenceTransformer moved to CUDA for topic scoring.")
                except Exception as e:
                    print("⚠️ Could not move SentenceTransformer to CUDA:", e)
    except Exception:
        pass


def _move_model_to_cuda_or_reload_quantized(
    model_or_path: Union[str, torch.nn.Module],
    tokenizer,
    quantize: bool = True,
    dtype=torch.float16,
):
    """
    If model_or_path is a path string -> load quantized (8-bit) with device_map="auto".
    If it's a model instance -> try to .to('cuda') it; if OOM, raise an explanatory error.
    Returns: model (on CUDA) and a flag `reloaded` (True if reloaded from path).
    """
    reloaded = False
    # If user passed a path string: prefer to load with 8-bit quantization
    if isinstance(model_or_path, str):
        model_path = model_or_path
        try:
            print(f"Loading model from path '{model_path}' with 8-bit quantization...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                load_in_8bit=True,
                torch_dtype=dtype,
            )
            reloaded = True
            return model, reloaded
        except Exception as e:
            print("⚠️  Failed to load in 8-bit automatically:", e)
            print("Attempting standard (non-quantized) load to try fallback...")
            model = AutoModelForCausalLM.from_pretrained(model_path)
            model.to("cuda")
            reloaded = True
            return model, reloaded

    # If user passed a model instance: try to move it to cuda
    elif isinstance(model_or_path, torch.nn.Module):
        model = model_or_path
        reloaded = False
        try:
            # Check if model is already on CUDA. device_map="auto" might have placed it there.
            is_on_cuda = next(model.parameters()).is_cuda
            # For quantized models with device_map, some parts might be on CPU.
            # The presence of `hf_device_map` is a better indicator.
            is_quantized_and_mapped = hasattr(model, "hf_device_map")

            if is_quantized_and_mapped:
                print("✅ Model is already quantized and mapped to devices.")
            elif not is_on_cuda:
                print("Attempting to move model to CUDA...")
                model.to("cuda")
                print("✅ Model is on CUDA.")

            return model, reloaded
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(
                "CUDA out of memory when trying to move model to GPU. "
                "If the model was not loaded with quantization, try passing the model path string "
                "to `evaluate_model_on_requests` to load it in 8-bit automatically."
            ) from e
    return model_or_path, False


async def evaluate_model_on_requests(
    model_or_path: Union[str, torch.nn.Module],
    tokenizer: AutoTokenizer,
    reward_fn,
    requests: List[Dict],
    output_path: str = None,
    batch_size: int = 4,
    max_new_tokens: int = 400,
    quantize_if_needed: bool = True,
    save_each_batch: bool = True,
    generation_kwargs: Dict = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    GPU-optimised, batched evaluation. Accepts either:
      - a model instance (torch.nn.Module) OR
      - a path string to the model (will try to load in 8-bit)

    Returns (low_scoring, high_scoring)

    Args:
        ...
        generation_kwargs: Optional dictionary of kwargs to pass to `model.generate`.
    """

    # 1) Load / move model to GPU (or load 8-bit if path given)
    model, reloaded = _move_model_to_cuda_or_reload_quantized(
        model_or_path, tokenizer, quantize=quantize_if_needed
    )

    model.eval()
    # 2) Move SentenceTransformer (TopicScorer) to GPU if possible
    _maybe_move_sentence_transformer_to_cuda(reward_fn)

    # 3) Sanity print
    print(
        f"📊 Evaluating {len(requests)} requests in batches of {batch_size} (max_new_tokens={max_new_tokens})"
    )
    try:
        print("Model device:", next(model.parameters()).device)
    except StopIteration:
        print("Warning: model has no parameters (?)")

    low_scoring = []
    high_scoring = []

    # Helper to save partial results
    def _save_partial():
        if not output_path:
            return
        results = {
            "low_scoring": low_scoring,
            "high_scoring": high_scoring,
            "stats": {
                "processed": len(low_scoring) + len(high_scoring),
                "low_count": len(low_scoring),
                "high_count": len(high_scoring),
            },
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    # iterate batches
    n_batches = (len(requests) + batch_size - 1) // batch_size
    for bidx in tqdm(range(n_batches), desc="Evaluating Batches"):
        start = bidx * batch_size
        batch = requests[start : start + batch_size]

        # Use the official, imported prompt formatter
        # The `add_examples` parameter is now ignored by the formatter, so we can omit it.
        prompts = [format_prompt_with_chat_template(req, tokenizer) for req in batch]

        # Tokenize batched prompts
        # For causal LMs we pass input_ids and attention_mask to generate
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate on GPU
        with torch.no_grad():
            try:
                # Set default generation strategy if not provided
                gen_kwargs = generation_kwargs or {
                    "do_sample": False,  # deterministic by default
                }

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    **gen_kwargs,
                )
            except RuntimeError as e:
                # If OOM at generation time, try smaller batch or suggest quantized reload
                torch.cuda.empty_cache()
                gc.collect()
                raise RuntimeError(
                    "RuntimeError during generation (possible OOM). Try reducing batch_size, "
                    "or reload model in 8-bit. Original error:\n" + str(e)
                ) from e

        # outputs shape: (batch_size, seq_len_out)
        # The 'outputs' from generate includes the prompt tokens. We must slice them off.
        prompt_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, prompt_len:]

        # Decode only the newly generated part
        generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for i, generated_text in enumerate(generated_texts):

            request = batch[i]
            exercises = None
            try:
                # Use regex to find the JSON array. This is more robust than simple slicing.
                # It handles cases where the model might add extra text after the JSON.
                # The pattern looks for a string that starts with '[' and ends with ']'
                # and contains balanced curly braces inside.
                match = re.search(r"(\[[\s\S]*\])", generated_text)
                if not match:
                    raise json.JSONDecodeError(
                        "No valid JSON array found in the output.", generated_text, 0
                    )

                json_str = match.group(1)

                # Use json5 for robust parsing (handles trailing commas, etc.)
                exercises = json5.loads(json_str)
                if not isinstance(exercises, list):
                    exercises = [exercises]

                if not exercises:
                    raise ValueError("Parsed JSON is an empty list.")

                # --- ASYNC REWARD CALCULATION ---
                # Use the efficient `score_exercises` to run all scoring concurrently.
                # We `await` it directly now that the parent function is async.
                avg_score, results = await reward_fn.score_exercises(exercises, request)

                # Get the breakdown from the first exercise for logging purposes
                first_breakdown = results[0][1] if results else "N/A"

                if avg_score < 92:
                    low_scoring.append(
                        {
                            "request": request,
                            "generated": generated_text,
                            "parsed_exercises": exercises,
                            "score": avg_score,
                            "breakdown": str(first_breakdown),
                        }
                    )
                else:
                    high_scoring.append(
                        {
                            "request": request,
                            "generated": generated_text,
                            "score": avg_score,
                        }
                    )

            except Exception as e:
                low_scoring.append(
                    {
                        "request": request,
                        "generated": generated_text,  # Log the full text on error
                        "score": 10.0,
                        "error": str(e),
                    }
                )

        # finished batch -> free some GPU memory and optionally save partial results
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        if output_path and save_each_batch:
            _save_partial()

    # done
    total = len(low_scoring) + len(high_scoring)
    avg_score = None
    if total > 0:
        avg_score = sum(r["score"] for r in low_scoring + high_scoring) / total

    print(f"\n✅ Evaluation complete. Processed: {total}")
    print(f"  Low scoring (<92): {len(low_scoring)}")
    print(f"  High scoring (>=92): {len(high_scoring)}")
    if avg_score is not None:
        print(f"  Avg score (processed): {avg_score:.2f}")

    # final save
    if output_path:
        _save_partial()
        print(f"💾 Final results saved to {output_path}")

    return low_scoring, high_scoring


# Example usage:
# In a Jupyter cell or async script, you would now run it with `await`:
#
# low, high = await evaluate_model_on_requests(
#     './models/italian_v6_grpo_pilot', tokenizer, reward_fn, validation_requests, batch_size=4
# )
