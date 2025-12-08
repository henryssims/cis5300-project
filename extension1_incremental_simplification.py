import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm
from score import sari_score
import numpy as np
import os


def get_sentence_prefix(source: str, completion_ratio: float) -> str:
    """
    Extract a prefix of the source sentence based on completion ratio.
    
    Args:
        source: Full source sentence
        completion_ratio: Fraction of sentence to include (0.25, 0.5, 0.75, 1.0)
    
    Returns:
        Prefix of the sentence
    """
    tokens = source.split()
    num_tokens = len(tokens)
    prefix_length = max(1, int(num_tokens * completion_ratio))
    
    # Try to end at a word boundary (don't cut mid-word)
    prefix_tokens = tokens[:prefix_length]
    return ' '.join(prefix_tokens)


def incremental_simplify(
    model,
    tokenizer,
    source: str,
    completion_ratio: float,
    max_length: int = 128,
    num_beams: int = 4,
    length_penalty: float = 0.6,
    device: str = 'cuda'
) -> str:
    """
    Generate simplified text from a partial sentence prefix.
    
    Args:
        model: Fine-tuned T5 model
        tokenizer: T5 tokenizer
        source: Full source sentence
        completion_ratio: Fraction of sentence to process
        max_length: Maximum generation length
        num_beams: Beam search width
        length_penalty: Penalty for longer sequences (lower = shorter outputs)
        device: Device to run on
    
    Returns:
        Simplified text
    """
    # Get the prefix
    prefix = get_sentence_prefix(source, completion_ratio)
    
    model.eval()
    with torch.no_grad():
        input_text = "simplify: " + prefix
        input_enc = tokenizer(
            input_text,
            max_length=max_length,
            truncation=True,
            return_tensors='pt'
        ).to(device)

        output_ids = model.generate(
            input_ids=input_enc['input_ids'],
            attention_mask=input_enc['attention_mask'],
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=True,
            no_repeat_ngram_size=2  # Avoid repetition in partial contexts
        )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_text


def adaptive_incremental_simplify(
    model,
    tokenizer,
    source: str,
    completion_ratio: float,
    max_length: int = 128,
    num_beams: int = 4,
    device: str = 'cuda'
) -> str:
    """
    Generate simplified text with adaptive length penalty based on context availability.
    Less context -> shorter outputs (higher penalty), more context -> allow longer outputs.
    """
    # Adaptive length penalty: less context = shorter outputs
    # This helps avoid generating incomplete or incoherent simplifications
    if completion_ratio <= 0.25:
        length_penalty = 0.4  # Strong penalty for very short outputs
    elif completion_ratio <= 0.5:
        length_penalty = 0.5
    elif completion_ratio <= 0.75:
        length_penalty = 0.6
    else:
        length_penalty = 0.7  # Allow longer outputs with full context
    
    return incremental_simplify(
        model, tokenizer, source, completion_ratio,
        max_length=max_length,
        num_beams=num_beams,
        length_penalty=length_penalty,
        device=device
    )


def main():
    model_dir = './t5-simplification'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_length = 128
    
    # Convert to absolute path to avoid HuggingFace repo ID validation
    model_dir = os.path.abspath(model_dir)
    
    splits = {
        'test': 'wiki.full.aner.ori.test.95.tsv'
    }

    print("Loading fine-tuned T5 model...")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model directory {model_dir} not found. "
            "Please run strong_baseline.py first to train the model."
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")

    print("Loading WikiLarge test dataset...")
    test = pd.read_csv(
        "hf://datasets/bogdancazan/wikilarge-text-simplification/" + splits["test"],
        sep="\t"
    )
    print(f"Loaded {len(test)} test examples")

    # Test different completion ratios
    completion_ratios = [0.25, 0.5, 0.75, 1.0]
    
    # Evaluate fixed length penalty strategy
    print("\nEvaluating incremental simplification with fixed length penalty...")
    fixed_results = {ratio: {'sari': [], 'keep': [], 'delete': [], 'add': []} for ratio in completion_ratios}
    
    for idx, row in tqdm(test.iterrows(), total=len(test), desc="Processing"):
        source = row['Normal']
        reference = row['Simple']
        
        for ratio in completion_ratios:
            output = incremental_simplify(
                model, tokenizer, source, ratio,
                max_length=max_length,
                num_beams=4,
                length_penalty=0.6,
                device=device
            )
            
            sari, components = sari_score(source, output, [reference])
            fixed_results[ratio]['sari'].append(sari)
            fixed_results[ratio]['keep'].append(components['keep'])
            fixed_results[ratio]['delete'].append(components['delete'])
            fixed_results[ratio]['add'].append(components['add'])

    # Evaluate adaptive length penalty strategy
    print("\nEvaluating incremental simplification with adaptive length penalty...")
    adaptive_results = {ratio: {'sari': [], 'keep': [], 'delete': [], 'add': []} for ratio in completion_ratios}
    
    for idx, row in tqdm(test.iterrows(), total=len(test), desc="Processing"):
        source = row['Normal']
        reference = row['Simple']
        
        for ratio in completion_ratios:
            output = adaptive_incremental_simplify(
                model, tokenizer, source, ratio,
                max_length=max_length,
                num_beams=4,
                device=device
            )
            
            sari, components = sari_score(source, output, [reference])
            adaptive_results[ratio]['sari'].append(sari)
            adaptive_results[ratio]['keep'].append(components['keep'])
            adaptive_results[ratio]['delete'].append(components['delete'])
            adaptive_results[ratio]['add'].append(components['add'])

    # Print fixed results
    print("\n" + "=" * 70)
    print("Fixed Length Penalty Results")
    print("=" * 70)
    print(f"\n{'Completion Ratio':<20} {'SARI':<10} {'Keep':<10} {'Delete':<10} {'Add':<10}")
    print("-" * 70)
    
    for ratio in completion_ratios:
        sari_avg = np.mean(fixed_results[ratio]['sari'])
        keep_avg = np.mean(fixed_results[ratio]['keep'])
        del_avg = np.mean(fixed_results[ratio]['delete'])
        add_avg = np.mean(fixed_results[ratio]['add'])
        
        print(f"{ratio*100:>5.0f}%{'':<14} {sari_avg:>6.2f}    {keep_avg:>6.2f}    {del_avg:>6.2f}    {add_avg:>6.2f}")
    
    print("=" * 70)

    # Print adaptive results
    print("\n" + "=" * 70)
    print("Adaptive Length Penalty Results")
    print("=" * 70)
    print(f"\n{'Completion Ratio':<20} {'SARI':<10} {'Keep':<10} {'Delete':<10} {'Add':<10}")
    print("-" * 70)
    
    for ratio in completion_ratios:
        sari_avg = np.mean(adaptive_results[ratio]['sari'])
        keep_avg = np.mean(adaptive_results[ratio]['keep'])
        del_avg = np.mean(adaptive_results[ratio]['delete'])
        add_avg = np.mean(adaptive_results[ratio]['add'])
        
        print(f"{ratio*100:>5.0f}%{'':<14} {sari_avg:>6.2f}    {keep_avg:>6.2f}    {del_avg:>6.2f}    {add_avg:>6.2f}")
    
    print("=" * 70)

    # Compare fixed vs adaptive
    print("\n" + "=" * 70)
    print("Comparison: Fixed vs Adaptive Length Penalty")
    print("=" * 70)
    print(f"\n{'Completion Ratio':<20} {'Fixed SARI':<15} {'Adaptive SARI':<15} {'Improvement':<15}")
    print("-" * 70)
    
    for ratio in completion_ratios:
        fixed_sari = np.mean(fixed_results[ratio]['sari'])
        adaptive_sari = np.mean(adaptive_results[ratio]['sari'])
        improvement = adaptive_sari - fixed_sari
        
        print(f"{ratio*100:>5.0f}%{'':<14} {fixed_sari:>8.2f}      {adaptive_sari:>8.2f}      {improvement:>+8.2f}")
    
    print("=" * 70)

    # Quality degradation analysis
    full_context_sari = np.mean(adaptive_results[1.0]['sari'])
    print("\n" + "=" * 70)
    print("Quality Degradation Analysis (Adaptive Strategy)")
    print("=" * 70)
    print(f"\nFull context (100%) SARI: {full_context_sari:.2f}")
    print(f"\n{'Completion Ratio':<20} {'SARI':<15} {'% of Full Quality':<20}")
    print("-" * 70)
    
    for ratio in completion_ratios:
        sari = np.mean(adaptive_results[ratio]['sari'])
        pct_quality = (sari / full_context_sari) * 100 if full_context_sari > 0 else 0
        print(f"{ratio*100:>5.0f}%{'':<14} {sari:>8.2f}      {pct_quality:>15.1f}%")
    
    print("=" * 70)

    # Example outputs
    print("\n" + "=" * 70)
    print("Example: Incremental Simplification")
    print("=" * 70)
    
    example_idx = 0
    source = test.iloc[example_idx]['Normal']
    reference = test.iloc[example_idx]['Simple']
    
    print(f"\nSource: {source}")
    print(f"Reference: {reference}")
    print("\n" + "-" * 70)
    
    for ratio in completion_ratios:
        prefix = get_sentence_prefix(source, ratio)
        output = adaptive_incremental_simplify(
            model, tokenizer, source, ratio,
            max_length=max_length,
            num_beams=4,
            device=device
        )
        sari, _ = sari_score(source, output, [reference])
        
        print(f"\n[{ratio*100:.0f}% context]")
        print(f"  Prefix: {prefix}")
        print(f"  Output: {output}")
        print(f"  SARI: {sari:.2f}")


if __name__ == '__main__':
    main()
