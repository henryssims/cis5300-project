import argparse
import json
from collections import Counter
from typing import List, Tuple


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def get_ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def sari_score(source: str, prediction: str, references: List[str]) -> Tuple[float, dict]:
    src_tokens = tokenize(source)
    pred_tokens = tokenize(prediction)
    ref_tokens_list = [tokenize(ref) for ref in references]
    
    keep_scores = []
    del_scores = []
    add_scores = []
    
    for n in range(1, 5):
        src_ngrams = get_ngrams(src_tokens, n)
        pred_ngrams = get_ngrams(pred_tokens, n)
        ref_ngrams_list = [get_ngrams(ref_tokens, n) for ref_tokens in ref_tokens_list]
        
        src_in_refs = Counter()
        for ngram in src_ngrams:
            count = sum(1 for ref_ngrams in ref_ngrams_list if ngram in ref_ngrams)
            if count > 0:
                src_in_refs[ngram] = count
        
        kept_ngrams = Counter()
        for ngram in pred_ngrams:
            if ngram in src_ngrams:
                kept_ngrams[ngram] = min(pred_ngrams[ngram], src_ngrams[ngram])
        
        keep_prec_num = sum(min(kept_ngrams[ng], 1) for ng in kept_ngrams if ng in src_in_refs)
        keep_prec_den = sum(kept_ngrams.values())
        keep_prec = keep_prec_num / keep_prec_den if keep_prec_den > 0 else 0
        
        keep_rec_num = sum(min(kept_ngrams[ng], 1) for ng in src_in_refs if ng in kept_ngrams)
        keep_rec_den = len(src_in_refs)
        keep_rec = keep_rec_num / keep_rec_den if keep_rec_den > 0 else 0
        
        keep_f1 = 2 * keep_prec * keep_rec / (keep_prec + keep_rec) if (keep_prec + keep_rec) > 0 else 0
        keep_scores.append(keep_f1)
        
        src_not_in_refs = set(ng for ng in src_ngrams if ng not in src_in_refs)
        deleted_ngrams = set(ng for ng in src_ngrams if ng not in pred_ngrams)
        
        del_prec_num = len(deleted_ngrams & src_not_in_refs)
        del_prec_den = len(deleted_ngrams)
        del_prec = del_prec_num / del_prec_den if del_prec_den > 0 else 0
        
        del_rec_num = len(deleted_ngrams & src_not_in_refs)
        del_rec_den = len(src_not_in_refs)
        del_rec = del_rec_num / del_rec_den if del_rec_den > 0 else 0
        
        del_f1 = 2 * del_prec * del_rec / (del_prec + del_rec) if (del_prec + del_rec) > 0 else 0
        del_scores.append(del_f1)
        
        added_ngrams = set(ng for ng in pred_ngrams if ng not in src_ngrams)
        
        ref_not_in_src = set()
        for ref_ngrams in ref_ngrams_list:
            ref_not_in_src.update(ng for ng in ref_ngrams if ng not in src_ngrams)
        
        add_prec_num = len(added_ngrams & ref_not_in_src)
        add_prec_den = len(added_ngrams)
        add_prec = add_prec_num / add_prec_den if add_prec_den > 0 else 0
        
        add_scores.append(add_prec)
    
    keep_avg = sum(keep_scores) / 4
    del_avg = sum(del_scores) / 4
    add_avg = sum(add_scores) / 4
    
    sari = (keep_avg + del_avg + add_avg) / 3 * 100
    
    components = {
        'keep': keep_avg * 100,
        'delete': del_avg * 100,
        'add': add_avg * 100
    }
    
    return sari, components


def load_file(filepath: str) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]


def load_references(filepath: str, num_refs: int = 1) -> List[List[str]]:
    lines = load_file(filepath)
    
    if '\t' in lines[0]:
        return [line.split('\t') for line in lines]
    
    return [[line] for line in lines]


def main():
    parser = argparse.ArgumentParser(
        description='Calculate SARI score for text simplification outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--outputs', required=True, help='Path to system output file (one per line)')
    parser.add_argument('--references', required=True, help='Path to reference file (one per line, tab-separated for multiple refs)')
    parser.add_argument('--sources', required=True, help='Path to source/input file (one per line)')

    args = parser.parse_args()

    outputs = load_file(args.outputs)
    references = load_references(args.references)
    sources = load_file(args.sources)

    if not (len(outputs) == len(references) == len(sources)):
        raise ValueError(f"Mismatched lengths: {len(outputs)} outputs, "
                        f"{len(references)} references, {len(sources)} sources")

    sari_scores = []
    keep_scores = []
    del_scores = []
    add_scores = []

    for output, refs, src in zip(outputs, references, sources):
        sari, components = sari_score(src, output, refs)
        sari_scores.append(sari)
        keep_scores.append(components['keep'])
        del_scores.append(components['delete'])
        add_scores.append(components['add'])

    results = {
        'SARI': sum(sari_scores) / len(sari_scores),
        'SARI_keep': sum(keep_scores) / len(keep_scores),
        'SARI_delete': sum(del_scores) / len(del_scores),
        'SARI_add': sum(add_scores) / len(add_scores),
        'num_samples': len(outputs)
    }

    print("=" * 50)
    print("SARI Score Results")
    print("=" * 50)
    print(f"Number of samples: {results['num_samples']}")
    print()
    print(f"  SARI:        {results['SARI']:.2f}")
    print(f"    - Keep:    {results['SARI_keep']:.2f}")
    print(f"    - Delete:  {results['SARI_delete']:.2f}")
    print(f"    - Add:     {results['SARI_add']:.2f}")
    print("=" * 50)


if __name__ == '__main__':
    main()