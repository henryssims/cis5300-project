import pandas as pd
from score import sari_score

def simple_baseline(source: str) -> str:
    return source


def main():
    splits = {
        'train': 'wiki.full.aner.ori.train.95.tsv',
        'validation': 'wiki.full.aner.ori.valid.95.tsv',
        'test': 'wiki.full.aner.ori.test.95.tsv'
    }

    test = pd.read_csv(
        "hf://datasets/bogdancazan/wikilarge-text-simplification/" + splits["test"],
        sep="\t"
    )


    outputs = []
    for source in test['Normal']:
        output = simple_baseline(source)
        outputs.append(output)


    sari_scores = []
    keep_scores = []
    del_scores = []
    add_scores = []

    for output, source, reference in zip(outputs, test['Normal'], test['Simple']):
        sari, components = sari_score(source, output, [reference])
        sari_scores.append(sari)
        keep_scores.append(components['keep'])
        del_scores.append(components['delete'])
        add_scores.append(components['add'])

    print("\n" + "=" * 50)
    print("Simple Baseline SARI Score Results")
    print("=" * 50)
    print(f"Number of samples: {len(sari_scores)}")
    print()
    print(f"  SARI:        {sum(sari_scores) / len(sari_scores):.2f}")
    print(f"    - Keep:    {sum(keep_scores) / len(keep_scores):.2f}")
    print(f"    - Delete:  {sum(del_scores) / len(del_scores):.2f}")
    print(f"    - Add:     {sum(add_scores) / len(add_scores):.2f}")
    print("=" * 50)


if __name__ == '__main__':
    main()