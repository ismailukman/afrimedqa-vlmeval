import pandas as pd


def load_parallel_corpus(
    source_file: str,
    target_file: str,
) -> pd.DataFrame:
    """Load English and target-language TSVs, filter to SAQ rows, and merge
    on sample_id to produce aligned translation pairs.

    Returns a DataFrame with below columns:
        sample_id, question_src, answer_src, question_tgt, answer_tgt,
        specialty, country, gender
    """
    src = pd.read_csv(source_file, sep='\t')
    tgt = pd.read_csv(target_file, sep='\t')

    src_saq = src[src['question_type'] == 'SAQ'].copy()
    tgt_saq = tgt[tgt['question_type'] == 'SAQ'].copy()

    if src_saq.empty:
        raise ValueError(f"No SAQ rows found in source file: {source_file}")
    if tgt_saq.empty:
        raise ValueError(f"No SAQ rows found in target file: {target_file}")

    keep_src = ['sample_id', 'question', 'answer', 'specialty', 'country', 'gender']
    keep_tgt = ['sample_id', 'question', 'answer']

    # Only keep columns that actually exist
    keep_src = [c for c in keep_src if c in src_saq.columns]
    keep_tgt = [c for c in keep_tgt if c in tgt_saq.columns]

    merged = src_saq[keep_src].merge(
        tgt_saq[keep_tgt],
        on='sample_id',
        suffixes=('_src', '_tgt'),
    )

    n_src = len(src_saq)
    n_tgt = len(tgt_saq)
    n_merged = len(merged)
    print(f"SAQ pairs: {n_src} source / {n_tgt} target → {n_merged} aligned")

    unmatched = set(src_saq['sample_id']) - set(tgt_saq['sample_id'])
    if unmatched:
        print(f"Warning: {len(unmatched)} source sample_ids have no target match: {sorted(unmatched)}")

    return merged
