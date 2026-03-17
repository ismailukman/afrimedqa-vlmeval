from __future__ import annotations

from comet import download_model, load_from_checkpoint

_model = None


def _get_model():
    global _model
    if _model is None:
        model_path = download_model("McGill-NLP/ssa-comet-mtl")
        _model = load_from_checkpoint(model_path)
    return _model


def score_ssa_comet(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    batch_size: int = 8,
    gpus: int = 1,
) -> tuple[list[float], float]:
    """Run SSA-COMET on a src, hyp, ref triples.

    Returns:
        sentence_scores: per-sample scores (0–1)
        system_score:    corpus-level score (0–1)
    """
    model = _get_model()
    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]
    prediction = model.predict(data, batch_size=batch_size, gpus=gpus)
    return prediction.scores, prediction.system_score
