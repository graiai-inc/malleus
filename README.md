# malleus

Confidence-weighted ROVER fusion for clinical ASR using model log-probabilities.

## Overview

Standard ROVER (Recognizer Output Voting Error Reduction) uses majority voting or flat confidence to combine ASR hypotheses. This project extends ROVER with per-word confidence scores derived from model log-probabilities, temperature-scaled calibration, and Tsallis entropy-based hallucination detection to improve fusion quality for clinical speech recognition.

## Approach

1. **Token log-probability extraction** from sherpa-onnx internals (already computed during decoding, normally discarded)
2. **Confidence calibration** via temperature scaling and isotonic regression, with model-specific adjustments
3. **Confidence-weighted ROVER** where per-word model confidence drives the voting, not flat weights
4. **Hallucination detection** using Tsallis entropy over the probability distribution

## Baseline

Naive ROVER results from [stapes](https://github.com/graiai-inc/stapes) serve as the baseline comparison.

## Status

Research and implementation design complete. Experiments pending.

## License

MIT
