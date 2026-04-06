import 'dart:math';

/// Calibrates raw confidence scores from ASR models to improve reliability
/// Uses temperature scaling and isotonic regression techniques
class ConfidenceCalibrator {
  // Temperature values per research (full_pipeline.md Phase 4)
  // Streaming: T=2.3, Batch: T=1.5
  static const double defaultStreamingTemperature = 2.3;
  static const double defaultBatchTemperature = 1.5;

  // Entropy thresholds for hallucination detection
  static const double entropyThreshold = 2.5;
  static const double tsallisAlpha = 0.333; // Optimal for ASR per research

  final double streamingTemp;
  final double batchTemp;

  // Calibration statistics (would be learned from data in production)
  final List<CalibrationBin> _calibrationBins = [];

  ConfidenceCalibrator({
    this.streamingTemp = defaultStreamingTemperature,
    this.batchTemp = defaultBatchTemperature,
  }) {
    _initializeCalibrationBins();
  }

  /// Initialize calibration bins based on empirical data
  void _initializeCalibrationBins() {
    // These values would be learned from actual medical ASR data
    // Current values are based on typical ASR calibration curves
    _calibrationBins.addAll([
      CalibrationBin(0.0, 0.1, 0.05),
      CalibrationBin(0.1, 0.2, 0.12),
      CalibrationBin(0.2, 0.3, 0.22),
      CalibrationBin(0.3, 0.4, 0.31),
      CalibrationBin(0.4, 0.5, 0.42),
      CalibrationBin(0.5, 0.6, 0.54),
      CalibrationBin(0.6, 0.7, 0.67),
      CalibrationBin(0.7, 0.8, 0.78),
      CalibrationBin(0.8, 0.9, 0.86),
      CalibrationBin(0.9, 1.0, 0.94),
    ]);
  }

  /// Calibrate confidence scores for a streaming model
  List<double> calibrateStreamingConfidences(List<double> rawScores) {
    return rawScores
        .map(
            (score) => _calibrateScore(score, streamingTemp, isStreaming: true))
        .toList();
  }

  /// Calibrate confidence scores for a batch model
  List<double> calibrateBatchConfidences(List<double> rawScores) {
    return rawScores
        .map((score) => _calibrateScore(score, batchTemp, isStreaming: false))
        .toList();
  }

  /// Apply temperature scaling and isotonic regression
  double _calibrateScore(double rawScore, double temperature,
      {required bool isStreaming}) {
    // Clamp to valid range
    rawScore = rawScore.clamp(0.0, 1.0);

    // Step 1: Temperature scaling
    double scaledScore = _temperatureScale(rawScore, temperature);

    // Step 2: Isotonic regression calibration
    scaledScore = _isotonicCalibration(scaledScore);

    // Step 3: Model-specific adjustment
    if (isStreaming) {
      // Streaming models tend to be overconfident on short utterances
      scaledScore = _adjustForStreamingModel(scaledScore);
    } else {
      // Batch models may hallucinate with high confidence
      scaledScore = _adjustForBatchModel(scaledScore);
    }

    return scaledScore.clamp(0.0, 1.0);
  }

  /// Apply temperature scaling to logits
  double _temperatureScale(double score, double temperature) {
    if (score <= 0.0 || score >= 1.0) return score;

    // Convert probability to logit
    final logit = log(score / (1 - score));

    // Scale by temperature
    final scaledLogit = logit / temperature;

    // Convert back to probability
    return 1.0 / (1.0 + exp(-scaledLogit));
  }

  /// Apply isotonic regression calibration using bins
  double _isotonicCalibration(double score) {
    for (final bin in _calibrationBins) {
      if (score >= bin.minScore && score < bin.maxScore) {
        return bin.calibratedValue;
      }
    }
    return score; // Fallback to original if no bin matches
  }

  /// Adjust confidence for streaming model characteristics
  double _adjustForStreamingModel(double score) {
    // Streaming models are less reliable at beginning/end of utterances
    // Apply a slight penalty
    return score * 0.95;
  }

  /// Adjust confidence for batch model characteristics
  double _adjustForBatchModel(double score) {
    // Batch models may hallucinate with high confidence
    // Apply stronger penalty for very high scores
    if (score > 0.95) {
      return score * 0.92;
    }
    return score;
  }

  /// Calculate Tsallis entropy for hallucination detection
  double calculateTsallisEntropy(List<double> probabilities) {
    if (probabilities.isEmpty) return 0.0;

    double sum = 0.0;
    for (final p in probabilities) {
      if (p > 0) {
        sum += pow(p, tsallisAlpha).toDouble();
      }
    }

    return (1.0 - sum) / (tsallisAlpha - 1.0);
  }

  /// Detect potential hallucinations based on entropy
  /// Requires MULTIPLE indicators, not just one
  bool isPotentialHallucination(List<double> wordConfidences) {
    if (wordConfidences.isEmpty) return false;

    // Calculate metrics
    final entropy = calculateTsallisEntropy(wordConfidences);
    final avgConfidence =
        wordConfidences.reduce((a, b) => a + b) / wordConfidences.length;
    final allHighConfidence = wordConfidences.every((c) => c > 0.9);

    // Count hallucination indicators
    int indicatorCount = 0;

    // Indicator 1: Suspiciously high entropy (very uncertain)
    if (entropy > entropyThreshold) {
      indicatorCount++;
    }

    // Indicator 2: Suspiciously low entropy WITH high confidences (overconfident)
    if (entropy < 0.05 && avgConfidence > 0.85) {
      indicatorCount++;
    }

    // Indicator 3: All words have suspiciously high confidence
    if (allHighConfidence && avgConfidence > 0.85) {
      indicatorCount++;
    }

    // Require AT LEAST 2 indicators to flag as hallucination
    return indicatorCount >= 2;
  }

  /// Extract word-level confidence scores from raw model output
  /// This would interface with sherpa-onnx native code
  List<double> extractWordConfidences(String modelOutput) {
    // This is a placeholder - actual implementation would parse
    // the native sherpa-onnx output format
    // For now, return synthetic confidences for testing

    final words = modelOutput.split(' ');
    final confidences = <double>[];

    for (final word in words) {
      // Generate realistic confidence based on word characteristics
      double confidence = 0.8; // Base confidence

      // Medical terms get lower initial confidence (harder to recognize)
      if (_isMedicalTerm(word)) {
        confidence -= 0.1;
      }

      // Short words are generally more confident
      if (word.length <= 3) {
        confidence += 0.1;
      }

      // Numbers are usually recognized well
      if (RegExp(r'^\d+\.?\d*$').hasMatch(word)) {
        confidence += 0.15;
      }

      confidences.add(confidence.clamp(0.1, 0.99));
    }

    return confidences;
  }

  /// Check if a word is likely a medical term
  bool _isMedicalTerm(String word) {
    // Common medical suffixes
    final medicalSuffixes = [
      'itis',
      'osis',
      'emia',
      'pathy',
      'ectomy',
      'otomy',
      'ostomy',
      'plasty',
      'pexy',
      'tripsy'
    ];

    final lowerWord = word.toLowerCase();

    // Check for medical suffixes
    for (final suffix in medicalSuffixes) {
      if (lowerWord.endsWith(suffix)) return true;
    }

    // Check for drug name patterns
    if (lowerWord.endsWith('azole') ||
        lowerWord.endsWith('cillin') ||
        lowerWord.endsWith('mycin') ||
        lowerWord.endsWith('pril') ||
        lowerWord.endsWith('sartan') ||
        lowerWord.endsWith('statin') ||
        lowerWord.endsWith('olol') ||
        lowerWord.endsWith('prazole')) {
      return true;
    }

    // Check against common medical terms
    final commonTerms = {
      'diagnosis',
      'prognosis',
      'symptom',
      'syndrome',
      'chronic',
      'acute',
      'bilateral',
      'unilateral',
      'proximal',
      'distal',
      'anterior',
      'posterior',
      'superior',
      'inferior',
      'lateral',
      'medial',
      'hypertension',
      'hypotension',
      'tachycardia',
      'bradycardia',
      'diabetes',
      'pneumonia',
      'bronchitis',
      'asthma',
    };

    return commonTerms.contains(lowerWord);
  }

  /// Aggregate word confidences to utterance level
  double aggregateConfidences(List<double> wordConfidences) {
    if (wordConfidences.isEmpty) return 0.0;

    // Use minimum confidence as aggregate (most conservative)
    // Research shows this is most correlated with utterance accuracy
    final minConfidence = wordConfidences.reduce((a, b) => a < b ? a : b);

    // Apply slight boost based on average to avoid being too pessimistic
    final avgConfidence =
        wordConfidences.reduce((a, b) => a + b) / wordConfidences.length;

    // Weighted combination: 70% min, 30% average
    return (minConfidence * 0.7 + avgConfidence * 0.3).clamp(0.0, 1.0);
  }

  /// Create calibration statistics from training data
  /// In production, this would be called periodically with real data
  void updateCalibration(List<CalibrationSample> samples) {
    // Group samples by confidence bins
    final binData = <double, List<bool>>{};

    for (final sample in samples) {
      final binKey = (sample.predictedConfidence * 10).floor() / 10.0;
      binData.putIfAbsent(binKey, () => []).add(sample.wasCorrect);
    }

    // Update calibration bins
    _calibrationBins.clear();

    for (final entry in binData.entries) {
      final accuracy = entry.value.where((b) => b).length / entry.value.length;
      _calibrationBins.add(CalibrationBin(
        entry.key,
        entry.key + 0.1,
        accuracy,
      ));
    }

    // Sort bins by min score
    _calibrationBins.sort((a, b) => a.minScore.compareTo(b.minScore));
  }
}

/// Represents a calibration bin for isotonic regression
class CalibrationBin {
  final double minScore;
  final double maxScore;
  final double calibratedValue;

  CalibrationBin(this.minScore, this.maxScore, this.calibratedValue);
}

/// Sample for calibration training
class CalibrationSample {
  final double predictedConfidence;
  final bool wasCorrect;

  CalibrationSample(this.predictedConfidence, this.wasCorrect);
}
