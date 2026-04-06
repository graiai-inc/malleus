import 'dart:math';

/// Implements the Recognizer Output Voting Error Reduction (ROVER) algorithm
/// for combining multiple ASR hypotheses with confidence weighting
class ROVERFusion {
  // Minimum confidence threshold for considering a word
  static const double minConfidenceThreshold = 0.1;

  // Weight for streaming vs batch model (0.3 = 30% streaming, 70% batch)
  static const double streamingWeight = 0.3;

  /// Main fusion method combining two ASR outputs with confidence scores
  String fuse({
    required String streamingText,
    required String batchText,
    List<double>? streamingConfidences,
    List<double>? batchConfidences,
  }) {
    // Tokenize inputs
    final streamingWords = _tokenize(streamingText);
    final batchWords = _tokenize(batchText);

    // Generate uniform confidences if not provided
    final streamConf =
        streamingConfidences ?? List.filled(streamingWords.length, 0.7);
    final batchConf = batchConfidences ?? List.filled(batchWords.length, 0.9);

    // Build word transition network
    final wtn = _buildWordTransitionNetwork(
      streamingWords,
      batchWords,
      streamConf,
      batchConf,
    );

    // Find best path through network
    final bestPath = _findBestPath(wtn);

    // Post-process for medical context
    return _postProcessMedical(bestPath.join(' '));
  }

  /// Tokenize text into words, preserving medical formatting
  List<String> _tokenize(String text) {
    // Preserve medical abbreviations and numbers
    text = text.replaceAllMapped(
      RegExp(r'(\d+\.?\d*)\s*(mg|mcg|ml|cc|units?|IU|mEq|g|kg|L)'),
      (m) => '${m.group(1)}${m.group(2)}',
    );

    // Split on whitespace but preserve punctuation
    return text.split(RegExp(r'\s+')).where((w) => w.isNotEmpty).toList();
  }

  /// Build Word Transition Network from multiple hypotheses
  List<WordSlot> _buildWordTransitionNetwork(
    List<String> words1,
    List<String> words2,
    List<double> conf1,
    List<double> conf2,
  ) {
    final alignment = _alignSequences(words1, words2);
    final network = <WordSlot>[];

    for (final alignedPair in alignment) {
      final slot = WordSlot();

      if (alignedPair.word1 != null) {
        final confidence = (conf1.length > alignedPair.index1)
            ? conf1[alignedPair.index1] * streamingWeight
            : 0.7 * streamingWeight;
        slot.addCandidate(alignedPair.word1!, confidence);
      }

      if (alignedPair.word2 != null) {
        final confidence = (conf2.length > alignedPair.index2)
            ? conf2[alignedPair.index2] * (1 - streamingWeight)
            : 0.9 * (1 - streamingWeight);
        slot.addCandidate(alignedPair.word2!, confidence);
      }

      if (slot.candidates.isNotEmpty) {
        network.add(slot);
      }
    }

    return network;
  }

  /// Align two word sequences using dynamic programming
  List<AlignedPair> _alignSequences(List<String> seq1, List<String> seq2) {
    final m = seq1.length;
    final n = seq2.length;

    // Initialize DP table
    final dp = List.generate(
      m + 1,
      (_) => List.filled(n + 1, 0.0),
    );

    // Initialize backtrack table
    final backtrack = List.generate(
      m + 1,
      (_) => List.filled(n + 1, ''),
    );

    // Fill DP table
    for (int i = 1; i <= m; i++) {
      dp[i][0] = dp[i - 1][0] - 1; // Deletion cost
      backtrack[i][0] = 'D';
    }

    for (int j = 1; j <= n; j++) {
      dp[0][j] = dp[0][j - 1] - 1; // Insertion cost
      backtrack[0][j] = 'I';
    }

    for (int i = 1; i <= m; i++) {
      for (int j = 1; j <= n; j++) {
        final matchScore = _wordSimilarity(seq1[i - 1], seq2[j - 1]);
        final scores = [
          dp[i - 1][j - 1] + matchScore, // Match/substitute
          dp[i - 1][j] - 1, // Delete from seq1
          dp[i][j - 1] - 1, // Insert from seq2
        ];

        final maxScore = scores.reduce(max);
        dp[i][j] = maxScore;

        if (maxScore == scores[0]) {
          backtrack[i][j] = 'M';
        } else if (maxScore == scores[1]) {
          backtrack[i][j] = 'D';
        } else {
          backtrack[i][j] = 'I';
        }
      }
    }

    // Backtrack to find alignment
    final alignment = <AlignedPair>[];
    int i = m, j = n;

    while (i > 0 || j > 0) {
      if (i == 0) {
        alignment.add(AlignedPair(null, seq2[j - 1], -1, j - 1));
        j--;
      } else if (j == 0) {
        alignment.add(AlignedPair(seq1[i - 1], null, i - 1, -1));
        i--;
      } else {
        final op = backtrack[i][j];
        if (op == 'M') {
          alignment.add(AlignedPair(seq1[i - 1], seq2[j - 1], i - 1, j - 1));
          i--;
          j--;
        } else if (op == 'D') {
          alignment.add(AlignedPair(seq1[i - 1], null, i - 1, -1));
          i--;
        } else {
          alignment.add(AlignedPair(null, seq2[j - 1], -1, j - 1));
          j--;
        }
      }
    }

    return alignment.reversed.toList();
  }

  /// Calculate similarity between two words (for medical terms)
  double _wordSimilarity(String word1, String word2) {
    if (word1 == word2) return 2.0;

    // Handle medical abbreviations
    if (_areMedicalSynonyms(word1, word2)) return 1.8;

    // Handle numbers with units
    if (_areEquivalentDosages(word1, word2)) return 1.9;

    // Levenshtein distance normalized
    final distance = _levenshteinDistance(word1, word2);
    final maxLen = max(word1.length, word2.length);
    final similarity = 1.0 - (distance / maxLen);

    return similarity > 0.7 ? similarity : -1.0;
  }

  /// Check if two words are medical synonyms
  bool _areMedicalSynonyms(String word1, String word2) {
    final synonyms = {
      'mg': ['milligram', 'milligrams'],
      'mcg': ['microgram', 'micrograms'],
      'ml': ['milliliter', 'milliliters', 'cc'],
      'bid': ['twice', 'daily', 'b.i.d.'],
      'tid': ['three', 'times', 't.i.d.'],
      'qid': ['four', 'times', 'q.i.d.'],
      'prn': ['as', 'needed', 'p.r.n.'],
      'po': ['by', 'mouth', 'p.o.'],
      'iv': ['intravenous', 'i.v.'],
      'im': ['intramuscular', 'i.m.'],
    };

    for (final entry in synonyms.entries) {
      if ((word1 == entry.key && entry.value.contains(word2)) ||
          (word2 == entry.key && entry.value.contains(word1))) {
        return true;
      }
    }

    return false;
  }

  /// Check if two dosage expressions are equivalent
  bool _areEquivalentDosages(String word1, String word2) {
    final dosagePattern =
        RegExp(r'^(\d+\.?\d*)(mg|mcg|ml|cc|units?|IU|mEq|g|kg|L)$');
    final match1 = dosagePattern.firstMatch(word1);
    final match2 = dosagePattern.firstMatch(word2);

    if (match1 != null && match2 != null) {
      final num1 = double.tryParse(match1.group(1)!);
      final num2 = double.tryParse(match2.group(1)!);
      final unit1 = match1.group(2);
      final unit2 = match2.group(2);

      if (num1 == num2 && unit1 == unit2) return true;

      // Handle unit conversions
      if (unit1 == 'g' && unit2 == 'mg' && num1 != null && num2 != null) {
        return (num1 * 1000 - num2).abs() < 0.01;
      }
      if (unit1 == 'mg' && unit2 == 'mcg' && num1 != null && num2 != null) {
        return (num1 * 1000 - num2).abs() < 0.01;
      }
    }

    return false;
  }

  /// Calculate Levenshtein distance between two strings
  int _levenshteinDistance(String s1, String s2) {
    final m = s1.length;
    final n = s2.length;
    final dp = List.generate(m + 1, (_) => List.filled(n + 1, 0));

    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
      for (int j = 1; j <= n; j++) {
        if (s1[i - 1] == s2[j - 1]) {
          dp[i][j] = dp[i - 1][j - 1];
        } else {
          dp[i][j] = 1 +
              [
                dp[i - 1][j], // deletion
                dp[i][j - 1], // insertion
                dp[i - 1][j - 1] // substitution
              ].reduce((a, b) => a < b ? a : b);
        }
      }
    }

    return dp[m][n];
  }

  /// Find best path through Word Transition Network
  List<String> _findBestPath(List<WordSlot> network) {
    final bestPath = <String>[];

    for (final slot in network) {
      final bestWord = slot.getBestWord();
      if (bestWord != null && bestWord.isNotEmpty) {
        bestPath.add(bestWord);
      }
    }

    return bestPath;
  }

  /// Post-process for medical context
  String _postProcessMedical(String text) {
    // Fix dosage formatting
    text = text.replaceAllMapped(
      RegExp(r'(\d+\.?\d*)\s*(mg|mcg|ml|cc|units?|IU|mEq|g|kg|L)'),
      (m) => '${m.group(1)} ${m.group(2)}',
    );

    // Ensure leading zeros for decimals
    text = text.replaceAllMapped(
      RegExp(r'\b\.(\d+)\s*(mg|mcg|ml|cc|units?|IU|mEq|g|kg|L)'),
      (m) => '0.${m.group(1)} ${m.group(2)}',
    );

    // Capitalize drug names (simplified - should use drug database)
    text = text.replaceAllMapped(
      RegExp(
          r'\b(aspirin|acetaminophen|ibuprofen|metformin|lisinopril|atorvastatin|levothyroxine|amlodipine|metoprolol|omeprazole)\b',
          caseSensitive: false),
      (m) =>
          m.group(0)!.substring(0, 1).toUpperCase() +
          m.group(0)!.substring(1).toLowerCase(),
    );

    // Expand critical abbreviations
    final abbreviations = {
      r'\bb\.?i\.?d\.?\b': 'twice daily',
      r'\bt\.?i\.?d\.?\b': 'three times daily',
      r'\bq\.?i\.?d\.?\b': 'four times daily',
      r'\bp\.?r\.?n\.?\b': 'as needed',
      r'\bp\.?o\.?\b': 'by mouth',
      r'\bi\.?v\.?\b': 'intravenously',
      r'\bi\.?m\.?\b': 'intramuscularly',
      r'\bs\.?c\.?\b': 'subcutaneously',
    };

    for (final entry in abbreviations.entries) {
      text =
          text.replaceAll(RegExp(entry.key, caseSensitive: false), entry.value);
    }

    return text.trim();
  }
}

/// Represents a slot in the Word Transition Network
class WordSlot {
  final Map<String, double> candidates = {};

  void addCandidate(String word, double confidence) {
    if (candidates.containsKey(word)) {
      candidates[word] = candidates[word]! + confidence;
    } else {
      candidates[word] = confidence;
    }
  }

  String? getBestWord() {
    if (candidates.isEmpty) return null;

    double maxConfidence = -1;
    String? bestWord;

    for (final entry in candidates.entries) {
      if (entry.value > maxConfidence) {
        maxConfidence = entry.value;
        bestWord = entry.key;
      }
    }

    return bestWord;
  }
}

/// Represents an aligned pair of words from two sequences
class AlignedPair {
  final String? word1;
  final String? word2;
  final int index1;
  final int index2;

  AlignedPair(this.word1, this.word2, this.index1, this.index2);
}
