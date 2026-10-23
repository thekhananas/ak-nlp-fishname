const examples = {
  saviour: {
    prefix: 'Community: poor families',
    gold: 'Gold Label: PCL (1)',
    text: '“<mark>These poor people</mark> have so little, yet <mark>they still manage to smile.</mark>”',
    explanation:
      'Pity frames the community as helpless objects of sympathy. Surface bag-of-words misses the condescension; self-attention captures the saviour tone.',
    tfidfWidth: '34%',
    tfidfVerdict: '0.34 • Missed (FN)',
    robertaWidth: '89%',
    robertaVerdict: '0.89 • Caught (TP)',
  },
  agency: {
    prefix: 'Community: vulnerable',
    gold: 'Gold Label: No-PCL (0)',
    text: '“Residents organised a community fund to support families affected by the floods.”',
    explanation:
      'Factual reporting preserves community agency. Keyword overlap ("support families") triggers a TF-IDF false alarm, while RoBERTa-large stays below τ = 0.77.',
    tfidfWidth: '82%',
    tfidfVerdict: '0.82 • False Alarm',
    robertaWidth: '14%',
    robertaVerdict: '0.14 • Correct (TN)',
  },
  crisis: {
    prefix: 'Community: refugees',
    gold: 'Gold Label: No-PCL (0)',
    text: '“Winter shortages across border camps have left thousands of displaced families without heating.”',
    explanation:
      'Urgent humanitarian reporting describes hardship objectively without superior or pitying framing — correctly rejected by the calibrated τ = 0.77 boundary.',
    tfidfWidth: '61%',
    tfidfVerdict: '0.61 • Correct (TN)',
    robertaWidth: '21%',
    robertaVerdict: '0.21 • Correct (TN)',
  },
};

document.querySelectorAll('[data-example]').forEach((button) => {
  button.addEventListener('click', () => {
    const example = examples[button.dataset.example];
    document.querySelectorAll('[data-example]').forEach((item) => {
      item.setAttribute('aria-pressed', String(item === button));
    });
    document.getElementById('example-prefix').textContent = example.prefix;
    document.getElementById('example-gold').textContent = example.gold;
    document.getElementById('example-text').innerHTML = example.text;
    document.getElementById('example-explanation').textContent = example.explanation;
    document.getElementById('tfidf-bar').style.width = example.tfidfWidth;
    document.getElementById('tfidf-verdict').textContent = example.tfidfVerdict;
    document.getElementById('roberta-bar').style.width = example.robertaWidth;
    document.getElementById('roberta-verdict').textContent = example.robertaVerdict;
  });
});
