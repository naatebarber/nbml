use ndarray::Array2;

pub fn roc_auc_score(y_true: Array2<f32>, y_scores: Array2<f32>) -> f32 {
    assert_eq!(
        y_true.len(),
        y_scores.len(),
        "y_true and y_scores must have the same length"
    );

    let mut pairs: Vec<(f32, bool)> = y_scores
        .iter()
        .zip(y_true.iter())
        .map(|(&s, &y)| (s, y > 0.5))
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    let n_pos = pairs.iter().filter(|p| p.1).count();
    let n_neg = pairs.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return f32::NAN; // undefined with a single class
    }

    // Mann-Whitney U, sum of the positives ranks, ties given their average rank
    let mut rank_sum_pos = 0f64;
    let mut i = 0;
    while i < pairs.len() {
        let mut j = i;
        while j < pairs.len() && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        let avg_rank = (i + 1 + j) as f64 / 2.0; // ranks i+1 ..= j
        let n_pos_tied = pairs[i..j].iter().filter(|p| p.1).count();
        rank_sum_pos += avg_rank * n_pos_tied as f64;
        i = j;
    }

    let u = rank_sum_pos - (n_pos * (n_pos + 1)) as f64 / 2.0;
    (u / (n_pos as f64 * n_neg as f64)) as f32
}

pub fn pr_auc_score(y_true: Array2<f32>, y_scores: Array2<f32>) -> f32 {
    assert_eq!(
        y_true.len(),
        y_scores.len(),
        "y_true and y_scores must have the same length"
    );

    let mut pairs: Vec<(f32, bool)> = y_scores
        .iter()
        .zip(y_true.iter())
        .map(|(&s, &y)| (s, y > 0.5))
        .collect();

    // descending, walk thresholds from strictest to loosest
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));

    let n_pos = pairs.iter().filter(|p| p.1).count();
    if n_pos == 0 {
        return f32::NAN; // undefined without positives
    }

    // average precision, sum over thresholds of (Δrecall × precision).
    // tied scores form one threshold, so they enter together
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut ap = 0f64;
    let mut i = 0;
    while i < pairs.len() {
        let mut j = i;
        while j < pairs.len() && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        let n_pos_tied = pairs[i..j].iter().filter(|p| p.1).count();
        tp += n_pos_tied;
        fp += (j - i) - n_pos_tied;
        if n_pos_tied > 0 {
            let precision = tp as f64 / (tp + fp) as f64;
            let delta_recall = n_pos_tied as f64 / n_pos as f64;
            ap += delta_recall * precision;
        }
        i = j;
    }

    ap as f32
}
