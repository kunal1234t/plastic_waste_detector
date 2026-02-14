// File: src/utils/riskCalculator.js

// Weights for different plastic types (0-100 scale impact)
const PLASTIC_WEIGHTS = {
    'plastic_bottle': 10,
    'plastic_bag': 15,
    'wrapper': 5,
    'container': 12,
    'unknown': 8
};

/**
 * Calculates risk score based on detections.
 * @param {Array} detections - List of recent detections
 * @returns {Number} - Calculated risk score (0-100)
 */
export const calculateRiskScore = (detections) => {
    if (!detections || detections.length === 0) return 0;

    let rawScore = 0;

    // Simple algorithm: Sum of weights * confidence
    detections.forEach(d => {
        const weight = PLASTIC_WEIGHTS[d.plastic_type] || 5;
        // normalized confidence treated as multiplier
        rawScore += weight * (d.confidence || 0.5);
    });

    // Normalize to 0-100 (assuming 20 items is max saturation for a zone)
    // This is a heuristic for the hackathon
    const normalizedScore = Math.min(100, Math.max(0, rawScore));

    return parseFloat(normalizedScore.toFixed(2));
};

/**
 * Simulates a prediction based on current metrics
 */
export const predictFutureRisk = (currentScore) => {
    // Simple trend simulation: fluctuating around current score + trend
    const variance = (Math.random() * 10) - 5; // +/- 5
    const trend = (Math.random() * 5); // Slight increase over time if not verified

    let simulated = currentScore + variance + trend;
    return Math.min(100, Math.max(0, parseFloat(simulated.toFixed(2))));
};
