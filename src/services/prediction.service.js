// File: src/services/prediction.service.js
import { createPrediction, getAllPredictions } from '../models/prediction.model.js';
import { getAllZones } from '../models/zone.model.js';
import { predictFutureRisk } from '../utils/riskCalculator.js';

export const generatePredictions = async () => {
    // Generates predictions for all zones based on current metrics
    const zones = await getAllZones();
    const predictions = [];

    for (const zone of zones) {
        const currentRisk = parseFloat(zone.risk_score || 0);
        const predictedRisk = predictFutureRisk(currentRisk);

        // Simple heuristic for confidence
        const confidence = 0.7 + (Math.random() * 0.2); // 0.7 - 0.9

        const prediction = await createPrediction(
            zone.id,
            '24h',
            predictedRisk,
            confidence
        );
        predictions.push(prediction);
    }
    return predictions;
};

export const fetchPredictions = async () => {
    // If no predictions exist, generate some? Or just return existing.
    // For demo, let's return existing.
    return await getAllPredictions();
};
