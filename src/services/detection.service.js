// File: src/services/detection.service.js
import { createDetection, getDetectionsByZone } from '../models/detection.model.js';
import { updateZoneMetrics, getZoneById, createZone } from '../models/zone.model.js';
import { calculateRiskScore } from '../utils/riskCalculator.js';

export const ingestDetection = async (data) => {
    // 1. Ensure Zone Exists (Auto-create for hackathon simplicity if needed, or error)
    // For strictness, we should probably check if zone exists. 
    // Data: { zoneId, plasticType, confidence, timestamp }

    let zone = await getZoneById(data.zoneId);
    if (!zone) {
        // Auto-create zone for demo purposes if it doesn't exist
        zone = await createZone(data.zoneId, `Zone ${data.zoneId}`, 'Auto-generated zone', 0, 0);
    }

    // 2. Save Detection
    const detection = await createDetection(
        data.zoneId,
        data.plasticType,
        data.confidence,
        data.timestamp
    );

    // 3. Recalculate Risk Score for Zone
    // Fetch recent detections to calculate moving average or cumulative risk
    const recentDetections = await getDetectionsByZone(data.zoneId, 20);
    const newRiskScore = calculateRiskScore(recentDetections);

    // 4. Update Zone Metrics
    await updateZoneMetrics(data.zoneId, newRiskScore);

    return { detection, riskScore: newRiskScore };
};

export const getDetections = async (criteria) => {
    // Placeholder for more complex filtering
    return [];
};
