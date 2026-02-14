// File: src/controllers/detection.controller.js
import * as detectionService from '../services/detection.service.js';

export const createDetection = async (req, res, next) => {
    try {
        const { zoneId, plasticType, confidence, timestamp } = req.body;

        if (!zoneId || !plasticType || confidence === undefined || !timestamp) {
            return res.status(400).json({ error: 'Missing required detection fields' });
        }

        const result = await detectionService.ingestDetection({
            zoneId,
            plasticType,
            confidence,
            timestamp
        });

        res.status(201).json({
            message: 'Detection processed successfully',
            data: result
        });
    } catch (err) {
        next(err);
    }
};

export const getDetections = async (req, res, next) => {
    try {
        // Implementation for retrieving detections if needed
        res.status(501).json({ message: 'Not implemented' });
    } catch (err) {
        next(err);
    }
};
