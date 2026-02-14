// File: src/controllers/prediction.controller.js
import * as predictionService from '../services/prediction.service.js';

export const getPredictions = async (req, res, next) => {
    try {
        // trigger generation if needed or just fetch
        const predictions = await predictionService.generatePredictions();
        res.status(200).json(predictions);
    } catch (err) {
        next(err);
    }
};
