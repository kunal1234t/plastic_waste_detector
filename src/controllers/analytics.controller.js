// File: src/controllers/analytics.controller.js
import * as analyticsService from '../services/analytics.service.js';

export const getAnalytics = async (req, res, next) => {
    try {
        const analytics = await analyticsService.getGlobalAnalytics();
        res.status(200).json(analytics);
    } catch (err) {
        next(err);
    }
};
