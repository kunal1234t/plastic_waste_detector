// File: src/controllers/zone.controller.js
import * as zoneService from '../services/zone.service.js';

export const getAllZones = async (req, res, next) => {
    try {
        const zones = await zoneService.fetchAllZones();
        res.status(200).json(zones);
    } catch (err) {
        next(err);
    }
};

export const getZoneById = async (req, res, next) => {
    try {
        const { id } = req.params;
        const zone = await zoneService.fetchZoneDetails(id);
        res.status(200).json(zone);
    } catch (err) {
        next(err);
    }
};
