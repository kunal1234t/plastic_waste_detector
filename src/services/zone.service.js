// File: src/services/zone.service.js
import { getAllZones, getZoneById } from '../models/zone.model.js';

export const fetchAllZones = async () => {
    return await getAllZones();
};

export const fetchZoneDetails = async (id) => {
    const zone = await getZoneById(id);
    if (!zone) throw { status: 404, message: 'Zone not found' };
    return zone;
};
