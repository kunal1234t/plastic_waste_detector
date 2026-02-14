// File: src/models/detection.model.js
import pool from '../config/db.js';

export const createDetection = async (zoneId, plasticType, confidence, timestamp) => {
    const query = `
        INSERT INTO detections (zone_id, plastic_type, confidence, detected_at)
        VALUES (?, ?, ?, ?)
    `;
    const [result] = await pool.query(query, [zoneId, plasticType, confidence, timestamp]);

    // Fetch created detection
    const [rows] = await pool.query('SELECT * FROM detections WHERE id = ?', [result.insertId]);
    return rows[0];
};

export const getRecentDetections = async (limit = 50) => {
    const query = `
        SELECT d.*, z.name as zone_name
        FROM detections d
        JOIN zones z ON d.zone_id = z.id
        ORDER BY d.detected_at DESC
        LIMIT ?
    `;
    // MySQL limit requires integer
    const [rows] = await pool.query(query, [parseInt(limit)]);
    return rows;
};

export const getDetectionsByZone = async (zoneId, limit = 20) => {
    const query = `
        SELECT * FROM detections
        WHERE zone_id = ?
        ORDER BY detected_at DESC
        LIMIT ?
    `;
    const [rows] = await pool.query(query, [zoneId, parseInt(limit)]);
    return rows;
};
