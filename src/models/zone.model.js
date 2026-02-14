// File: src/models/zone.model.js
import pool from '../config/db.js';

export const createZone = async (id, name, description, lat, lng) => {
    const query = `
        INSERT INTO zones (id, name, description, latitude, longitude)
        VALUES (?, ?, ?, ?, ?)
    `;
    await pool.query(query, [id, name, description, lat, lng]);

    // Initialize metrics
    await pool.query('INSERT INTO zone_metrics (zone_id) VALUES (?)', [id]);

    // Return created zone
    const [rows] = await pool.query('SELECT * FROM zones WHERE id = ?', [id]);
    return rows[0];
};

export const getAllZones = async () => {
    const query = `
        SELECT z.*, zm.risk_score, zm.total_detections, zm.last_updated
        FROM zones z
        LEFT JOIN zone_metrics zm ON z.id = zm.zone_id
        ORDER BY zm.risk_score DESC
    `;
    const [rows] = await pool.query(query);
    return rows;
};

export const getZoneById = async (id) => {
    const query = `
        SELECT z.*, zm.risk_score, zm.total_detections, zm.last_updated
        FROM zones z
        LEFT JOIN zone_metrics zm ON z.id = zm.zone_id
        WHERE z.id = ?
    `;
    const [rows] = await pool.query(query, [id]);
    return rows[0];
};

export const updateZoneMetrics = async (zoneId, riskScore) => {
    const query = `
        UPDATE zone_metrics
        SET risk_score = ?, 
            total_detections = total_detections + 1,
            last_updated = CURRENT_TIMESTAMP
        WHERE zone_id = ?
    `;
    await pool.query(query, [riskScore, zoneId]);

    // Return updated metrics
    const [rows] = await pool.query('SELECT * FROM zone_metrics WHERE zone_id = ?', [zoneId]);
    return rows[0];
};
