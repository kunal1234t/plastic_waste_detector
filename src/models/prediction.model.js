// File: src/models/prediction.model.js
import pool from '../config/db.js';

export const createPrediction = async (zoneId, window, expectedRisk, confidence) => {
    const query = `
        INSERT INTO predictions (zone_id, prediction_window, expected_risk, confidence)
        VALUES (?, ?, ?, ?)
    `;
    const [result] = await pool.query(query, [zoneId, window, expectedRisk, confidence]);

    // Fetch inserted prediction
    const [rows] = await pool.query('SELECT * FROM predictions WHERE id = ?', [result.insertId]);
    return rows[0];
};

export const getPredictionsByZone = async (zoneId) => {
    const query = `
        SELECT * FROM predictions
        WHERE zone_id = ?
        ORDER BY created_at DESC
        LIMIT 5
    `;
    const [rows] = await pool.query(query, [zoneId]);
    return rows;
};

export const getAllPredictions = async () => {
    const query = `
        SELECT p.*, z.name as zone_name
        FROM predictions p
        JOIN zones z ON p.zone_id = z.id
        ORDER BY p.expected_risk DESC, p.created_at DESC
        LIMIT 50
    `;
    const [rows] = await pool.query(query);
    return rows;
};
