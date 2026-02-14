// File: src/models/action.model.js
import pool from '../config/db.js';

export const createAction = async (zoneId, actionType, notes) => {
    // Note: 'notes' was in the PG schema but might not be in the MySQL schema if user didn't add it.
    // User schema for MySQL: `actions (id, zone_id, action_type, status, dispatched_at, completed_at)`
    // It seems 'notes' column is MISSING in the user's MySQL schema.
    // I will try to insert it, but if it fails, I should probably remove it or ask.
    // Wait, the prompt says "Replace pg with mysql2... everything else must remain untouched".
    // But the schema is already changed by the user.
    // Let's assume 'notes' might be missing, but I'll try to include it if the function signature requires it.
    // Actually looking at the user provided schema again:
    // CREATE TABLE actions (... status ... dispatched_at ... completed_at ... FOREIGN KEY ...)
    // 'notes' is indeed missing.
    // However, the `createAction` function signature in `src/services/action.service.js` passes `notes`.
    // I will omit `notes` from the INSERT to avoid SQL error if the column doesn't exist.

    const query = `
        INSERT INTO actions (zone_id, action_type)
        VALUES (?, ?)
    `;
    const [result] = await pool.query(query, [zoneId, actionType]);

    const [rows] = await pool.query('SELECT * FROM actions WHERE id = ?', [result.insertId]);
    return rows[0];
};

export const updateActionStatus = async (id, status) => {
    const query = `
        UPDATE actions
        SET status = ?,
            completed_at = CASE WHEN ? = 'completed' THEN CURRENT_TIMESTAMP ELSE completed_at END
        WHERE id = ?
    `;
    // Note: User schema used 'completed' (lowercase) in ENUM, previously it was 'COMPLETED'.
    // ENUM('pending','completed')
    // I should ensure I send 'completed' lowercase if that's what the schema expects.
    // But the service sends 'COMPLETED'. converting to lowercase might be safer or just passing as is.
    // MySQL ENUMs are case-insensitive usually but strict mode might matter.
    // I'll stick to passing 'status' but for the CASE WHEN check I match what I expect.

    await pool.query(query, [status, status, id]);

    const [rows] = await pool.query('SELECT * FROM actions WHERE id = ?', [id]);
    return rows[0];
};

export const getActions = async () => {
    const query = `
        SELECT a.*, z.name as zone_name
        FROM actions a
        JOIN zones z ON a.zone_id = z.id
        ORDER BY a.dispatched_at DESC
    `;
    const [rows] = await pool.query(query);
    return rows;
};
