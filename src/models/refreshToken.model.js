// File: src/models/refreshToken.model.js
import pool from '../config/db.js';

export const saveRefreshToken = async (userId, token, expiresAt) => {
    const query = `
        INSERT INTO refresh_tokens (user_id, token, expires_at)
        VALUES (?, ?, ?)
    `;
    const [result] = await pool.query(query, [userId, token, expiresAt]);

    const [rows] = await pool.query('SELECT * FROM refresh_tokens WHERE id = ?', [result.insertId]);
    return rows[0];
};

export const findRefreshToken = async (token) => {
    const query = 'SELECT * FROM refresh_tokens WHERE token = ?';
    const [rows] = await pool.query(query, [token]);
    return rows[0];
};

export const deleteRefreshToken = async (token) => {
    const query = 'DELETE FROM refresh_tokens WHERE token = ?';
    await pool.query(query, [token]);
};

export const deleteUserRefreshTokens = async (userId) => {
    const query = 'DELETE FROM refresh_tokens WHERE user_id = ?';
    await pool.query(query, [userId]);
};
