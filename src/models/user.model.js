// File: src/models/user.model.js
import pool from '../config/db.js';

export const createUser = async (username, email, passwordHash, role = 'viewer') => {
    const query = `
        INSERT INTO users (username, email, password_hash, role)
        VALUES (?, ?, ?, ?)
    `;
    const [result] = await pool.query(query, [username, email, passwordHash, role]);

    // Fetch the created user to return full object
    const [rows] = await pool.query('SELECT id, username, email, role, created_at FROM users WHERE id = ?', [result.insertId]);
    return rows[0];
};

export const findUserByEmail = async (email) => {
    const query = 'SELECT * FROM users WHERE email = ?';
    const [rows] = await pool.query(query, [email]);
    return rows[0];
};

export const findUserById = async (id) => {
    const query = 'SELECT id, username, email, role, created_at FROM users WHERE id = ?';
    const [rows] = await pool.query(query, [id]);
    return rows[0];
};
