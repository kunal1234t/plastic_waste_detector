// File: src/utils/token.js
import jwt from 'jsonwebtoken';
import { JWT_CONFIG } from '../config/env.js';

export const generateAccessToken = (userId, role) => {
    return jwt.sign({ userId, role }, JWT_CONFIG.ACCESS_SECRET, {
        expiresIn: JWT_CONFIG.ACCESS_EXPIRES,
    });
};

export const generateRefreshToken = (userId) => {
    return jwt.sign({ userId }, JWT_CONFIG.REFRESH_SECRET, {
        expiresIn: JWT_CONFIG.REFRESH_EXPIRES,
    });
};

export const verifyAccessToken = (token) => {
    return jwt.verify(token, JWT_CONFIG.ACCESS_SECRET);
};

export const verifyRefreshToken = (token) => {
    return jwt.verify(token, JWT_CONFIG.REFRESH_SECRET);
};
