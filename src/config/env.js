// File: src/config/env.js
import dotenv from 'dotenv';
dotenv.config();

export const PORT = process.env.PORT || 5000;

export const DB_CONFIG = {
    user: process.env.DB_USER || 'root',
    host: process.env.DB_HOST || 'localhost',
    database: process.env.DB_NAME || 'plastic_waste',
    password: process.env.DB_PASSWORD || 'iiitnBX248',
    port: process.env.DB_PORT || 3306,
};

export const JWT_CONFIG = {
    ACCESS_SECRET: process.env.ACCESS_TOKEN_SECRET || 'access_secret_123',
    REFRESH_SECRET: process.env.REFRESH_TOKEN_SECRET || 'refresh_secret_456',
    ACCESS_EXPIRES: '15m',
    REFRESH_EXPIRES: '7d',
};
