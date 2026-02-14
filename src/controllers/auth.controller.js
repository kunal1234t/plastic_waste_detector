// File: src/controllers/auth.controller.js
import * as authService from '../services/auth.service.js';

export const register = async (req, res, next) => {
    try {
        const { username, email, password, role } = req.body;
        // Basic validation
        if (!username || !email || !password) {
            return res.status(400).json({ error: 'Username, email, and password are required' });
        }

        const user = await authService.register(username, email, password, role);
        res.status(201).json({ message: 'User registered successfully', userId: user.id });
    } catch (err) {
        next(err);
    }
};

export const login = async (req, res, next) => {
    try {
        const { email, password } = req.body;
        if (!email || !password) {
            return res.status(400).json({ error: 'Email and password are required' });
        }

        const result = await authService.login(email, password);
        res.status(200).json(result);
    } catch (err) {
        next(err);
    }
};

export const refresh = async (req, res, next) => {
    try {
        const { token } = req.body;
        if (!token) {
            return res.status(400).json({ error: 'Refresh token is required' });
        }

        const result = await authService.refreshToken(token);
        res.status(200).json(result);
    } catch (err) {
        next(err);
    }
};

export const logout = async (req, res, next) => {
    try {
        const { token } = req.body;
        await authService.logout(token);
        res.status(204).send();
    } catch (err) {
        next(err);
    }
};
