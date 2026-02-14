// File: src/services/auth.service.js
import bcrypt from 'bcrypt';
import { createUser, findUserByEmail, findUserById } from '../models/user.model.js';
import { generateAccessToken, generateRefreshToken, verifyRefreshToken } from '../utils/token.js';
import { saveRefreshToken, findRefreshToken, deleteRefreshToken } from '../models/refreshToken.model.js';

export const register = async (username, email, password, role) => {
    const existingUser = await findUserByEmail(email);
    if (existingUser) {
        throw { status: 409, message: 'Email already exists' };
    }

    const salt = await bcrypt.genSalt(10);
    const passwordHash = await bcrypt.hash(password, salt);

    const user = await createUser(username, email, passwordHash, role);
    return user;
};

export const login = async (email, password) => {
    const user = await findUserByEmail(email);
    if (!user) {
        throw { status: 401, message: 'Invalid credentials' };
    }

    const isMatch = await bcrypt.compare(password, user.password_hash);
    if (!isMatch) {
        throw { status: 401, message: 'Invalid credentials' };
    }

    const accessToken = generateAccessToken(user.id, user.role);
    const refreshToken = generateRefreshToken(user.id);

    // Save refresh token with 7 days expiry
    const expiresAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000);
    await saveRefreshToken(user.id, refreshToken, expiresAt);

    return {
        user: { id: user.id, username: user.username, email: user.email, role: user.role },
        accessToken,
        refreshToken
    };
};

export const refreshToken = async (token) => {
    if (!token) throw { status: 401, message: 'Token required' };

    const storedToken = await findRefreshToken(token);
    if (!storedToken) throw { status: 403, message: 'Invalid refresh token' };

    try {
        const decoded = verifyRefreshToken(token);
        // Additional check if user exists or is banned could go here

        const accessToken = generateAccessToken(decoded.userId, 'viewer'); // Retrieve role if possible or store in refresh token
        // For simplicity, we might need to fetch user to get role again or encode it in refresh token 
        // Let's fetch user to be safe
        const user = await findUserById(decoded.userId);
        const newAccessToken = generateAccessToken(user.id, user.role);

        return { accessToken: newAccessToken };
    } catch (err) {
        throw { status: 403, message: 'Expired or invalid refresh token' };
    }
};

export const logout = async (token) => {
    await deleteRefreshToken(token);
};
