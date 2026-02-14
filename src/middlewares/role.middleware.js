// File: src/middlewares/role.middleware.js

export const requireRole = (role) => {
    return (req, res, next) => {
        if (!req.user || req.user.role !== role) {
            return res.status(403).json({ error: `Access denied. Requires ${role} role.` });
        }
        next();
    };
};
