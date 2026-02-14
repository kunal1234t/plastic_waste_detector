// File: src/middlewares/error.middleware.js

export const errorHandler = (err, req, res, next) => {
    console.error(err.stack);

    const status = err.status || 500;
    const message = err.message || 'Internal Server Error';

    res.status(status).json({
        error: {
            message: message,
            status: status
        }
    });
};
