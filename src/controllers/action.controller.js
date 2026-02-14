// File: src/controllers/action.controller.js
import * as actionService from '../services/action.service.js';

export const dispatchAction = async (req, res, next) => {
    try {
        const { zoneId, actionType, notes } = req.body;

        if (!zoneId || !actionType) {
            return res.status(400).json({ error: 'Zone ID and Action Type are required' });
        }

        const action = await actionService.dispatchAction(zoneId, actionType, notes);
        res.status(201).json({
            message: 'Action dispatched successfully',
            data: action
        });
    } catch (err) {
        next(err);
    }
};

export const getActions = async (req, res, next) => {
    try {
        const actions = await actionService.listActions();
        res.status(200).json(actions);
    } catch (err) {
        next(err);
    }
};
