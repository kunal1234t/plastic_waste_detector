// File: src/services/action.service.js
import { createAction, getActions, updateActionStatus } from '../models/action.model.js';

export const dispatchAction = async (zoneId, actionType, notes) => {
    return await createAction(zoneId, actionType, notes);
};

export const listActions = async () => {
    return await getActions();
};

export const completeAction = async (id) => {
    return await updateActionStatus(id, 'COMPLETED');
};
