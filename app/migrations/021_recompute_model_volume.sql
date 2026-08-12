-- Recomputes chat_model_volume from the actual transition counts.
--
-- Migration 017 deleted enumerator rows from transitions/transitions3 without
-- decrementing the incremental volume counters, breaking the invariant
-- volume2/volume3 = SUM(cnt) (see MarkovRepo.get_model_volume). The counters
-- gate readiness to reply, so affected chats look "fuller" than they are.
-- One-off resync; the write path keeps the counters exact from here on.
--
-- The migration runner wraps this script in BEGIN/COMMIT; do not add them here.

UPDATE chat_model_volume SET
    volume2 = COALESCE(
        (SELECT SUM(cnt) FROM transitions
         WHERE transitions.chat_id = chat_model_volume.chat_id), 0),
    volume3 = COALESCE(
        (SELECT SUM(cnt) FROM transitions3
         WHERE transitions3.chat_id = chat_model_volume.chat_id), 0);
