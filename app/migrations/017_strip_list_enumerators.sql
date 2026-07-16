-- Removes numbered-list enumerator corridors from the already-learned model.
--
-- Multi-line numbered lists used to survive sanitize_text: whitespace
-- normalization folded "…\n5. хоссейн" into a bare "5 ." token pair, so the
-- chain learned "5. хоссейн"-style starts and mid-walk fragments (backlog
-- "чистка корпуса", OPEN.md). Ingestion now strips line-leading markers
-- (remove_list_markers, app/core/text.py); this one-off cleans what past
-- messages already taught.
--
-- The enumerator shape is <digits-only> '.' <non-digits-next>. The non-digit
-- guard on the following token keeps decimals ("0 . 5" from "0.5") intact;
-- pairs at the end of a row (no next token visible) are left alone for the
-- same reason. Legit prose of the same shape ("приду в 5. завтра…") is
-- indistinguishable and is deleted with the garbage — a rare corridor traded
-- for never opening a reply with "5.".
--
-- GLOB notes: <digits-only> is `NOT GLOB '*[^0-9]*'` (no non-digit chars),
-- <has-a-non-digit> is `GLOB '*[^0-9]*'`; tokens are never empty strings.
--
-- messages.normalized_text is left as-is: newlines are already collapsed
-- there, so a line-leading marker cannot be told apart from an inline "в 5.",
-- and retention ages the window out anyway.

-- Order-3 starts: replies must not open with "5. <word>".
DELETE FROM starts3
WHERE w2 = '.'
  AND w1 NOT GLOB '*[^0-9]*'
  AND w3 GLOB '*[^0-9]*';

-- Order-2 starts see only two tokens, so decimal starts ("0.5 уже выпил")
-- look identical to enumerators. Keep the pair when starts3 holds decimal
-- evidence (same start continuing into a digits-only third token).
DELETE FROM starts
WHERE w2 = '.'
  AND w1 NOT GLOB '*[^0-9]*'
  AND NOT EXISTS (
    SELECT 1 FROM starts3 s3
    WHERE s3.chat_id = starts.chat_id
      AND s3.w1 = starts.w1
      AND s3.w2 = '.'
      AND s3.w3 NOT GLOB '*[^0-9]*'
  );

DELETE FROM transitions
WHERE w2 = '.'
  AND w1 NOT GLOB '*[^0-9]*'
  AND w3 GLOB '*[^0-9]*';

DELETE FROM transitions3
WHERE (w2 = '.' AND w1 NOT GLOB '*[^0-9]*' AND w3 GLOB '*[^0-9]*')
   OR (w3 = '.' AND w2 NOT GLOB '*[^0-9]*' AND w4 GLOB '*[^0-9]*');

DELETE FROM chat_hot_ngrams
WHERE w2 = '.'
  AND w1 NOT GLOB '*[^0-9]*'
  AND w3 GLOB '*[^0-9]*';
