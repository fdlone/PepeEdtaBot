-- Fixture: реальная схема БД до введения миграций.
--
-- Воспроизводит точную структуру markov.db, полученную из рабочего проекта:
--   - messages БЕЗ колонки normalized_text
--   - НЕТ таблицы pivo_chat_members
--   - индекс idx_starts_chat_id (отличается от нашего 001_initial.sql)
--   - все остальные таблицы и индексы — реальные
--
-- Данные синтетические (реальный контент сообщений не хранится).
-- Используется в TestMigratorRealSchemaFixture.

CREATE TABLE messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    INTEGER NOT NULL,
    author_id  INTEGER NOT NULL,
    text       TEXT    NOT NULL,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE starts (
    chat_id INTEGER NOT NULL,
    w1      TEXT    NOT NULL,
    w2      TEXT    NOT NULL,
    cnt     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(chat_id, w1, w2)
);

CREATE TABLE starts3 (
    chat_id INTEGER NOT NULL,
    w1      TEXT    NOT NULL,
    w2      TEXT    NOT NULL,
    w3      TEXT    NOT NULL,
    cnt     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(chat_id, w1, w2, w3)
);

CREATE TABLE transitions (
    chat_id INTEGER NOT NULL,
    w1      TEXT    NOT NULL,
    w2      TEXT    NOT NULL,
    w3      TEXT    NOT NULL,
    cnt     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(chat_id, w1, w2, w3)
);

CREATE TABLE transitions1 (
    chat_id INTEGER NOT NULL,
    w1      TEXT    NOT NULL,
    w2      TEXT    NOT NULL,
    cnt     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(chat_id, w1, w2)
);

CREATE TABLE transitions3 (
    chat_id INTEGER NOT NULL,
    w1      TEXT    NOT NULL,
    w2      TEXT    NOT NULL,
    w3      TEXT    NOT NULL,
    w4      TEXT    NOT NULL,
    cnt     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(chat_id, w1, w2, w3, w4)
);

-- Индексы в точности как в реальной БД (idx_starts_chat_id вместо отсутствующего)
CREATE INDEX idx_messages_chat_id    ON messages(chat_id);
CREATE INDEX idx_starts_chat_id      ON starts(chat_id);
CREATE INDEX idx_starts_lookup       ON starts(chat_id, w1, w2);
CREATE INDEX idx_starts3_chat_id     ON starts3(chat_id);
CREATE INDEX idx_transitions_lookup  ON transitions(chat_id, w1, w2);
CREATE INDEX idx_transitions1_lookup ON transitions1(chat_id, w1);
CREATE INDEX idx_transitions3_lookup ON transitions3(chat_id, w1, w2, w3);

-- Данные: 5 сообщений с реальными author_id (ещё не анонимизированы)
INSERT INTO messages(chat_id, author_id, text) VALUES
    (-1009876543210, 396273827, 'привет всем'),
    (-1009876543210, 249369775, 'как дела'),
    (-1009876543210, 396273827, 'всё хорошо'),
    (-1009876543210, 249369775, 'пойдём пить кофе'),
    (-1001234567890, 111222333, 'другой чат тоже здесь');

-- Данные Маркова n=2
INSERT INTO starts(chat_id, w1, w2, cnt) VALUES
    (-1009876543210, 'привет',  'всем',   1),
    (-1009876543210, 'как',     'дела',   1),
    (-1009876543210, 'всё',     'хорошо', 1),
    (-1009876543210, 'пойдём',  'пить',   1);

INSERT INTO transitions(chat_id, w1, w2, w3, cnt) VALUES
    (-1009876543210, 'пойдём', 'пить', 'кофе', 1),
    (-1009876543210, 'привет', 'всем', 'как',  1);

-- Данные Маркова n=3
INSERT INTO starts3(chat_id, w1, w2, w3, cnt) VALUES
    (-1009876543210, 'пойдём', 'пить', 'кофе', 1);

INSERT INTO transitions3(chat_id, w1, w2, w3, w4, cnt) VALUES
    (-1009876543210, 'а', 'пойдём', 'пить', 'кофе', 1);

-- Данные Маркова n=1
INSERT INTO transitions1(chat_id, w1, w2, cnt) VALUES
    (-1009876543210, 'пойдём', 'пить',   1),
    (-1009876543210, 'пить',   'кофе',   1),
    (-1009876543210, 'привет', 'всем',   1),
    (-1009876543210, 'всем',   'как',    1),
    (-1009876543210, 'как',    'дела',   1);
