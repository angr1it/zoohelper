## zoohelper

t.me/zoohelper_helpful_bot

### Оглавление

1. [Запуск приложения](README.md#запуск-приложения)
2. [Исследование](notebooks/)

### Запуск приложения

#### 1. Подготовка: .env файл
Заполнить .env файл, добавив туда переменную BOT_TOKEN, содержащую токен чат-бота в telegram (см. [example.env](example.env))

#### 2. Запуск через docker-контейнер
Для запуска приложения достаточно выполнить комманду:
```
docker-compose --profile prod up
```