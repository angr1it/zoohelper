## zoohelper



### Оглавление
1. [Описание задачи](#описание-задачи)
2. [Исследование](notebooks/README.MD)
3. [Описание параметров](notebooks/features.md#описание-параметров-модели-на-ввод)
4. [Запуск приложения](README.md#запуск-приложения)
5. [Работа с ботом](README.md#работа-с-ботом)
6. [Развитие приложения (инференс)](README.md#развитие-приложения-инференс)

### Описание задачи

Гипотеза: лечение животного стоит дорого и не всегда удачно, сервис предоставляющий оценку вероятности успешного лечения мог бы быть полезен.

Что делаем:

- Даем рекомендации о целесообразности лечения лошадей;
- Клиники могут рекомендовать лечение даже в случае тяжелой болезни, когда вероятность хорошего исхода низка;


Оплата: подписка, оплата за одну рекомендацию;

Перспективы развития приложения: 
- расширение под другие классы животных;
- улучшение интерфейса (веб-страница с опросом);
- расширить вывод -- предоставлять доводы в пользу рекомандации, возможные способы решения, стоимость.

### Запуск приложения

#### 1. Подготовка: .env файл
Заполнить .env файл, добавив туда переменную BOT_TOKEN, содержащую токен чат-бота в telegram (см. [example.env](example.env))

#### 2. Запуск через docker-контейнер
Для запуска приложения достаточно выполнить команду:
```
docker-compose --profile prod up
```

### 3. Запуск на локальной машине
Из корневой директории репозитория проекта:
```
pip install -r requirements.txt
python ./src/bot.py
```

### Работа с ботом

- Подключаться к настроенному чату ([см. пункт 2.1](README.md#1-подготовка-env-файл))
- Следовать подсказкам в чате (команда /help)


### Развитие приложения (инференс)

В данный момент точкой входа является скрипт src/bot.py, который совмещает и инференс модели, и поддерживает работу приложения telegram бота.

Очевидно, узким местом текущей версии приложения является именно запуск инференса, что является cpu-bound задачей. По этой причине, не принималось попыток, например, оптимизировать приложение путем перехода к асинхронному коду, т.к. это не даст ощутимых результатов.

Лучшим решением, считаем, переход к сервисной архитектуре, при котором запуск и поддержка инференса модели выделится в отдельный сервис. 
- Сервис-бот будет обрабатывать сообщения и отправлять параметры в сервис инференса через мессенджер сообщений;
- Мессенджер будет выполнять роль очереди, что само по себе будет сглаживать нагрузку;
- На стороне сервиса инференса можно будет производить батчинг (при необходимости)