# Интеллектуальная система распознавания маркировки

Задача была разбита на несколько составляющих: определение типа маркировки (круглая или прямая), локализация маркировки на изображении и распознавание текста на полученной области и сопоставление с базой известных маркировок и порядковых номеров.

## Структура проекта

Ниже представлена структура проекта

```plaintext
.
├── server/                  # Папка для серверной части проекта
│   ├── main.py              # Основной файл приложения FastAPI
│   ├── models.py            # Модели SQLAlchemy для работы с базой данных
│   ├── schemas.py           # Pydantic-схемы для валидации данных
│   ├── database.py          # Настройки подключения к базе данных
│   ├── config.py            # Подгрузка данных из .env
│   └── requirements.txt     # Зависимости для серверной части
├── ml/                      # Папка с модулями машинного обучения (если используется)
│   ├── init.py              # Инициализация объектов моделей
│   ├── image_processing.py  # Обработка изображений и предсказание
│   └── yolo11s_best.pt      # Лучшие веса для детектора       
├── mobile/                  # Папка с кодом мобильного приложения
│   └── <Rosatom CV>         # Исходный код React Native приложения
├── docker-compose.yml       # Конфигурация Docker для запуска всего проекта
├── Dockerfile               # Dockerfile для серверной части
├── README.md                # Этот файл
└── requirements.txt         # Зависимости для серверной части
```
## Инструкция

### Развертывание сервера
1. Для начала, клонируйте репозиторий:

```bash
git clone https://github.com/GaagAlex1/rosatom_text_recognition.git
```

2. Установите Docker и Docker compose
3. Создайте .env файл со следующим содержимым
```plaintext
DB_HOST=postgres
DB_PORT=...
DB_USER=...
DB_PASS=...
DB_NAME=...
```
4. Находясь в корне проекта выпоните
```bash
docker compose up --build
```
5. Поздравляем, серверная часть запущена и может быть использована!

### Работа с мобильным приложением

1. Скачайте .APK по ссылке
2. Установите .APK
3. Поздравляем, вы получили доступ к приложению!

### Отладка и сборка приложения
#### Отладка
```bash
npx expo start -c
```

#### Сборка
```bash
eas build -p android --profile preview --clear-cache
```

## Ссылка на APK
https://drive.google.com/file/d/1kC09g39nU_T5DJL74r-hVX4BwDBXc0G0/view?usp=sharing

## Пример прогноза 
![Прогноз на радиальной детали](https://github.com/GaagAlex1/rosatom_text_recognition/blob/main/example.jpg)
(195-30-1286 3090)


Детекция области текста с помощью новейшей версии YOLO v11
Определение вида детали (круглая или обычная) при помощи кругового преобразования Хафа.
Если деталь круглая то производится выпрямление текста в прямую при помощи непрерывного отображения.
Распознавание текста на выпрямленном изображении при помощи модели GOT-OCR2.0
Обработка выделенного текста алгоритмом и сопоставление с базой данных всех деталей с помощью библиотеки rapid fuzz. 
В качестве детектора была выбрана YOLO так как она является одной из лучших моделей для данной задачи и удобно интегрируется с клиент-серверными приложениями. А трансформер GOT-OCR2.0 отлично справляется с задачей распознавания текста на металлическом фоне.

