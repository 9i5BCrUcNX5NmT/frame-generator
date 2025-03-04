# Описываю свою работу

## Направления развития:
- обработать данные
- улучшить слои(использовать диффузионную модель? Модель EDM?)
- сделать демонстрационный вывод генерации
- переработать структуру проекта(сделать несколько модулей для обучения, обработки данных и тд)
- сделать сжатие данных до входа в модель и возвращение после(по аналогии с stable diffusion)

## Обработка данных

Для правильного обучения нейронной сети необходимо правильно собрать и обработать данные.

Изображения(цвета в каждом пикселе) + нажатия клавиш(номер клавиши в кодировке)

![Данные](screenshots/image.png)

Есть варианты соединения:
- каждому кадру — все клавиши, которые были нажаты(могут быть пересечения)(подходит, если кадры медленнее считываются, чем нажатия)
- одному кадру — одно нажатие(для синхронизации)(брать самое длинное по времени нажатие клавиши)
- убирать лишние кадры, если нажатия редкие

Без соединения, можно сохранять время кадра и нажатия, но предсказывать будет сложнее.

Может использовать порядок для обозначения связи?

Так как в процессе генерации может быть задействовано несколько клавиш, то единственный оставшийся вариант — каждому кадру соответствие множества клавиш.

Для автоматического совмещения можно или изменить способ сбора данных в датасет(вместо obs для записи включить сохранение фото в python скрипт или подобный), или совмещать на пост обработке, уже после преобразования видео в кадры и сбора отдельно клавиш.

Для 1го способа нужно сделать новый скрипт для сбора и обработки фото и клавиш.
Для 2го способа нужно сделать только скрипт для пост обработки, что легче, чем для 1го способа.

![Генерация](screenshots/image-1.png)

![alt text](screenshots/image.png)

Возможно использовать формат данных hdf5(по аналогии с diamond)

Данные для обучения будут в виде цветов пикселей изображения вместе с вектором, обозначающим нажата ли была какая-либо кнопка.

Надо сделать программу для сбора данных для обучения. +

Надо объединить сбор данных в основной модуль.

Надо сделать запись видео совместно с нажатием кнопок(для автоматизации)


### Задача 5?

Итак, есть выбор:
- считывать скрин экрана одновременно с нажатиями(для соответствия их друг другу)
- записывать виде и нажатия раздельно(потом надо будет разбитое на картинки видео соотнести с нажатиями)

Анализ:

Выбор между этими подходами зависит от ваших целей, требований к производительности и сложности реализации. Рассмотрим оба варианта.

1. Считывание скриншотов экрана одновременно с нажатиями

    - Преимущества:
        - Простота синхронизации: данные записываются в одном потоке, что упрощает синхронизацию и анализ.
        - Меньше обработки данных: не нужно разбивать видео и сопоставлять кадры с нажатиями.
        - Меньше шансов на потерю данных: записываются одновременно, вероятность несоответствий снижается.

    - Недостатки:
        - Нагрузка на систему: может быть ресурсоемким, особенно при высокой частоте (20 кадров в секунду).
        - Ограниченная гибкость: изменение частоты записи требует значительных изменений в коде.

2. Запись видео и нажатий раздельно

    - Преимущества:

        - Гибкость: настраиваемая частота записи независимо друг от друга.

        - Меньшая нагрузка на систему: запись видео с низкой частотой снижает нагрузку.

    - Недостатки:

        - Сложность синхронизации: разбивка видео на кадры и сопоставление с нажатиями трудоемки.

        - Дополнительная обработка данных: увеличивает сложность проекта.

Рекомендации:

- Высокая точность и простота синхронизации: первый подход упростит анализ данных и снизит ошибки.

- Гибкость и дополнительная сложность: второй вариант предпочтителен для оптимизации производительности и специфических требований к частоте записи.

Можно начать с первого подхода, а затем перейти ко второму.

## Работа с данными для обучения
- Использование формата hdf5(можно про него рассказать)