### Серверная часть

В этом проекте происходит обработка эталонной карты и спутникового снимка. Вся логика работы описана в файле server.

---

### Инициализация


В директории в которую был был склонирован репозиторий создаете виртуальную среду
```bash    
python -m venv .venv
```

После этого активируем виртуальную среду
```bash
.venv\Scripts\Activate
```


Далее необходимо установить бибилиотеки
```bash
pip install -r requirements.txt
```

Также вам необходимо добавить вашы веса в следующую дирректорию:
`models\weights`
В данном случае необходимы веса моделей SuperPoint и SuperGlue

Для запуска сервера мы использвали ngrok, если он у вас не скачен, то можете перейти на официальный сайт и [скачать](https://ngrok.com/).
После успешной установки мы можем запустиТЬ сервер

```bash
ngrok http 5000
```

В данном случае я запускаю на 5000 порту.
После чего можно запускать сервер.

```bash
python server.py
```


---

### Описание серверной части
Серверная часть реализована на базе Python с использованием Quart – асинхронного веб-фреймворка для создания REST API. Она обрабатывает запросы, принимает изображения, выполняет их обработку с использованием моделей SuperPoint и SuperGlue, и возвращает результирующее изображение с отмеченной областью совпадения.

### Функциональность сервера
Прием изображений:
  Сервер принимает два изображения через POST-запрос:
  Эталонная карта (map_image).
  Спутниковый снимок (satellite_image).
  Изображения передаются в формате multipart/form-data.
  
Обработка изображений:
  Загруженные изображения конвертируются в формат, пригодный для обработки нейронными сетями.
  SuperPoint используется для извлечения ключевых точек и дескрипторов.
  SuperGlue сопоставляет ключевые точки на двух изображениях.

Выделение совпадающих областей:
  На эталонной карте определяется ограничивающий прямоугольник, охватывающий совпадающие области.
  Прямоугольник визуализируется на эталонной карте.
  
Возврат результата:
  Результирующее изображение возвращается клиенту в виде файла изображения в формате JPEG.


---
