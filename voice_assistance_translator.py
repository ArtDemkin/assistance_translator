import sounddevice as sd
from scipy.io.wavfile import write
import os
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pyttsx3
import speech_recognition as sr
import time
import pyaudio
import wave
import requests
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import schedule
import threading
import random
import json
import dateparser
from natasha import (Segmenter, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, NewsNERTagger, MorphVocab, Doc)
import spacy
from dateparser.search import search_dates

# Убедитесь, что модели загружены
nlp_en = spacy.load('en_core_web_sm')  
nlp_ru = spacy.load('ru_core_news_sm')  

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
morph_vocab = MorphVocab()

engine = pyttsx3.init()
voices = engine.getProperty('voices')
ru_voice_id = None
en_voice_id = None
for voice in voices:
    if 'RU' in voice.id:
        ru_voice_id = voice.id
    elif 'EN' in voice.id:
        en_voice_id = voice.id
engine.stop()

sd.default.dtype = 'int32', 'int32'
fs = 44100  
seconds = 600  
audio_dir = 'audio'
if not os.path.exists(audio_dir):  
    os.makedirs(audio_dir)

whisper_model = whisper.load_model("base")

en_ru_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru", padding_side='left')
en_ru_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
ru_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en", padding_side='left')
ru_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")

reminder_storage = []

def text_to_speech(text, voice, rate=150, volume=1.0, filename=None):
    try:
        engine = pyttsx3.init()
        engine.setProperty('voice', voice)
        engine.setProperty('rate', rate)
        engine.setProperty('volume', volume)
        if filename:
            engine.save_to_file(text, filename)
            engine.runAndWait()
        else:
            engine.say(text)
            engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

def speak_and_print(text, voice_id=ru_voice_id):
    print(text)
    text_to_speech(text, voice=voice_id)

def start_recording(data):
    global myrecording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    print("Начало записи...")

def stop_recording(data):
    sd.stop()
    speak_and_print("Остановлена запись, начинаю обработку...")
    audio_file = os.path.join(audio_dir, 'file.wav')
    global last_recording_filename
    last_recording_filename = audio_file
    write(audio_file, fs, myrecording)
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    _, probs = whisper_model.detect_language(mel)
    detected_lang = next(iter({max(probs, key=probs.get)}))
    print("Обнаружен язык: ", detected_lang)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(whisper_model, mel, options)
    text = result.text
    print(text)
    global last_phrase
    last_phrase = text  # Сохраняем последнюю распознанную фразу
    if detected_lang == 'en':
        input_ids = en_ru_tokenizer.encode(text, return_tensors="pt")
        outputs = en_ru_model.generate(input_ids)
        translated = en_ru_tokenizer.decode(outputs[0], skip_special_tokens=True)
        voice_id = ru_voice_id
        print(translated)
    elif detected_lang == 'ru':
        input_ids = ru_en_tokenizer.encode(text, return_tensors="pt")
        outputs = ru_en_model.generate(input_ids)
        translated = ru_en_tokenizer.decode(outputs[0], skip_special_tokens=True)
        voice_id = en_voice_id
        print(translated)
    else:
        translated = None

    if translated:
        text_to_speech(translated, voice=voice_id, rate=150, volume=1.0)
    else:
        text = "Этот язык не поддерживается"
        print(text)
        text_to_speech(text, voice=ru_voice_id, rate=150, volume=1.0)

def voice_control():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5  # Уменьшено для ускорения тестов
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("* Прослушивание...")

    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* Запись завершена.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    recognizer = sr.Recognizer()
    with sr.AudioFile(WAVE_OUTPUT_FILENAME) as source:
        audio_data = recognizer.record(source)
        try:
            recognized_text = recognizer.recognize_google(audio_data, language="ru")
            print("Распознанный текст:", recognized_text)
            return recognized_text
        except sr.UnknownValueError:
            print("Извините, я не понял, что вы сказали.")
        except sr.RequestError:
            speak_and_print("Извините, мой сервис распознавания речи в настоящее время недоступен. Пожалуйста, попробуйте позже.")

    return None

def save_credentials(credentials, filename):
    try:
        data = []
        if os.path.exists(filename):
            with open(filename, "r") as file:
                data = json.load(file)
        data.append(credentials)
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
        speak_and_print(f"Учетные данные сохранены в {filename}")
    except Exception as e:
        print(f"Произошла ошибка при сохранении учетных данных: {str(e)}")

def load_credentials(filename):
    try:
        with open(filename, "r") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return [data]
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Ошибка загрузки {filename}: {str(e)}")
        return []

def get_voice_input(prompt, valid_commands=None):
    print(prompt)
    while True:
        response = voice_control()
        if response:
            if valid_commands and response.lower() in valid_commands:
                return response
            elif not valid_commands:
                return response
            else:
                speak_and_print("Недопустимая команда. Пожалуйста, попробуйте еще раз.")
                text_to_speech("Недопустимая команда. Пожалуйста, попробуйте еще раз.", voice=en_voice_id)
        else:
            text_to_speech("Извините, я не расслышал. Пожалуйста, попробуйте еще раз.", voice=en_voice_id)

def load_profile_by_name(name, filename='recipient_credentials.json'):
    name = name.lower().replace('ё', 'е').rstrip('у')
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            profiles = json.load(file)
            for profile in profiles:
                profile_name = profile['name'].lower().replace('ё', 'е')
                if profile_name.startswith(name):
                    return profile['email']
    except FileNotFoundError:
        print(f"Файл не найден: {filename}")
    except Exception as e:
        print(f"Ошибка чтения из {filename}: {e}")
    return None

def extract_email_details(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    
    email_details = {
        'name': None,
        'email': None,
        'body': text  # По умолчанию - весь текст, если имя не найдено
    }

    for span in doc.spans:
        if span.type == 'PER':
            name = span.text
            email = load_profile_by_name(name)
            if email:
                email_details['name'] = name
                email_details['email'] = email
                email_details['body'] = text.replace(name, "").strip()  # Удаление имени из тела
                break

    return email_details

def get_profile_by_name(name, filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            profiles = json.load(file)
            for profile in profiles:
                if profile['name'].lower() == name.lower():
                    return profile
    except FileNotFoundError:
        speak_and_print("Файл профиля не найден.")
    except json.JSONDecodeError:
        speak_and_print("Ошибка декодирования JSON из файла профиля.")
    return None

def set_or_get_profile(profile_type):
    filename = 'sender_credentials.json' if profile_type == "sender" else 'recipient_credentials.json'
    print(f"Пожалуйста, скажите имя {profile_type}:")
    name = voice_control()  # Получаем имя голосом
    if not name:
        speak_and_print("Не удалось распознать имя. Пожалуйста, попробуйте еще раз.")
        return None

    profile = get_profile_by_name(name, filename)

    if profile:
        print(f"Профиль найден: {profile['name']} с email {profile['email']}")
        return profile
    else:
        print(f"Профиль для {name} не найден. Создание нового профиля.")
        print(f"Пожалуйста, скажите email {profile_type}:")
        email = voice_control()  # Получаем email голосом
        if not email:
            print("Не удалось распознать email. Пожалуйста, попробуйте еще раз.")
            return None

        credentials = {'name': name, 'email': email}
        if profile_type == "sender":
            speak_and_print("Пожалуйста, скажите пароль отправителя:")
            password = voice_control()  # Получаем пароль голосом
            if not password:
                speak_and_print("Не удалось распознать пароль. Пожалуйста, попробуйте еще раз.")
                return None
            credentials['password'] = password
        
        save_credentials(credentials, filename)
        return credentials

def set_recipient_profile_voice():
    speak_and_print("Пожалуйста, скажите имя получателя:")
    name = voice_control()
    if not name:
        speak_and_print("Не удалось распознать имя. Пожалуйста, попробуйте еще раз.")
        return None

    speak_and_print("Введите email получателя: ")
    email = input()
    if not email:
        speak_and_print("Не удалось распознать email. Пожалуйста, попробуйте еще раз.")
        return None

    credentials = {'name': name, 'email': email}
    save_credentials(credentials, 'recipient_credentials.json')
    return credentials

def send_email_with_profiles():
    speak_and_print("Настройка профиля отправителя...")
    sender_profile = set_or_get_profile("sender")
    
    if not sender_profile:
        speak_and_print("Не удалось получить или настроить профиль отправителя.")
        return

    # Запрос на создание нового профиля получателя
    create_new_recipient = get_voice_input("Создать новый профиль получателя? Скажите 'да' или 'нет'.", valid_commands=['да', 'нет'])
    recipient_profile = None

    if create_new_recipient.lower() == 'да':
        recipient_profile = set_recipient_profile_voice()
        if recipient_profile is None:
            speak_and_print("Не удалось создать новый профиль получателя.")
            return

    # Запрос деталей письма
    speak_and_print("Пожалуйста, скажите детали письма, включая имя получателя и сообщение:")
    spoken_text = get_voice_input("Пожалуйста, говорите")
    email_info = extract_email_details(spoken_text)

    # Если создан новый профиль, он становится получателем
    if recipient_profile:
        email_info['name'] = recipient_profile['name']
        email_info['email'] = recipient_profile['email']
    else:
        # Иначе используем найденные данные или ошибка
        if not email_info['email']:
            speak_and_print("Не удалось найти email адрес получателя в произнесенном сообщении. Пожалуйста, попробуйте еще раз.")
            return

    print(f"Используется отправитель: {sender_profile['name']} ({sender_profile['email']})")
    print(f"Отправка письма {email_info['name']} на {email_info['email']}")
    subject = ""  # Вы можете добавить логику извлечения темы здесь, если необходимо
    send_email(sender_profile['email'], email_info['email'], subject, email_info['body'], sender_profile.get('password'))

def send_email(sender_email, recipient_email, subject, body, password):
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        speak_and_print("Письмо успешно отправлено!")
    except Exception as e:
        print(f"Не удалось отправить письмо: {str(e)}")

def set_sender_profile():
    name = input("Введите имя отправителя: ")
    email = input("Введите email отправителя: ")
    password = input("Введите пароль отправителя: ")
    credentials = {'name': name, 'email': email, 'password': password}
    save_credentials(credentials, 'sender_credentials.json')
    return credentials

def set_recipient_profile():
    name = input("Введите имя получателя: ")
    email = input("Введите email получателя: ")
    credentials = {'name': name, 'email': email}
    save_credentials(credentials, 'recipient_credentials.json')
    return credentials

def save_last_phrase():
    global last_phrase
    if last_phrase:
        speak_and_print("Сохранение последней фразы в файл...")
        filename = "last_phrase.txt"
        try:
            with open(filename, "w", encoding="utf-8") as file:
                file.write(last_phrase)
            print(f"Последняя фраза успешно сохранена в '{filename}'.")
        except Exception as e:
            print(f"Не удалось сохранить последнюю фразу в файл: {str(e)}")
    else:
        speak_and_print("Нет доступной фразы для сохранения.")

def clear_last_phrase():
    global last_phrase
    if last_phrase:
        speak_and_print("Очистка последней фразы...")
        last_phrase = None
        speak_and_print("Последняя фраза очищена.")
    else:
        speak_and_print("Нет доступной фразы для очистки.")

def play_last_recording():
    global last_recording_filename
    if last_recording_filename:
        speak_and_print("Воспроизведение последней записи...")
        os.system(f"start {last_recording_filename}")
    else:
        speak_and_print("Нет доступной записи для воспроизведения.")

def get_weather(city):
    api_key = 'ef5c2bd6b4f1f2b88d5fd9cb522f9f4b'
    base_url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'
    
    try:
        response = requests.get(base_url)
        data = response.json()
        
        if response.status_code == 200:
            weather_desc = data['weather'][0]['description'].capitalize()
            temp = data['main']['temp']
            humidity = data['main']['humidity']
            wind_speed = data['wind']['speed']
            
            weather_info = f"Текущая погода в {city}: {weather_desc}. Температура: {temp}°C, Влажность: {humidity}%, Скорость ветра: {wind_speed} м/с."
            print(weather_info)
            text_to_speech(weather_info, voice=en_voice_id, rate=150, volume=1.0)
        else:
            print("Не удалось получить информацию о погоде. Пожалуйста, попробуйте еще раз.")
            text_to_speech("Не удалось получить информацию о погоде. Пожалуйста, попробуйте еще раз.", voice=ru_voice_id, rate=150, volume=1.0)
    except Exception as e:
        print(f"Произошла ошибка при получении информации о погоде: {str(e)}")
        text_to_speech("Произошла ошибка при получении информации о погоде.", voice=ru_voice_id, rate=150, volume=1.0)

def get_current_time():
    current_time = datetime.datetime.now().strftime("%H:%M")
    print("Текущее время:", current_time)
    text_to_speech(f"Текущее время {current_time}", voice=en_voice_id, rate=150, volume=1.0)

def get_current_date():
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    print("Текущая дата:", current_date)
    text_to_speech(f"Сегодняшняя дата {current_date}", voice=en_voice_id, rate=150, volume=1.0)

def get_current_day():
    current_day = datetime.datetime.now().strftime("%A")
    print("Сегодня:", current_day)
    text_to_speech(f"Сегодня {current_day}", voice=en_voice_id, rate=150, volume=1.0)

def set_reminder(reminder_text):
    print("Напоминание:", reminder_text)
    text_to_speech(f"Напоминание: {reminder_text}", voice=en_voice_id, rate=150, volume=1.0)

def combine_date_and_time(date, time):
    return datetime.datetime.combine(date.date(), time.time())

def add_reminder():
    speak_and_print("Пожалуйста, предоставьте детали напоминания:")
    reminder_text = get_voice_input("Что напомнить?")
    reminder_time_text = get_voice_input("Когда напомнить? Например, 'завтра в 15 часов' или '15 мая в 17:00'.")

    parsed_dates = search_dates(reminder_time_text, languages=['ru'])
    print(f"Распознанные даты: {parsed_dates}")  # Добавить логирование распознанных дат

    if not parsed_dates:
        text_to_speech("Не удалось распознать дату и время. Попробуйте еще раз.", voice=en_voice_id)
        return

    # Инициализация переменных
    date_part = None
    time_part = None

    # Извлечение частей даты и времени
    for part in parsed_dates:
        if 'через' in part[0] or 'день' in part[0] or 'дней' in part[0] or 'завтра' in part[0] or 'послезавтра' in part[0]:
            date_part = part[1]
        elif 'в' in part[0] or ':' in part[0]:
            time_part = part[1]

    # Проверка на "вечера" для установки правильного часа
    if 'вечера' in reminder_time_text and time_part and time_part.hour < 12:
        time_part = time_part.replace(hour=time_part.hour + 12)

    if date_part and time_part:
        reminder_datetime = combine_date_and_time(date_part, time_part)
    else:
        # Резервный вариант, если найдена только одна часть
        reminder_datetime = parsed_dates[0][1]

    # Если найдена только часть даты, добавьте также часть времени
    if date_part and not time_part:
        reminder_datetime = date_part
    elif time_part and not date_part:
        reminder_datetime = time_part

    print(f"Распознанное время: {reminder_datetime}")  # Добавить логирование распознанного времени

    if reminder_datetime < datetime.datetime.now():
        text_to_speech("Дата и время уже прошли. Укажите будущее время.", voice=en_voice_id)
        return

    # Запланировать напоминание
    schedule_time = reminder_datetime.strftime("%H:%M")
    job = schedule.every().day.at(schedule_time).do(set_reminder, reminder_text=reminder_text)
    reminder_storage.append((reminder_datetime.strftime('%Y-%m-%d %H:%M'), reminder_text))
    print(f"Напоминание установлено на: {reminder_datetime.strftime('%Y-%m-%d %H:%M')}")
    text_to_speech(f"Напоминание установлено на {reminder_datetime.strftime('%Y-%m-%d %H:%M')}", voice=en_voice_id)

def list_reminders():
    if reminder_storage:
        for date, text in reminder_storage:
            print(f"Напоминание: {text} на {date}")
    else:
        speak_and_print("Напоминаний нет.")
        
def get_quote():
    try:
        response = requests.get("https://api.quotable.io/random")
        if response.status_code == 200:
            data = response.json()
            quote = data["content"]
            author = data["author"]
            full_quote = f"{quote} - {author}"
            return full_quote
        else:
            return "Не удалось получить цитату. Пожалуйста, попробуйте позже."
    except Exception as e:
        print(f"Произошла ошибка при получении цитаты: {str(e)}")
        return "Произошла ошибка при получении цитаты. Пожалуйста, попробуйте позже."

def get_random_quote():
    quote = get_quote()
    speak_and_print("Вот цитата для вас:")
    print(quote)
    text_to_speech(quote, voice=en_voice_id, rate=150, volume=1.0)

def get_anecdote():
    try:
        response = requests.get("https://api.chucknorris.io/jokes/random")
        if response.status_code == 200:
            data = response.json()
            joke = data["value"]
            speak_and_print("Вот анекдот для вас:")
            print(joke)
            text_to_speech(joke, voice=en_voice_id, rate=150, volume=1.0)
        else:
            speak_and_print("Не удалось получить анекдот. Пожалуйста, попробуйте позже.")
            text_to_speech("Не удалось получить анекдот. Пожалуйста, попробуйте позже.", voice=ru_voice_id, rate=150, volume=1.0)
    except Exception as e:
        print(f"Произошла ошибка при получении анекдота: {str(e)}")
        text_to_speech("Произошла ошибка при получении анекдота. Пожалуйста, попробуйте позже.", voice=ru_voice_id, rate=150, volume=1.0)

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

scheduler_thread = threading.Thread(target=run_scheduler)
scheduler_thread.daemon = True
scheduler_thread.start()

last_phrase = None
last_recording_filename = None
while True:
    print("Прослушивание голосовой команды...")
    choice = voice_control()
    if choice is not None:
        if choice == 'старт':
            start_recording(None)
        elif choice == 'стоп':
            stop_recording(None)
        elif choice == 'выход':
            speak_and_print("Выход из программы.")
            break
        elif choice == 'пауза':
            speak_and_print("Программа приостановлена...")
            print("Скажите 'продолжить', чтобы продолжить, или 'выход', чтобы выйти из программы.")
            while True:
                new_command = voice_control()
                if new_command is not None:
                    if new_command == 'продолжить':
                        speak_and_print("Возобновление программы...")
                        break
                    elif new_command == 'выход':
                        speak_and_print("Выход из программы.")
                        exit()
                    else:
                        print("Неправильная команда после приостановки. Пожалуйста, скажите 'продолжить' или 'выход'.")
                else:
                    print("Голосовая команда не распознана. Пожалуйста, попробуйте еще раз.")
        elif choice == 'помощь':
            speak_and_print("Доступные команды:")
            print("- 'старт' чтобы начать запись")
            print("- 'стоп' чтобы остановить запись и перевести")
            print("- 'пауза' чтобы приостановить программу")
            print("- 'продолжить' чтобы возобновить программу, если она приостановлена")
            print("- 'повторить' чтобы повторить последнюю произнесенную фразу")
            print("- 'сохранить' чтобы сохранить последнюю произнесенную фразу в текстовый файл")
            print("- 'очистить' чтобы очистить последнюю произнесенную фразу")
            print("- 'проиграть' чтобы воспроизвести последнюю запись")
            print("- 'погода' чтобы получить информацию о погоде для города")
            print("- 'время' чтобы узнать текущее время")
            print("- 'дата' чтобы узнать текущую дату")
            print("- 'день' чтобы узнать текущий день")
            print("- 'отправить письмо' чтобы отправить email")
            print("- 'напоминание' чтобы установить напоминание")
            print("- 'список напоминаний' чтобы получить список напоминаний")
            print("- 'цитата' чтобы получить случайную цитату")
            print("- 'анекдот' чтобы получить случайный анекдот")
            print("- 'выход' чтобы выйти из программы")
        elif choice == 'повторить':
            repeat_last_phrase()
        elif choice == 'сохранить':
            save_last_phrase()
        elif choice == 'очистить':
            clear_last_phrase()
        elif choice == 'проиграть':
            play_last_recording()
        elif choice == 'погода':
            speak_and_print("Пожалуйста, укажите город для информации о погоде:")
            city = voice_control()
            if city:
                get_weather(city)
        elif choice == 'время':
            get_current_time()
        elif choice == 'дата':
            get_current_date()
        elif choice == 'день':
            get_current_day()
        elif choice == 'отправить письмо':
            send_email_with_profiles()
        elif choice == 'напоминание':
            add_reminder()
        elif choice.lower() == 'список напоминаний':
            list_reminders()
        elif choice == 'цитата':
            get_random_quote()
        elif choice == 'анекдот':
            get_anecdote()
        else:
            print("Недопустимый выбор. Пожалуйста, попробуйте еще раз.")
    else:
        print("Голосовая команда не распознана. Пожалуйста, попробуйте еще раз.")
