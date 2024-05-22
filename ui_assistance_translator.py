import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import sounddevice as sd
from scipy.io.wavfile import write
import os
import pyttsx3
import datetime
import json
import schedule
import threading
import requests
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gc
import librosa
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

# Загрузка модели Whisper
whisper_model = whisper.load_model("base")

# Настройка для моделей перевода
en_ru_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
en_ru_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
ru_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
ru_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")

ru_voice_id = None
en_voice_id = None

# Инициализация аудио директории
audio_dir = 'audio'
if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)

# Инициализация голосового движка
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    if 'russian' in voice.name.lower():
        ru_voice_id = voice.id
    elif 'english' in voice.name.lower():
        en_voice_id = voice.id

# Глобальные переменные
fs = 44100
seconds = 10
sd.default.dtype = 'int32', 'int32'
reminder_storage = []
is_paused = False
last_phrase = None

def text_to_speech(text, voice=None, rate=150, volume=1.0):
    global engine
    if voice is None:
        voice = engine.getProperty('voice')
    try:
        engine.setProperty('voice', voice)
        engine.setProperty('rate', rate)
        engine.setProperty('volume', volume)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

def start_recording():
    global audio_data
    audio_data = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    text_to_speech("Начало записи...")

def stop_recording():
    global last_recording_filename
    audio_file = os.path.join(audio_dir, 'file.wav')
    write(audio_file, fs, audio_data)
    last_recording_filename = audio_file
    text_to_speech("Запись завершена.")
    process_and_speak(audio_file)

def play_recording():
    os.system(f"start {os.path.join(audio_dir, 'file.wav')}")

def save_recording():
    filepath = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
    if filepath:
        write(filepath, fs, audio_data)
        text_to_speech("Запись сохранена.")

def process_and_speak(audio_path):
    try:
        # Загрузка аудио из файла
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
        
        # Определение языка
        _, probs = whisper_model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        print(f"Обнаруженный язык: {detected_lang}")

        # Декодирование аудио
        options = whisper.DecodingOptions(fp16=False, language=detected_lang)
        result = whisper.decode(whisper_model, mel, options)
        text = result.text
        print(f"Распознанный текст: {text}")
        
        # Определение нужного голоса и возможный перевод
        voice_id = en_voice_id if detected_lang == 'ru' else ru_voice_id
        if detected_lang == 'en':
            input_ids = en_ru_tokenizer.encode(text, return_tensors="pt")
            outputs = en_ru_model.generate(input_ids)
            translated_text = en_ru_tokenizer.decode(outputs[0], skip_special_tokens=True)
        elif detected_lang == 'ru':
            input_ids = ru_en_tokenizer.encode(text, return_tensors="pt")
            outputs = ru_en_model.generate(input_ids)
            translated_text = ru_en_tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            translated_text = text  # Если язык не поддерживается, используется оригинальный текст
        
        # Воспроизведение результата
        text_to_speech(translated_text, voice=voice_id)
        
    except Exception as e:
        print(f"Во время обработки произошла ошибка: {str(e)}")
        text_to_speech("Произошла ошибка. Пожалуйста повторите снова.", voice=en_voice_id)

def repeat_last_phrase():
    if last_recording_filename:
        process_and_speak(last_recording_filename)
    else:
        text_to_speech("Нет записей для воспроизведения.")

def clear_last_phrase():
    global last_phrase
    last_phrase = None
    text_to_speech("Последняя фраза очистилась.")

def pause_resume():
    global is_paused
    is_paused = not is_paused
    text_to_speech("Пауза." if is_paused else "Возобновление записи.")

def send_email_with_profiles():
    # Запрашиваем данные у пользователя через диалоговые окна
    sender_email = simpledialog.askstring("Электронная почта тправителя", "Enter sender's email:")
    sender_password = simpledialog.askstring("Пароль отправителя", "Enter sender's password:", show='*')
    recipient_email = simpledialog.askstring("Электронная почта получателя", "Enter recipient's email:")
    subject = simpledialog.askstring("Тема", "Enter the subject of the email:")
    message = simpledialog.askstring("Сообщение", "Enter the message to send:")

    if not all([sender_email, sender_password, recipient_email, subject, message]):
        messagebox.showerror("Ошибка", "Все поля обязательны для отправки письма.")
        return

    # Создаем MIME-сообщение
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    # Подключаемся к серверу и отправляем сообщение
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)  # Используется реальный адрес SMTP-сервера
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        messagebox.showinfo("Успех", "Письмо успешно отправлено!")
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось отправить электронное письмо: {str(e)}")
            
def execute_reminder(reminder_text):
    # Планируем выполнение в основном потоке GUI
    root.after(0, lambda: text_to_speech(reminder_text))
    root.after(0, lambda: messagebox.showinfo("Напоминание", reminder_text))

def add_reminder():
    reminder_text = simpledialog.askstring("Добавить напоминание", "Что это за напоминание?")
    reminder_date = simpledialog.askstring("Дата и время", "Когда мне следует напомнить тебе? (yyyy-mm-dd HH:MM)")
    try:
        reminder_datetime = datetime.datetime.strptime(reminder_date, "%Y-%m-%d %H:%M")
        reminder_storage.append((reminder_datetime, reminder_text))
        delta_ms = (reminder_datetime - datetime.datetime.now()).total_seconds() * 1000
        if delta_ms > 0:  # Убедимся, что время еще не наступило
            root.after(int(delta_ms), lambda: execute_reminder(reminder_text))
        text_to_speech("Напоминание добавлено.")
    except ValueError:
        messagebox.showerror("Ошибка", "Неправильный формат даты.")

def show_reminders():
    reminders = "\n".join([f"{rem[0].strftime('%Y-%m-%d %H:%M')}: {rem[1]}" for rem in reminder_storage])
    messagebox.showinfo("Напоминания", reminders if reminders else "No reminders set.")

def get_weather():
    city = simpledialog.askstring("Погода", "Введите название города:")
    api_key = 'ef5c2bd6b4f1f2b88d5fd9cb522f9f4b'
    response = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric")
    if response.status_code == 200:
        data = response.json()
        weather = data['weather'][0]['description']
        temp = data['main']['temp']
        text_to_speech(f"Погода в {city}: {weather}, Температура: {temp} градусов Цельсия.")
    else:
        text_to_speech("Не удалось получить информацию о погоде.")

def get_current_time():
    current_time = datetime.datetime.now().strftime("%H:%M")
    text_to_speech(f"Текущее время {current_time}")

def get_current_date():
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    text_to_speech(f"Сегодня {current_date}")

def get_current_day():
    current_day = datetime.datetime.now().strftime("%A")
    text_to_speech(f"Сегодня {current_day}")

def get_random_quote():
    response = requests.get("https://api.quotable.io/random")
    if response.status_code == 200:
        data = response.json()
        quote = f"{data['content']} - {data['author']}"
        text_to_speech(quote)
        return quote
    else:
        text_to_speech("Не удалось получить цитату.")

def get_anecdote():
    response = requests.get("https://api.chucknorris.io/jokes/random")
    if response.status_code == 200:
        data = response.json()
        joke = data["value"]
        text_to_speech(joke)
        return joke
    else:
        text_to_speech("Не удалось получить анекдот.")

# Графический интерфейс
root = tk.Tk()
root.title("Голосовой помощник")
root.geometry("400x600")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Кнопки управления
btn_record = tk.Button(frame, text="Начать запись", command=start_recording)
btn_record.pack(fill=tk.X)

btn_pause = tk.Button(frame, text="Пауза/продолжить", command=pause_resume)
btn_pause.pack(fill=tk.X, pady=5)

btn_stop = tk.Button(frame, text="Прекратить запись", command=stop_recording)
btn_stop.pack(fill=tk.X, pady=5)

btn_play = tk.Button(frame, text="Воспроизводить запись", command=play_recording)
btn_play.pack(fill=tk.X, pady=5)

btn_save = tk.Button(frame, text="Сохранить запись", command=save_recording)
btn_save.pack(fill=tk.X, pady=5)

btn_repeat = tk.Button(frame, text="Повторите последнюю фразу", command=repeat_last_phrase)
btn_repeat.pack(fill=tk.X)

btn_clear = tk.Button(frame, text="Очистить последнюю фразу", command=clear_last_phrase)
btn_clear.pack(fill=tk.X, pady=5)

btn_email = tk.Button(frame, text="Отправить электронное письмо", command=send_email_with_profiles)
btn_email.pack(fill=tk.X, pady=5)

btn_reminder = tk.Button(frame, text="Добавить напоминание", command=add_reminder)
btn_reminder.pack(fill=tk.X, pady=5)

btn_show_reminders = tk.Button(frame, text="Показать напоминания", command=show_reminders)
btn_show_reminders.pack(fill=tk.X, pady=5)

btn_weather = tk.Button(frame, text="Получить погоду", command=get_weather)
btn_weather.pack(fill=tk.X, pady=5)

btn_time = tk.Button(frame, text="Получить текущее время", command=get_current_time)
btn_time.pack(fill=tk.X, pady=5)

btn_date = tk.Button(frame, text="Получить текущую дату", command=get_current_date)
btn_date.pack(fill=tk.X, pady=5)

btn_day = tk.Button(frame, text="Получить текущий день", command=get_current_day)
btn_day.pack(fill=tk.X, pady=5)

btn_quote = tk.Button(frame, text="Получить случайную цитату", command=get_random_quote)
btn_quote.pack(fill=tk.X, pady=5)

btn_anecdote = tk.Button(frame, text="Получить анекдот", command=get_anecdote)
btn_anecdote.pack(fill=tk.X, pady=5)

btn_exit = tk.Button(frame, text="Выход", command=root.quit)
btn_exit.pack(fill=tk.X, pady=5)

scheduler_thread = threading.Thread(target=lambda: schedule.run_pending())
scheduler_thread.daemon = True
scheduler_thread.start()

root.mainloop()
