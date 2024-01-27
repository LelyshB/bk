import tensorflow as tf
import os
from PyPDF2 import PdfReader
import time

# Путь к папке с контрольными точками
checkpoint_dir = './checkpoints/train'

# Создание объекта чекпоинта
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

# Восстановление последней контрольной точки
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Restored from {}".format(ckpt_manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

# Функция для перевода текста
def translate(transformer, tokenizer_en, tokenizer_ru, sentence, max_length=50):
    # Токенизация предложения
    sentence = tokenizer_en.encode(sentence)
    sentence = tf.expand_dims(sentence, 0)

    # Декодер начинает с токена [START]
    decoder_input = [tokenizer_ru.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    # Предсказание последовательно, токен за токеном
    for i in range(max_length):
        predictions, _ = transformer([sentence, output], training=False)

        # Получаем последний предсказанный токен
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # Если [END] токен, завершаем
        if predicted_id == tokenizer_ru.vocab_size+1:
            break

        # Конкатенируем предсказанный токен к выходу
        output = tf.concat([output, predicted_id], axis=-1)

    translated_text = tokenizer_ru.decode([i for i in output if i < tokenizer_ru.vocab_size])
    
    return translated_text

# Функция для чтения PDF и перевода текста
def translate_pdf(pdf_path, transformer, tokenizer_en, tokenizer_ru):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    translated_text = translate(transformer, tokenizer_en, tokenizer_ru, text)
    
    # Сохраняем переведенный текст
    output_path = os.path.join(checkpoint_dir, f'translated_{time.time()}.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(translated_text)
    
    print(f"Перевод сохранен в {output_path}")

# Пример использования
pdf_path = 'path_to_pdf_file.pdf'
translate_pdf(pdf_path, transformer, tokenizer_en, tokenizer_ru)
































