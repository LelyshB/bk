import time
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.layers import MultiHeadAttention
import keras_tuner as kt
from keras.callbacks import EarlyStopping, ModelCheckpoint
import nlpaug.augmenter.word as naw


en_file_path_train = 'multitarget-ted/en-ru/tok/ted_train_en-ru.tok.en'
ru_file_path_train = 'multitarget-ted/en-ru/tok/ted_train_en-ru.tok.ru'
en_file_path_val = 'multitarget-ted/en-ru/tok/ted_dev_en-ru.tok.en'
ru_file_path_val = 'multitarget-ted/en-ru/tok/ted_dev_en-ru.tok.ru'
en_file_path_test = 'multitarget-ted/en-ru/tok/ted_test1_en-ru.tok.en'
ru_file_path_test = 'multitarget-ted/en-ru/tok/ted_test1_en-ru.tok.ru'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# загрузка данных
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    sentences = [' '.join(line.strip().split()) for line in lines]
    return sentences

BATCH_SIZE = 32
BUFFER_SIZE = 20000
max_length = 70

def augment_sentences(sentences, num_augments=1):
    aug = naw.SynonymAug(aug_src='wordnet')
    augmented_sentences = []
    for sentence in sentences:
        augmented_sentences.append(sentence)
        for _ in range(num_augments):
            augmented_sentence = aug.augment(sentence)
            augmented_sentences.append(augmented_sentence)
    return augmented_sentences

# Текстовая токенизация и детокенизация
en_sentences_train = load_data(en_file_path_train)
ru_sentences_train = load_data(ru_file_path_train)

en_sentences_train = augment_sentences(en_sentences_train)
ru_sentences_train = augment_sentences(ru_sentences_train)

vocab_size_factor = 2**3  

tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (en_sentence for en_sentence in en_sentences_train), target_vocab_size=2**13 * vocab_size_factor)

tokenizer_ru = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (ru_sentence for ru_sentence in ru_sentences_train), target_vocab_size=2**13 * vocab_size_factor)


def verify_data_shapes(sentences, tokenizer, dataset_name):
    for sentence in sentences[:5]: 
        encoded = tokenizer.encode(sentence)
        if len(encoded) == 0:
            raise ValueError(f"Ошибка токенизации в {dataset_name}, пустая последовательность.")

    print(f"{dataset_name} прошло проверку токенизации.")

verify_data_shapes(en_sentences_train, tokenizer_en, "English Training Data")
verify_data_shapes(ru_sentences_train, tokenizer_ru, "Russian Training Data")

def encode_sentences(sentences, tokenizer):
    sequences = [[tokenizer.vocab_size] + tokenizer.encode(sentence) + [tokenizer.vocab_size+1] for sentence in sentences]
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post')
    return sequences  

# Теперь загрузим и обработаем каждую из выборок
en_sentences_train = load_data(en_file_path_train)
ru_sentences_train = load_data(ru_file_path_train)
en_sentences_val = load_data(en_file_path_val)
ru_sentences_val = load_data(ru_file_path_val)
en_sentences_test = load_data(en_file_path_test)
ru_sentences_test = load_data(ru_file_path_test)

# Токенизация и создание датасетов для каждой выборки
# Обучающая выборка
en_sequences_train = encode_sentences(en_sentences_train, tokenizer_en)
ru_sequences_train = encode_sentences(ru_sentences_train, tokenizer_ru)
train_dataset = tf.data.Dataset.from_tensor_slices((en_sequences_train, ru_sequences_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Валидационная выборка
en_sequences_val = encode_sentences(en_sentences_val, tokenizer_en)
ru_sequences_val = encode_sentences(ru_sentences_val, tokenizer_ru)
val_dataset = tf.data.Dataset.from_tensor_slices((en_sequences_val, ru_sequences_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Тестовая выборка
en_sequences_test = encode_sentences(en_sentences_test, tokenizer_en)
ru_sequences_test = encode_sentences(ru_sentences_test, tokenizer_ru)
test_dataset = tf.data.Dataset.from_tensor_slices((en_sequences_test, ru_sequences_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def verify_data_preprocessing(dataset, dataset_name):
    for (inp, tar) in dataset.take(1): 
        if inp.shape[1] != max_length or tar.shape[1] != max_length:
            raise ValueError(f"{dataset_name} имеет неверную длину последовательности.")
        print(f"{dataset_name} прошло проверку размерности.")

verify_data_preprocessing(train_dataset, "Train Dataset")
verify_data_preprocessing(val_dataset, "Validation Dataset")
verify_data_preprocessing(test_dataset, "Test Dataset")

### Позиционное кодирование

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

n, d = 2048, 512
pos_encoding = positional_encoding(n, d)
print(pos_encoding.shape)
pos_encoding = pos_encoding[0]

# Juggle the dimensions for the plot
pos_encoding = tf.reshape(pos_encoding, (n, d//2, 2))
pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
pos_encoding = tf.reshape(pos_encoding, (d, n))

### Маскировка

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    if len(seq.shape) == 1:
        seq = tf.expand_dims(seq, 0)  
    return seq[:, tf.newaxis, tf.newaxis, :]  
 

x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
create_padding_mask(x)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask 

x = tf.random.uniform((1, 3))
temp = create_look_ahead_mask(tf.shape(x)[1])

### Внимание к масштабируемому точечному произведению

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True) 

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)


  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  

  output = tf.matmul(attention_weights, v) 

  return output, attention_weights

def print_out(q, k, v):
  temp_out, temp_attn = scaled_dot_product_attention(
      q, k, v, None)
  print('Attention weights are:')
  print(temp_attn)
  print('Output is:')
  print(temp_out)

np.set_printoptions(suppress=True)

temp_k = tf.constant([[10, 0, 0],
                      [0, 10, 0],
                      [0, 0, 10],
                      [0, 0, 10]], dtype=tf.float32)  # (4, 3)

temp_v = tf.constant([[1, 0],
                      [10, 0],
                      [100, 5],
                      [1000, 6]], dtype=tf.float32)  # (4, 2)

# This `query` aligns with the second `key`,
# so the second `value` is returned.
temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)

# This query aligns with a repeated key (third and fourth),
# so all associated values get averaged.
temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)

# This query aligns equally with the first and second key,
# so their values get averaged.
temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)

temp_q = tf.constant([[0, 0, 10],
                      [0, 10, 0],
                      [10, 10, 0]], dtype=tf.float32)  # (3, 3)
print_out(temp_q, temp_k, temp_v)


### Многоголовое внимание 

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  
    k = self.wk(k) 
    v = self.wv(v)  

    q = self.split_heads(q, batch_size) 
    k = self.split_heads(k, batch_size) 
    v = self.split_heads(v, batch_size) 

    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model)) 

    output = self.dense(concat_attention) 

    return output, attention_weights

temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60, 512)) 
out, attn = temp_mha(y, k=y, q=y, mask=None)
out.shape, attn.shape



### Сеть точечной прямой связи

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

sample_ffn = point_wise_feed_forward_network(512, 2048)
sample_ffn(tf.random.uniform((64, 50, 512))).shape


### Кодер и декодер

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2

### Слой декодера

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  

    ffn_output = self.ffn(out2)  
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  

    return out3, attn_weights_block1, attn_weights_block2


### Кодер

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding,
                                            self.d_model)

    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    seq_len = tf.shape(x)[1]

    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    return x  # (batch_size, input_seq_len, d_model)

### Декодер

class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}

    x = self.embedding(x)  
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)

      attention_weights[f'decoder_layer{i+1}_block1'] = block1
      attention_weights[f'decoder_layer{i+1}_block2'] = block2

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
    pred = tf.cast(tf.argmax(pred, axis=2), tf.int32)  # Приведение к типу int32
    accuracies = tf.equal(real, pred)

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

def check_preprocessing(sentences, tokenizer, dataset_name):
    sequences = encode_sentences(sentences, tokenizer)
    if len(sequences.shape) != 2:
        raise ValueError(f"{dataset_name} preprocessing resulted in non-2D data")
    print(f"{dataset_name} preprocessing check passed")

check_preprocessing(en_sentences_train, tokenizer_en, "English Training Data")
check_preprocessing(ru_sentences_train, tokenizer_ru, "Russian Training Data")


def check_preprocessing(sentences, tokenizer, dataset_name):
    sequences = encode_sentences(sentences, tokenizer)
    if len(sequences.shape) != 2:
        raise ValueError(f"{dataset_name} preprocessing resulted in non-2D data")
    print(f"{dataset_name} preprocessing check passed")

check_preprocessing(en_sentences_val, tokenizer_en, "English Val Data")
check_preprocessing(ru_sentences_val, tokenizer_ru, "Russian Val Data")

def check_preprocessing(sentences, tokenizer, dataset_name):
    sequences = encode_sentences(sentences, tokenizer)
    if len(sequences.shape) != 2:
        raise ValueError(f"{dataset_name} preprocessing resulted in non-2D data")
    print(f"{dataset_name} preprocessing check passed")

check_preprocessing(en_sentences_test, tokenizer_en, "English Test Data")
check_preprocessing(ru_sentences_test, tokenizer_ru, "Russian Test Data")

### Создайте Трансформера

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)
        
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs, training=False, mask=None):
      # Разделяем входные данные на две части: inp и tar
      inp, tar = inputs[0], inputs[1]

      print("Shape of inp:", inp.shape)
      print("Shape of tar:", tar.shape)
      if len(inp.shape) != 2 or len(tar.shape) != 2:
          raise ValueError("Input and target must be 2-dimensional")

        # Внутренние маски для кодировщика и декодировщика
      enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)
        
        # Пропускаем через кодировщик
      enc_output = self.encoder(inp, training, enc_padding_mask)
        
        # Пропускаем через декодировщик
      dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        
        # Применяем последний слой
      final_output = self.final_layer(dec_output)
        
      return final_output, attention_weights

  def create_masks(self, inp, tar):
    print("Creating masks for inp of shape:", inp.shape)
    print("Creating masks for tar of shape:", tar.shape)
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, look_ahead_mask, dec_padding_mask
temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

def check_dataset_shapes(dataset, dataset_name):
    for (batch, (inp, tar)) in enumerate(dataset.take(1)):
        if len(inp.shape) != 2 or len(tar.shape) != 2:
            raise ValueError(f"{dataset_name} data is not 2-dimensional")
        print(f"{dataset_name} - Shape of inp:", inp.shape)
        print(f"{dataset_name} - Shape of tar:", tar.shape)

check_dataset_shapes(train_dataset, "Train Dataset")
check_dataset_shapes(val_dataset, "Validation Dataset")
check_dataset_shapes(test_dataset, "Test Dataset")

### Установить гиперпараметры

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.3
input_vocab_size = tokenizer_en.vocab_size + 2
target_vocab_size = tokenizer_ru.vocab_size + 2

### Оптимизатор

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)  # Приведение step к типу float32
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
                                     
temp_learning_rate_schedule = CustomSchedule(d_model)    


### Потери и показател
                                     
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')                                   

                                     
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy') 
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')
                                     
### Обучение и контрольно-пропускной пункт 
                               

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]

def ensure_correct_data_shapes(dataset, expected_shape):
    for (inp, tar) in dataset.take(1):  # Проверяем только один батч
        assert inp.shape[1] == expected_shape and tar.shape[1] == expected_shape, "Некорректная размерность данных"
        print(f"Размерности данных в порядке: {inp.shape}, {tar.shape}")

# Вызовите эту функцию для каждого набора данных перед тренировкой
ensure_correct_data_shapes(train_dataset, max_length)
ensure_correct_data_shapes(val_dataset, max_length)
ensure_correct_data_shapes(test_dataset, max_length)

def check_dataset_shapes(dataset, dataset_name):
    for (batch, (inp, tar)) in enumerate(dataset.take(1)):
        print(f"{dataset_name} - Shape of inp:", inp.shape)
        print(f"{dataset_name} - Shape of tar:", tar.shape)
        assert len(inp.shape) == 2 and len(tar.shape) == 2, "Dataset shapes are not 2-dimensional"

check_dataset_shapes(train_dataset, "Train Dataset")
check_dataset_shapes(val_dataset, "Validation Dataset")
check_dataset_shapes(test_dataset, "Test Dataset")

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    with tf.GradientTape() as tape:
        predictions, _ = transformer([inp, tar_inp], training=True)
        loss = loss_function(tar_real, predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))

@tf.function(input_signature=train_step_signature)
def evaluate_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    predictions, _ = transformer([inp, tar_inp], training=False)
    loss = loss_function(tar_real, predictions)
    return loss, predictions

def check_train_evaluate_functions():
    for (inp, tar) in train_dataset.take(1):  # Проверяем только один батч
        try:
            train_step(inp, tar)
            evaluate_step(inp, tar)
            print("Функции train_step и evaluate_step работают корректно.")
        except Exception as e:
            print("Ошибка в функциях train_step/evaluate_step:", e)
            break

check_train_evaluate_functions()


def check_data_format(dataset, dataset_name):
    for (batch, (inp, tar)) in enumerate(dataset.take(1)):
        if len(inp.shape) != 2 or len(tar.shape) != 2:
            raise ValueError(f"{dataset_name} data is not 2-dimensional")
        print(f"{dataset_name} - Shape of inp:", inp.shape)
        print(f"{dataset_name} - Shape of tar:", tar.shape)

check_data_format(train_dataset, "Train Dataset")
check_data_format(val_dataset, "Validation Dataset")
check_data_format(test_dataset, "Test Dataset")


# Создание экземпляра модели
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    pe_input=max_length,
    pe_target=max_length,
    rate=dropout_rate
)

# Проверка, что transformer создан и имеет ожидаемые свойства
assert transformer, "Transformer не создан"
assert hasattr(transformer, 'call'), "Transformer не имеет метода call"

# Компиляция модели
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate)
transformer.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy_function])

# Подготовка к тренировке
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    with tf.GradientTape() as tape:
        predictions, _ = transformer([inp, tar_inp], training=True)
        loss = loss_function(tar_real, predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))

# Проверка train_step
for (inp, tar) in train_dataset.take(1):
    try:
        train_step(inp, tar)
        print("Функция train_step работает корректно.")
    except Exception as e:
        print("Ошибка в функции train_step:", e)

for (inp, tar) in val_dataset.take(1):
    try:
        train_step(inp, tar)
        print("Функция val_step работает корректно.")
    except Exception as e:
        print("Ошибка в функции val_step:", e)

# Тренировочный и валидационный циклы
EPOCHS = 12
for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)

    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

# Тестирование после завершения всех эпох
test_loss.reset_states()
test_accuracy.reset_states()

print("Checking shapes of data in test_dataset...")
for (batch, (inp, tar)) in enumerate(test_dataset.take(1)):
    print("Shape of inp in test dataset:", inp.shape)
    print("Shape of tar in test dataset:", tar.shape)
    loss, _ = evaluate_step(inp, tar)
    test_loss(loss)
    test_accuracy(accuracy_function(tar[:, 1:], _))

print(f'Test Loss {test_loss.result():.4f} Accuracy {test_accuracy.result():.4f}')
