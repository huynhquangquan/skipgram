import numpy as np
import matplotlib.pyplot as plt
from pyvi import ViTokenizer
import pandas as pd
import seaborn as sns
import openpyxl
import json

np.random.seed(42) # Cố định kết quả mỗi lần chạy

#-------------------TẠO HÀM/CHỨC NĂNG------------------#
# Hàm tiền xử lý văn bản
def preprocess_text(text, custom_phrases):
    text = text.lower()  # Chuyển thành chữ thường
    cleaned_text = ''.join(char for char in text if char.isalnum() or char.isspace())  # Loại bỏ ký tự đặc biệt
    words = ViTokenizer.tokenize(cleaned_text).split()  # Tách từ tiếng Việt
    # Nối cụm từ dựa trên danh sách custom_phrases
    for phrase in custom_phrases:
        phrase_words = phrase.split()  # Tách cụm từ thành danh sách các từ
        i = 0
        while i <= len(words) - len(phrase_words):
            if words[i:i + len(phrase_words)] == phrase_words:
                # Nối các từ trong cụm phrase thành một từ duy nhất
                words[i:i + len(phrase_words)] = ['_'.join(phrase_words)]
            else:
                i += 1
    return words

# Stopword
def remove_stopwords(text, stopwords):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return " ".join(filtered_words)

# Hàm xây dựng từ điển từ vựng
def build_vocab(sentences):
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

# Hàm tính softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Hàm tính cross-entropy loss
def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9))

# Hàm tạo cặp từ trung tâm và ngữ cảnh
def generate_training_data(sentences, vocab, window_size):
    data = []
    for sentence in sentences:
        sentence_len = len(sentence)
        for idx, word in enumerate(sentence):
            word_idx = vocab[word]
            context_indices = list(range(max(0, idx - window_size), min(sentence_len, idx + window_size + 1)))
            context_indices.remove(idx)
            for context_idx in context_indices:
                context_word = sentence[context_idx]
                context_word_idx = vocab[context_word]
                data.append((word_idx, context_word_idx))
    return data

# Hàm huấn luyện Skip-gram
def train_skipgram(sentences, vocab, epochs, learning_rate):
    window_size = 8
    vocab_size = len(vocab)
    embedding_dim = 10
    batch_size = 64
    print(f'Batch_size: {batch_size}\nEmbedding_dim: {embedding_dim}\n')

    # Khởi tạo trọng số
    W1 = np.random.rand(vocab_size, embedding_dim)
    W2 = np.random.rand(embedding_dim, vocab_size)

    # Chuẩn bị dữ liệu huấn luyện
    training_data = generate_training_data(sentences, vocab, window_size)

    # Theo dõi mất mát qua các epoch
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0

        # Chia dữ liệu thành các batch
        np.random.shuffle(training_data)
        batches = [training_data[i:i + batch_size] for i in range(0, len(training_data), batch_size)]

        for batch in batches:
            # Khởi tạo gradient tổng cho mỗi batch
            grad_W1 = np.zeros_like(W1)
            grad_W2 = np.zeros_like(W2)

            for word_idx, context_word_idx in batch:
                # Forward pass
                h = W1[word_idx]  # Nhúng từ trung tâm
                u = np.dot(W2.T, h)  # Ma trận đầu ra
                y_pred = softmax(u)

                # Tạo vector nhãn one-hot
                y_true = np.zeros(vocab_size)
                y_true[context_word_idx] = 1

                # Tính loss
                loss = cross_entropy_loss(y_true, y_pred)
                total_loss += loss

                # Backpropagation
                e = y_pred - y_true
                grad_W2 += np.outer(h, e)
                grad_W1[word_idx] += np.dot(W2, e)

            # Cập nhật trọng số sau mỗi batch
            W2 -= learning_rate * grad_W2
            W1 -= learning_rate * grad_W1

        # Ghi nhận mất mát
        loss_history.append(total_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")
    print("Huấn luyện hoàn tất!\n")
    return W1, W2, loss_history

# Hàm lưu mô hình huấn luyện
def save(W1, W2, vocab):
    # Lưu các ma trận trọng số W1, W2
    np.save("W1.npy", W1)
    np.save("W2.npy", W2)

    # Lưu từ điển vocab dưới dạng JSON
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

# Hàm hiển thị mất mát theo thời gian
def plot_loss(loss_history):
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Độ mất mát qua số Epochs")
    plt.savefig("Loss Over Epochs.jpg")
    plt.close()

#-----------------THỰC HIỆN-------------------#
stopwords = [
    "là", "và", "của", "có", "trong", "với", "được", "cho", "như", "cũng", "khi", "ở", "này", "ra", "lại", "về", "đã",
    "những", "rằng", "đó", "nên", "còn", "đi", "đây", "nhiều", "mà", "thì", "hoặc", "hay", "gì", "ai", "sao", "thế", "tại", "nào", "chỉ",
    "cái", "vì", "đâu", "vẫn", "đều", "nếu", "như", "vậy", "này", "ấy", "trên", "dưới", "từ", "tới", "hơn", "cả", "ít", "nhiều", "nữa",
    "nào", "vừa", "mỗi", "một", "mọi", "không", "rất"
]

# Từ vựng tùy chỉnh
custom = open("custom_dict.txt","r",encoding="utf-8")
custom_read = custom.read()
custom.close()
custom_phrases = []
for phrase in custom_read.splitlines():
    custom_phrases.append(phrase)

# Dữ liệu mẫu
file = open("train.txt", "r", encoding='utf-8')
texts = file.read()

# Tiền xử lý dữ liệu
# Stopword cho dữ liệu
texts = remove_stopwords(texts,stopwords)

# Tách chuỗi dựa trên nhiều dấu ngắt câu
separators = [".", "?", "!"]
for separator in separators:
    texts = texts.replace(separator, "|")  # Thay thế các ký tự phân cách bằng "|"
processed_text = [preprocess_text(text,custom_phrases) for text in texts.split("|")]
vocab = build_vocab(processed_text)
print(f'Số lượng từ: {len(vocab)}')

# Huấn luyện mô hình
W1, W2, loss_history = train_skipgram(processed_text, vocab, epochs=10, learning_rate=0.01)

# Lưu mô hình
save(W1, W2, vocab)

# Hiển thị và lưu kết quả mất mát
plot_loss(loss_history)