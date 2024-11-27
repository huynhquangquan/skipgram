import numpy as np
import matplotlib.pyplot as plt
from pyvi import ViTokenizer
import pandas as pd
import seaborn as sns
import openpyxl
np.random.seed(42) # Cố định kết quả mỗi lần chạy
#-------------------TẠO HÀM/CHỨC NĂNG------------------#
# Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()  # Chuyển thành chữ thường
    cleaned_text = ''.join(char for char in text if char.isalnum() or char.isspace())  # Loại bỏ ký tự đặc biệt
    words = ViTokenizer.tokenize(cleaned_text).split()  # Tách từ tiếng Việt
    return words

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


# Hàm huấn luyện Skip-gram
def train_skipgram(sentences, vocab, epochs, learning_rate):
    window_size = 2
    vocab_size = len(vocab)
    embedding_dim = 100

    # Khởi tạo trọng số
    W1 = np.random.rand(vocab_size, embedding_dim)
    W2 = np.random.rand(embedding_dim, vocab_size)

    # Theo dõi mất mát qua các epoch
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            for idx, word in enumerate(sentence):
                word_idx = vocab[word]

                # Lấy ngữ cảnh xung quanh từ trung tâm
                context_indices = list(range(max(0, idx - window_size), min(sentence_len, idx + window_size + 1)))
                context_indices.remove(idx)

                for context_idx in context_indices:
                    context_word = sentence[context_idx]
                    context_word_idx = vocab[context_word]

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
                    W2 -= learning_rate * np.outer(h, e)
                    W1[word_idx] -= learning_rate * np.dot(W2, e)

        # Ghi nhận mất mát
        loss_history.append(total_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")
    print("Huấn luyện hoàn tất!\n")
    return W1, W2, loss_history

# Hàm dự đoán ngữ cảnh từ từ trung tâm
def predict_context(center_word, vocab, W1, W2, top_n):
    if center_word not in vocab:
        print(f"Từ '{center_word}' không có trong từ điển!")
        return []

    word_idx = vocab[center_word]
    h = W1[word_idx]  # Vector nhúng của từ trung tâm
    u = np.dot(W2.T, h)  # Tính xác suất
    y_pred = softmax(u)

    # Lấy top_n từ có xác suất cao nhất
    context_indices = np.argsort(y_pred)[::-1][:top_n]
    context_words = [word for word, idx in vocab.items() if idx in context_indices]
    return context_words

# Hàm tính cosine similarity giữa các vector nhúng
def compute_cosine_similarity(word1, word2, vocab, W1):
    if word1 not in vocab or word2 not in vocab:
        print(f"Không tìm thấy một trong các từ '{word1}' hoặc '{word2}' trong từ điển!")
        return None

    idx1, idx2 = vocab[word1], vocab[word2]
    vec1, vec2 = W1[idx1], W1[idx2]

    # Tính cosine similarity bằng numpy
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

# Hàm tính toán ma trận cosine similarity giữa tất cả các cặp từ
def compute_similarity_matrix(vocab, W1):
    similarity_matrix = np.zeros((len(vocab), len(vocab)))
    words = list(vocab.keys())

    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            if i <= j:  # Chỉ tính cho phần trên của ma trận (symmetric matrix)
                similarity = compute_cosine_similarity(word1, word2, vocab, W1)
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity  # Ma trận đối xứng

    # Chuyển ma trận similarity thành DataFrame để dễ dàng hiển thị
    similarity_df = pd.DataFrame(similarity_matrix, index=words, columns=words)
    return similarity_df

# Hàm hiển thị mất mát theo thời gian
def plot_loss(loss_history):
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Độ mất mát qua số Epochs")
    plt.savefig("evaluate_result\\Loss Over Epochs.jpg")
    plt.close()

# Hàm hiển thị mối quan hệ tương đồng giữa các từ
def plot_cosine_simularity(similarity_df):
    plt.figure(figsize=(25, 10))
    sns.set(font_scale=0.8)
    sns.heatmap(similarity_df, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Heatmap Cosine Similarity giữa các từ")
    plt.xlabel("Từ")
    plt.ylabel("Từ")
    plt.savefig("evaluate_result\\Cosine Similarity Matrix.png")
    plt.close()
    similarity_df.to_excel("evaluate_result\\Cosine Similarity Matrix.xlsx")

# Hàm giảm chiều dữ liệu vector nhúng
def pca_reduce(W1, n_components):
    """
    Giảm chiều dữ liệu vector nhúng bằng PCA sử dụng NumPy.

    Args:
        W1 (numpy.ndarray): Ma trận vector nhúng (vocab_size x embedding_dim).
        n_components (int): Số chiều đầu ra (thường là 2).

    Returns:
        reduced_embeddings (numpy.ndarray): Ma trận vector nhúng giảm chiều (vocab_size x n_components).
    """
    # Chuẩn hóa ma trận nhúng (trung bình = 0)
    W1_mean = np.mean(W1, axis=0)
    W1_centered = W1 - W1_mean

    # Tính ma trận hiệp phương sai
    covariance_matrix = np.cov(W1_centered, rowvar=False)

    # Tính giá trị riêng và vector riêng
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sắp xếp giá trị riêng theo thứ tự giảm dần
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]

    # Dựng lại dữ liệu trong không gian mới
    reduced_embeddings = np.dot(W1_centered, top_eigenvectors)
    reduced_embeddings = np.real(reduced_embeddings)
    return reduced_embeddings

# Hàm hiển thị vector nhúng
def visualize_embeddings(vocab, W1):
    """
    Trực quan hóa vector nhúng sau khi giảm chiều bằng PCA.

    Args:
        vocab (dict): Từ điển từ vựng (word -> index).
        W1 (numpy.ndarray): Ma trận vector nhúng (vocab_size x embedding_dim).
    """
    reduced_embeddings = pca_reduce(W1, n_components=2)  # Giảm chiều xuống 2D

    # Vẽ biểu đồ scatter
    plt.figure(figsize=(10, 9))
    for word, idx in vocab.items():
        x, y = reduced_embeddings[idx]
        plt.scatter(x, y, marker='o', color='blue')
        plt.text(x + 0.02, y + 0.02, word, fontsize=8)

    plt.title("Trực quan hóa vector nhúng")
    plt.xlabel("PCA Thành Phần 1")
    plt.ylabel("PCA Thành Phần 2")
    plt.grid(True)
    plt.savefig("evaluate_result\\Word Embeddings Visualization.png")
    plt.close()

#-----------------THỰC HIỆN-------------------#

# Dữ liệu mẫu
file = open("text.txt", "r", encoding='utf-8')
texts = file.read()
file.close()

# Tiền xử lý dữ liệu
processed_texts = [preprocess_text(text) for text in texts.splitlines()]
vocab = build_vocab(processed_texts)

# Huấn luyện mô hình
W1, W2, loss_history = train_skipgram(processed_texts, vocab, epochs=100, learning_rate=0.01)

# Hiển thị và lưu kết quả mất mát
plot_loss(loss_history)

# Từ điển dự đoán và ma trận nhúng
for center_word in vocab.keys():
    predicted_context = predict_context(center_word, vocab, W1, W2, top_n=1)
    print(f"Ngữ cảnh dự đoán cho từ '{center_word}': {predicted_context}")

# Tính ma trận cosine similarity
similarity_df = compute_similarity_matrix(vocab, W1)

# Hiển thị và lưu kết quả ma trận tương đồng
plot_cosine_simularity(similarity_df)

# Gọi hàm hiển thị và lưu lưu trực quan hóa vector nhúng
visualize_embeddings(vocab, W1)