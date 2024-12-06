import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#----------------------CHỨC NĂNG--------------------------#
# Hàm tải lại các trọng số và từ điển
def load():
    W1 = np.load("W1.npy")
    W2 = np.load("W2.npy")
    with open("vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return W1, W2, vocab

# Hàm tính cosine similarity giữa các vector nhúng
def compute_cosine_similarity(slotA, slotB, vocab, W1):
    if slotA not in vocab or slotB not in vocab:
        return None

    idx1, idx2 = vocab[slotA], vocab[slotB]
    vec1, vec2 = W1[idx1], W1[idx2]

    # Tính cosine similarity bằng numpy
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

# Hàm tạo heatmap cho các từ
def plot_similarity_heatmap(word1, word2, vocab, W1):
    # Tạo ma trận tương đồng
    similarity_matrix = np.zeros((len(word1), len(word2)))

    for i, feature1 in enumerate(word1):
        for j, feature2 in enumerate(word2):
            try:
                similarity = compute_cosine_similarity(feature1, feature2, vocab, W1)
                similarity_matrix[i, j] = similarity if similarity is not None else 0
            except:
                similarity_matrix[i, j] = 0

    # Chuyển ma trận thành DataFrame để trực quan hóa
    df_similarity = pd.DataFrame(similarity_matrix, index=word1, columns=word2)

    # Vẽ heatmap
    plt.figure(figsize=(12,8))
    sns.heatmap(df_similarity,annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    sns.set(font_scale=1)
    plt.title("Cosine Similarity Heatmap")
    plt.xlabel("Từ 2")
    plt.ylabel("Từ 1")
    plt.tight_layout()
    plt.savefig("Cosine Similarity Heatmap.png")
    print("Đã lưu heatmap cosine similarity")
    plt.close()

#----------------------THỰC HIỆN--------------------------#
# Gọi hàm tải mô hình
W1, W2, vocab = load()

# Kiểm tra cosin của các cặp từ giữa word1 (context) và word2 (center)
word1 = ["tích_cực", "ăn_uống", "giáo_dục"]
word2 = ["tiêu_cực", "học", "lạc_quan"]
if word1.__sizeof__() < word2.__sizeof__():
    print("Danh sách 1 phải có số lượng từ nhiều hơn hoặc bằng số lượng từ của danh sách 2!!!")
    exit()
for feature1 in word1:
    for feature2 in word2:
        try:
            var = compute_cosine_similarity(feature1,feature2,vocab,W1)
            print(f'Độ tương đồng <{feature1}>|<{feature2}>: {var:.2f}')
        except:
            print(f"Không tìm thấy một trong các từ '{feature1}' hoặc '{feature2}' trong từ điển!")

# Gọi hàm lưu heatmap
plot_similarity_heatmap(word1, word2, vocab, W1)