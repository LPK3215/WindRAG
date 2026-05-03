"""
自适应检索融合模块
根据查询特征动态预测最优α权重，替代固定α=0.7

方法：多层感知机（MLP）回归器
输入：17维查询特征向量
输出：最优α值（0.1~0.9）
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import pickle
from query_features import QueryFeatureExtractor


class AdaptiveFusionModel:
    """自适应融合权重预测模型"""

    def __init__(self, model_dir: str = "data/adaptive_model"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.feature_extractor = QueryFeatureExtractor()

        # MLP模型（使用numpy手动实现，避免深度学习框架依赖）
        self.weights1 = None  # 输入层→隐藏层1
        self.bias1 = None
        self.weights2 = None  # 隐藏层1→隐藏层2
        self.bias2 = None
        self.weights3 = None  # 隐藏层2→输出层
        self.bias3 = None

        self.is_trained = False
        self.feature_mean = None  # 特征归一化参数
        self.feature_std = None

    def _init_weights(self, input_dim: int, hidden1: int = 64, hidden2: int = 32):
        """Xavier初始化"""
        np.random.seed(42)
        scale1 = np.sqrt(2.0 / input_dim)
        self.weights1 = np.random.randn(input_dim, hidden1) * scale1
        self.bias1 = np.zeros(hidden1)

        scale2 = np.sqrt(2.0 / hidden1)
        self.weights2 = np.random.randn(hidden1, hidden2) * scale2
        self.bias2 = np.zeros(hidden2)

        scale3 = np.sqrt(2.0 / hidden2)
        self.weights3 = np.random.randn(hidden2, 1) * scale3
        self.bias3 = np.zeros(1)

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播"""
        h1 = self._relu(X @ self.weights1 + self.bias1)
        h2 = self._relu(h1 @ self.weights2 + self.bias2)
        # 输出层用sigmoid映射到[0,1]，再缩放到[0.1, 0.9]
        out = self._sigmoid(h2 @ self.weights3 + self.bias3)
        return 0.1 + 0.8 * out  # 映射到[0.1, 0.9]

    def _forward_with_cache(self, X: np.ndarray):
        """前向传播（带中间结果缓存，用于反向传播）"""
        z1 = X @ self.weights1 + self.bias1
        h1 = self._relu(z1)
        z2 = h1 @ self.weights2 + self.bias2
        h2 = self._relu(z2)
        z3 = h2 @ self.weights3 + self.bias3
        out = self._sigmoid(z3)
        alpha_pred = 0.1 + 0.8 * out

        cache = {"X": X, "z1": z1, "h1": h1, "z2": z2, "h2": h2, "z3": z3, "out": out}
        return alpha_pred, cache

    def _backward(self, y: np.ndarray, cache: Dict, lr: float = 0.001):
        """反向传播（MSE损失）"""
        m = y.shape[0]
        alpha_pred = 0.1 + 0.8 * cache["out"]

        # 损失对alpha_pred的梯度
        d_alpha = (alpha_pred - y) / m  # (m, 1)

        # sigmoid'(z3) * 0.8
        d_out = d_alpha * 0.8 * cache["out"] * (1 - cache["out"])  # (m, 1)

        # 输出层梯度
        dW3 = cache["h2"].T @ d_out  # (hidden2, 1)
        db3 = np.sum(d_out, axis=0)  # (1,)

        # 隐藏层2梯度
        d_h2 = d_out @ self.weights3.T  # (m, hidden2)
        d_z2 = d_h2 * (cache["z2"] > 0).astype(float)  # ReLU'

        dW2 = cache["h1"].T @ d_z2
        db2 = np.sum(d_z2, axis=0)

        # 隐藏层1梯度
        d_h1 = d_z2 @ self.weights2.T
        d_z1 = d_h1 * (cache["z1"] > 0).astype(float)

        dW1 = cache["X"].T @ d_z1
        db1 = np.sum(d_z1, axis=0)

        # 梯度裁剪
        for grad in [dW1, dW2, dW3]:
            np.clip(grad, -1.0, 1.0, out=grad)

        # 更新权重
        self.weights1 -= lr * dW1
        self.bias1 -= lr * db1
        self.weights2 -= lr * dW2
        self.bias2 -= lr * db2
        self.weights3 -= lr * dW3
        self.bias3 -= lr * db3

    def train(self, training_data_path: str, epochs: int = 500, lr: float = 0.001,
              batch_size: int = 16, val_ratio: float = 0.2):
        """训练自适应融合模型"""
        print("加载训练数据...")
        with open(training_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        X = np.array([d["features"] for d in data], dtype=np.float32)
        y = np.array([[d["optimal_alpha"]] for d in data], dtype=np.float32)

        # 特征归一化
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0)
        self.feature_std[self.feature_std == 0] = 1e-8
        X = (X - self.feature_mean) / self.feature_std

        # 划分训练/验证集
        n = len(X)
        indices = np.random.permutation(n)
        val_size = int(n * val_ratio)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # 初始化权重
        input_dim = X.shape[1]
        self._init_weights(input_dim, hidden1=64, hidden2=32)

        # 训练
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0

        print(f"开始训练: {epochs} epochs, lr={lr}, batch_size={batch_size}")
        print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}")

        # L2正则化系数
        l2_lambda = 0.001

        for epoch in range(epochs):
            # Mini-batch训练
            perm = np.random.permutation(len(X_train))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(X_train), batch_size):
                end = min(start + batch_size, len(X_train))
                batch_idx = perm[start:end]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                alpha_pred, cache = self._forward_with_cache(X_batch)
                batch_loss = np.mean((alpha_pred - y_batch) ** 2)
                epoch_loss += batch_loss
                n_batches += 1

                self._backward(y_batch, cache, lr=lr)

                # L2正则化
                self.weights1 *= (1 - l2_lambda * lr)
                self.weights2 *= (1 - l2_lambda * lr)
                self.weights3 *= (1 - l2_lambda * lr)

            epoch_loss /= n_batches

            # 验证
            val_pred = self._forward(X_val)
            val_loss = np.mean((val_pred - y_val) ** 2)

            # 学习率衰减
            if epoch > 0 and epoch % 100 == 0:
                lr *= 0.5

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1:3d}: train_loss={epoch_loss:.6f}, val_loss={val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_best_weights()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    self._restore_best_weights()
                    break

        self._restore_best_weights()
        self.is_trained = True

        # 最终评估
        train_pred = self._forward(X_train)
        val_pred = self._forward(X_val)

        print(f"\n训练完成!")
        print(f"  训练MSE: {np.mean((train_pred - y_train) ** 2):.6f}")
        print(f"  验证MSE: {np.mean((val_pred - y_val) ** 2):.6f}")
        print(f"  训练MAE: {np.mean(np.abs(train_pred - y_train)):.4f}")
        print(f"  验证MAE: {np.mean(np.abs(val_pred - y_val)):.4f}")

        # 保存模型
        self.save()
        return best_val_loss

    def predict_alpha(self, query: str, query_embedding: np.ndarray = None) -> float:
        """预测查询的最优α值（规则先验 + MLP微调）"""
        if not self.is_trained:
            return self._rule_based_alpha(query)

        # 先用规则计算先验α
        rule_alpha = self._rule_based_alpha(query)

        # MLP预测残差
        feature_vector = self.feature_extractor.extract_vector(query, query_embedding)
        X = (feature_vector - self.feature_mean) / self.feature_std
        X = X.reshape(1, -1)

        mlp_alpha = float(self._forward(X)[0][0])

        # 融合：0.6*规则先验 + 0.4*MLP（规则更可靠，MLP提供微调）
        alpha = 0.6 * rule_alpha + 0.4 * mlp_alpha
        return np.clip(alpha, 0.1, 0.9)

    @staticmethod
    def _rule_based_alpha(query: str) -> float:
        """基于查询特征的规则先验α值"""
        from query_features import DOMAIN_TERMS, QUESTION_TYPE_KEYWORDS

        # 计算领域术语命中数
        term_hits = [t for t in DOMAIN_TERMS if t in query]
        term_count = len(term_hits)

        # 计算问题类型
        type_scores = {}
        for qtype, keywords in QUESTION_TYPE_KEYWORDS.items():
            type_scores[qtype] = sum(1 for kw in keywords if kw in query)

        # 规则1：含标准编号(ISO/GB/IEC)→BM25精确匹配重要→低α
        import re
        if re.search(r'(ISO|GB|IEC|DIN)\s*\d+', query):
            return 0.3

        # 规则2：含数值规格→BM25重要→较低α
        if re.search(r'\d+\.?\d*\s*(mm|℃|°C|m/s|kW|MW|%|V|A|Ω)', query):
            return 0.4

        # 规则3：术语密集(≥3个)→BM25召回好→较低α
        if term_count >= 3:
            return 0.5

        # 规则4：故障诊断类→语义理解重要→较高α
        if type_scores.get("故障诊断", 0) > 0:
            return 0.7

        # 规则5：系统原理类→语义理解最重要→高α
        if type_scores.get("系统原理", 0) > 0:
            return 0.8

        # 规则6：性能优化类→语义理解→较高α
        if type_scores.get("性能优化", 0) > 0:
            return 0.75

        # 规则7：维护操作类→含具体术语→中等α
        if type_scores.get("维护操作", 0) > 0:
            return 0.6

        # 默认：中等偏高
        return 0.65

    def _save_best_weights(self):
        """保存最佳权重（临时）"""
        self._best_weights = {
            "w1": self.weights1.copy(), "b1": self.bias1.copy(),
            "w2": self.weights2.copy(), "b2": self.bias2.copy(),
            "w3": self.weights3.copy(), "b3": self.bias3.copy(),
        }

    def _restore_best_weights(self):
        """恢复最佳权重"""
        if hasattr(self, '_best_weights'):
            self.weights1 = self._best_weights["w1"]
            self.bias1 = self._best_weights["b1"]
            self.weights2 = self._best_weights["w2"]
            self.bias2 = self._best_weights["b2"]
            self.weights3 = self._best_weights["w3"]
            self.bias3 = self._best_weights["b3"]

    def save(self):
        """保存模型到磁盘"""
        model_path = self.model_dir / "model.pkl"
        model_data = {
            "weights1": self.weights1, "bias1": self.bias1,
            "weights2": self.weights2, "bias2": self.bias2,
            "weights3": self.weights3, "bias3": self.bias3,
            "feature_mean": self.feature_mean, "feature_std": self.feature_std,
            "is_trained": self.is_trained,
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"模型已保存到 {model_path}")

    def load(self):
        """从磁盘加载模型"""
        model_path = self.model_dir / "model.pkl"
        if not model_path.exists():
            print(f"模型文件不存在: {model_path}")
            return False

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.weights1 = model_data["weights1"]
        self.bias1 = model_data["bias1"]
        self.weights2 = model_data["weights2"]
        self.bias2 = model_data["bias2"]
        self.weights3 = model_data["weights3"]
        self.bias3 = model_data["bias3"]
        self.feature_mean = model_data["feature_mean"]
        self.feature_std = model_data["feature_std"]
        self.is_trained = model_data["is_trained"]

        print(f"模型已从 {model_path} 加载")
        return True


if __name__ == "__main__":
    model = AdaptiveFusionModel()

    # 训练
    training_path = "data/adaptive_fusion_training.json"
    if Path(training_path).exists():
        model.train(training_path, epochs=200, lr=0.005)

        # 测试预测
        test_queries = [
            "齿轮箱温度过高怎么处理？",
            "ISO VG 320齿轮油更换周期",
            "偏航系统的工作原理是什么？",
            "如何提高发电效率？",
        ]
        print("\n预测测试:")
        for q in test_queries:
            alpha = model.predict_alpha(q)
            print(f"  查询: {q} → α={alpha:.3f}")
    else:
        print(f"训练数据不存在: {training_path}")
        print("请先运行 generate_training_data.py")
