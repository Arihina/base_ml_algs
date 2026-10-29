import numpy as np


# Генерация выборки
def make_moons(n_samples=400, noise=0.15, seed=0):
    """Два «полумесяца» — классы, которые НЕЛЬЗЯ разделить прямой линией.
    Поэтому одного линейного слоя не хватит, и нужен скрытый слой."""
    rng = np.random.default_rng(seed)
    n = n_samples // 2
    t = rng.uniform(0, np.pi, n)

    # класс 0 — верхний полумесяц
    x0 = np.c_[np.cos(t), np.sin(t)]
    # класс 1 — нижний полумесяц, сдвинутый вправо и вверх
    x1 = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]

    X = np.vstack([x0, x1]) + rng.normal(0, noise, (2 * n, 2))
    y = np.r_[np.zeros(n), np.ones(n)].reshape(-1, 1)  # форма (N, 1)

    idx = rng.permutation(2 * n)  # перемешиваем
    return X[idx], y[idx]


# Инициализация параметров
def init_params(n_in=2, n_hidden=16, n_out=1, seed=1):
    rng = np.random.default_rng(seed)
    return {
        # маленькие случайные веса (масштаб по Xavier), чтобы нейроны были разными
        "W1": rng.normal(0, 1 / np.sqrt(n_in), (n_in, n_hidden)),
        "b1": np.zeros((1, n_hidden)),
        "W2": rng.normal(0, 1 / np.sqrt(n_hidden), (n_hidden, n_out)),
        "b2": np.zeros((1, n_out)),
    }


# forward
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def forward(X, p):
    Z1 = X @ p["W1"] + p["b1"]  # (N, H)  линейный слой 1
    A1 = np.tanh(Z1)  # (N, H)  нелинейность
    Z2 = A1 @ p["W2"] + p["b2"]  # (N, 1)  линейный слой 2
    P = sigmoid(Z2)  # (N, 1)  вероятность класса 1
    cache = (X, A1, P)  # то, что понадобится в backward
    return P, cache


# бинарная кросс-энтропия (функция ошибки)
def bce_loss(P, y, eps=1e-12):
    P = np.clip(P, eps, 1 - eps)  # защита от log(0)
    return -np.mean(y * np.log(P) + (1 - y) * np.log(1 - P))


# Градиент (обратное распространение ошибки, backward)
def backward(cache, y, p):
    X, A1, P = cache
    N = X.shape[0]

    dZ2 = (P - y) / N  # (N, 1)  dL/dZ2
    dW2 = A1.T @ dZ2  # (H, 1)
    db2 = dZ2.sum(axis=0, keepdims=True)  # (1, 1)

    dA1 = dZ2 @ p["W2"].T  # (N, H)
    dZ1 = dA1 * (1 - A1 ** 2)  # (N, H)  производная tanh
    dW1 = X.T @ dZ1  # (2, H)
    db1 = dZ1.sum(axis=0, keepdims=True)  # (1, H)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# Численная проверка градиента
def grad_check(X, y, p, h=1e-5):
    """Сравниваем аналитический градиент с (L(w+h) - L(w-h)) / 2h."""
    P, cache = forward(X, p)
    grads = backward(cache, y, p)
    worst = 0.0
    for name in p:
        num = np.zeros_like(p[name])
        it = np.nditer(p[name], flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[name][i]
            p[name][i] = old + h;
            Lp = bce_loss(forward(X, p)[0], y)
            p[name][i] = old - h;
            Lm = bce_loss(forward(X, p)[0], y)
            p[name][i] = old
            num[i] = (Lp - Lm) / (2 * h)
        a = grads[name]
        rel = np.linalg.norm(a - num) / (np.linalg.norm(a) + np.linalg.norm(num) + 1e-12)
        print(f"  {name}: относительная ошибка = {rel:.2e}")
        worst = max(worst, rel)
    return worst


# Обучение (градиентный спуск)
def accuracy(P, y):
    return np.mean((P > 0.5) == y)


def train(X, y, p, lr=1.0, epochs=3000, log_every=500):
    for ep in range(epochs + 1):
        P, cache = forward(X, p)
        grads = backward(cache, y, p)
        for k in p:
            p[k] -= lr * grads[k]  # шаг против градиента
        if ep % log_every == 0:
            print(f"эпоха {ep:5d} | loss = {bce_loss(P, y):.4f} | acc = {accuracy(P, y):.3f}")
    return p


if __name__ == "__main__":
    X, y = make_moons(400, noise=0.15)
    split = 300
    X_tr, y_tr, X_te, y_te = X[:split], y[:split], X[split:], y[split:]

    params = init_params(n_hidden=16)

    print("Проверка градиента (должно быть ~1e-8 и меньше):")
    grad_check(X_tr[:20], y_tr[:20], params)

    print("\nОбучение:")
    params = train(X_tr, y_tr, params)

    P_te, _ = forward(X_te, params)
    print(f"\nТест: loss = {bce_loss(P_te, y_te):.4f}, acc = {accuracy(P_te, y_te):.3f}")
