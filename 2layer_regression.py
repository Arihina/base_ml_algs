import numpy as np


# Генерация выборки
def true_function(x):
    """Истинная зависимость, которую сеть должна восстановить."""
    return np.sin(3 * x) + 0.5 * x


def make_data(n_samples=300, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (n_samples, 1))  # (N, 1)
    y = true_function(X) + rng.normal(0, noise, (n_samples, 1))  # (N, 1)
    return X, y


# Инициализация параметров
def init_params(n_in=1, n_hidden=32, n_out=1, seed=1):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, 1 / np.sqrt(n_in), (n_in, n_hidden)),
        "b1": np.zeros((1, n_hidden)),
        "W2": rng.normal(0, 1 / np.sqrt(n_hidden), (n_hidden, n_out)),
        "b2": np.zeros((1, n_out)),
    }


# forward
def forward(X, p):
    Z1 = X @ p["W1"] + p["b1"]  # (N, H)
    A1 = np.tanh(Z1)  # (N, H)
    Y_hat = A1 @ p["W2"] + p["b2"]  # (N, 1)  выход БЕЗ активации (линейный)
    cache = (X, A1)
    return Y_hat, cache


# MSE (функция ошибки)
def mse_loss(Y_hat, y):
    return np.mean((Y_hat - y) ** 2)


# Градиент (обратное распространение ошибки, backward)
def backward(cache, Y_hat, y, p):
    X, A1 = cache
    N = X.shape[0]

    dZ2 = 2 * (Y_hat - y) / N  # (N, 1)  dL/dY_hat
    dW2 = A1.T @ dZ2  # (H, 1)
    db2 = dZ2.sum(axis=0, keepdims=True)  # (1, 1)

    dA1 = dZ2 @ p["W2"].T  # (N, H)
    dZ1 = dA1 * (1 - A1 ** 2)  # (N, H)  производная tanh
    dW1 = X.T @ dZ1  # (1, H)
    db1 = dZ1.sum(axis=0, keepdims=True)  # (1, H)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# Численная проверка градиента
def grad_check(X, y, p, h=1e-5):
    Y_hat, cache = forward(X, p)
    grads = backward(cache, Y_hat, y, p)
    for name in p:
        num = np.zeros_like(p[name])
        it = np.nditer(p[name], flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[name][i]
            p[name][i] = old + h;
            Lp = mse_loss(forward(X, p)[0], y)
            p[name][i] = old - h;
            Lm = mse_loss(forward(X, p)[0], y)
            p[name][i] = old
            num[i] = (Lp - Lm) / (2 * h)
        a = grads[name]
        rel = np.linalg.norm(a - num) / (np.linalg.norm(a) + np.linalg.norm(num) + 1e-12)
        print(f"  {name}: относительная ошибка = {rel:.2e}")


# Обучение (градиентный спуск)
def r2_score(Y_hat, y):
    """Коэффициент детерминации: 1 — идеально, 0 — не лучше среднего."""
    ss_res = np.sum((y - Y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def train(X, y, p, lr=0.05, epochs=10000, log_every=2000):
    for ep in range(epochs + 1):
        Y_hat, cache = forward(X, p)
        grads = backward(cache, Y_hat, y, p)
        for k in p:
            p[k] -= lr * grads[k]
        if ep % log_every == 0:
            print(f"эпоха {ep:5d} | MSE = {mse_loss(Y_hat, y):.4f} | R2 = {r2_score(Y_hat, y):.3f}")
    return p


if __name__ == "__main__":
    X, y = make_data(300, noise=0.1)
    split = 225
    X_tr, y_tr, X_te, y_te = X[:split], y[:split], X[split:], y[split:]

    params = init_params(n_hidden=32)

    print("Проверка градиента (должно быть ~1e-8 и меньше):")
    grad_check(X_tr[:20], y_tr[:20], params)

    print("\nОбучение:")
    params = train(X_tr, y_tr, params)

    Y_te, _ = forward(X_te, params)
    print(f"\nТест: MSE = {mse_loss(Y_te, y_te):.4f}, R2 = {r2_score(Y_te, y_te):.3f}")
    print("(MSE шума = 0.1^2 = 0.01 — ниже этого опуститься нельзя)")
