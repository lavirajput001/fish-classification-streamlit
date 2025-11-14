import numpy as np

def firefly_optimize(features, n_fireflies=15, iterations=20):
    """
    Binary Firefly Algorithm for feature selection
    """
    features = np.array(features, dtype=np.float32)
    dim = len(features)

    fireflies = np.random.randint(0, 2, (n_fireflies, dim))
    brightness = np.array([np.sum(ff * features) for ff in fireflies])

    gamma = 1
    beta0 = 2
    alpha = 0.2

    for _ in range(iterations):
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if brightness[j] > brightness[i]:
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta = beta0 * np.exp(-gamma * r**2)

                    fireflies[i] = fireflies[i] + beta * (fireflies[j] - fireflies[i]) + \
                                   alpha * (np.random.rand(dim) - 0.5)

                    fireflies[i] = (fireflies[i] > 0.5).astype(int)

        brightness = np.array([np.sum(ff * features) for ff in fireflies])

    best = fireflies[np.argmax(brightness)]
    optimized_features = features * best

    return optimized_features
