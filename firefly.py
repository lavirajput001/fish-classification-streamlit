# firefly.py
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def fitness_mask(mask, X, y):
    # mask: binary array of length d (which BoVW bins to keep)
    if mask.sum() == 0:
        return 0.0
    Xs = X[:, mask==1]
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    try:
        scores = cross_val_score(clf, Xs, y, cv=3, scoring='accuracy')
        return scores.mean()
    except Exception as e:
        return 0.0

def firefly_feature_select(X, y, n_fireflies=15, max_iter=20, alpha=0.2, beta0=1.0, gamma=1.0):
    d = X.shape[1]
    # initialize random binary fireflies
    pop = np.random.randint(0,2, size=(n_fireflies, d)).astype(np.int8)
    fitness = np.zeros(n_fireflies)
    for i in range(n_fireflies):
        fitness[i] = fitness_mask(pop[i], X, y)
    best_idx = np.argmax(fitness)
    best_mask = pop[best_idx].copy()
    best_score = fitness[best_idx]

    for t in range(max_iter):
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if fitness[j] > fitness[i]:
                    # distance between binary vectors
                    rij = np.linalg.norm(pop[i] - pop[j])
                    beta = beta0 * np.exp(-gamma * (rij**2))
                    # move i towards j probabilistically in binary space
                    prob = beta * (pop[j] - pop[i]) + alpha*(np.random.rand(d)-0.5)
                    # update: if prob > threshold -> 1 else 0
                    pop[i] = (prob > 0).astype(np.int8)
                    fitness[i] = fitness_mask(pop[i], X, y)
                    if fitness[i] > best_score:
                        best_score = fitness[i]
                        best_mask = pop[i].copy()
    return best_mask, best_score
