import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import TransformerMixin, BaseEstimator, ClusterMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_X_y


class NearestPrototypes(TransformerMixin, ClusterMixin, BaseEstimator):
    def __init__(self, n_prototypes_list=[3, 3], n_neighbors=5):
        # Définir une assertion pour contrôler que `n_prototypes_list`
        # et `n_neighbors` ont des valeurs cohérentes.
        assert(...)

        self.n_prototypes_list = n_prototypes_list
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        # Validation des entrées
        X, y = check_X_y(X, y)

        labels = np.unique(y)
        self.classes_ = labels
        assert(len(labels) == len(self.n_prototypes_list))
        assert(len(y) >= sum(self.n_prototypes_list))

        def prototypes(X, label, n_prototypes):
            """Sélectionne les individus d'étiquette `label` dans `X` et lance un
            algorithme des k-means pour calculer `n_prototypes`
            prototypes.
            """

            # Sélection du jeu de données d'étiquette `label`
            Xk = ...

            # Création d'un objet de classe `KMeans` avec le bon nombre
            # de prototypes
            cls = ...

            # Apprentissage des prototypes
            cls.fit(Xk)

            return cls.cluster_centers_

        # Concaténation de tous les prototypes pour toutes les
        # étiquettes et le nombre de prototypes correspondants.
        # Utiliser la fonction `prototypes` définies précédemment et
        # la fonction `np.concatenate`.
        self.prototypes_ = ...


        # Création des étiquettes pour tous les prototypes construits
        # précédemment. On pourra utiliser `np.repeat`.
        self.labels_ = ...


        # Création d'un objet KNeighborsClassifier
        self.nearest_prototypes_ = ...


        # Apprentissage du Knn sur les prototypes et leur étiquette
        self.nearest_prototypes_.fit(...)

    def predict(self, X):
        # Prédire les étiquettes en utilisant self.nearest_prototypes_
        return ...
