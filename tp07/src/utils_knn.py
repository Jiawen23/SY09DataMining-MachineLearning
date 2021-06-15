import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


def add_decision_boundary(
    model, levels=None, resolution=100, ax=None, label=None, color=None, region=True
):
    """Trace une frontière et des régions de décision sur une figure existante.

    La fonction requiert un modèle scikit-learn `model` pour prédire
    un score ou une classe. La discrétisation utilisée est fixée par
    l'argument `resolution`. Une (ou plusieurs frontières) sont
    ensuite tracées d'après le paramètre `levels` qui fixe la valeur
    des lignes de niveaux recherchées.

    """

    if ax is None:
        ax = plt.gca()

    # Create grid to evaluate model
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], resolution)
    yy = np.linspace(ylim[0], ylim[1], resolution)
    XX, YY = np.meshgrid(xx, yy)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.predict(xy).reshape(XX.shape)

    cat2num = {cat: num for num, cat in enumerate(model.classes_)}
    num2cat = {num: cat for num, cat in enumerate(model.classes_)}
    vcat2num = np.vectorize(lambda x: cat2num[x])
    Z_num = vcat2num(Z)

    # Add decision boundary to legend
    color = "red" if color is None else color
    sns.lineplot(x=[0], y=[0], label=label, ax=ax, color=color, linestyle="dashed")

    mask = np.zeros_like(Z_num, dtype=bool)
    for k in range(len(model.classes_) - 1):
        mask |= Z_num == k - 1
        Z_num_mask = np.ma.array(Z_num, mask=mask)
        ax.contour(
            XX,
            YY,
            Z_num_mask,
            levels=[k + 0.5],
            linestyles="dashed",
            corner_mask=True,
            colors=["red"],
            antialiased=True,
        )

    if region:
        # Hack to get colors
        # TODO use legend_out = True
        slabels = [str(l) for l in model.classes_]
        hdls, hlabels = ax.get_legend_handles_labels()
        hlabels_hdls = {l: h for l, h in zip(hlabels, hdls)}

        color_dict = {}
        for label in model.classes_:
            if str(label) in hlabels_hdls:
                hdl = hlabels_hdls[str(label)]
                color = hdl.get_facecolor().ravel()
                color_dict[label] = color
            else:
                raise Exception("No corresponding label found for ", label)

        colors = [color_dict[num2cat[i]] for i in range(len(model.classes_))]
        cmap = mpl.colors.ListedColormap(colors)

        ax.imshow(
            Z_num,
            interpolation="nearest",
            extent=ax.get_xlim() + ax.get_ylim(),
            aspect="auto",
            origin="lower",
            cmap=cmap,
            alpha=0.2,
        )
        
        
        
def plot_clustering(data, labels, markers=None, ax=None, **kwargs):

    #Affiche dans leur premier plan principal les données `data`,
    #colorée par `labels` avec éventuellement des symboles `markers`.

    if ax is None:
        ax = plt.gca()
    
    # Reduce to two dimensions
    if data.shape[1] == 2:
        data_pca = data.to_numpy()
    else:
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data)
    
    COLORS = np.array(['blue', 'green', 'red', 'purple', 'gray', 'cyan'])
    _, labels = np.unique(labels, return_inverse=True)
    colors = COLORS[labels]
    
    if markers is None:
        ax.scatter(*data_pca.T, c=colors)
    else:
        MARKERS = "o^sP*+xD"
    
        # Use integers
        markers_uniq, markers = np.unique(markers, return_inverse=True)
    
        for marker in range(len(markers_uniq)):
            data_pca_marker = data_pca[markers == marker, :]
            colors_marker = colors[markers == marker]
            ax.scatter(*data_pca_marker.T, c=colors_marker, marker=MARKERS[marker])
    
    if 'centers' in kwargs and 'covars' in kwargs:
        if data.shape[1] == 2:
            centers_2D = kwargs['centers']
            covars_2D = kwargs['covars']
        else:
            centers_2D = pca.transform(kwargs["centers"])
            covars_2D = [
                pca.components_ @ c @ pca.components_.T
                for c in kwargs['covars']
            ]
    
        p = 0.9
        sig = norm.ppf(p**(1/2))
    
        for i, (covar_2D, center_2D) in enumerate(zip(covars_2D, centers_2D)):
            v, w = linalg.eigh(covar_2D)
            print(v)
            v = 2. * sig * np.sqrt(v)
    
            u = w[0] / linalg.norm(w[0])
            if u[0] == 0:
                angle = np.pi / 2
            else:
                angle = np.arctan(u[1] / u[0])
    
            color = COLORS[i]
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(center_2D, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)
    
    return ax
        