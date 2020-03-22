import numpy as np


def cut_as_filter(fit_model, X, w, chosen_threshold_cut, y=None):
    print("Chosen Cut: ", chosen_threshold_cut)
    ypred = fit_model.predict_proba(X)
    cut = (ypred[:, 1] <= chosen_threshold_cut)

    background_like = X[cut]
    ypred_bg_like = ypred[cut]
    bg_like_weights = w[cut]

    flipped_cut = np.invert(cut)

    signal_like = X[flipped_cut]
    sg_like_weights = w[flipped_cut]

    assert all(i <= chosen_threshold_cut for i in ypred_bg_like[:, 1])
    assert len(signal_like) + len(background_like) == len(ypred)

    if y is not None:
        sg_like_y = y[flipped_cut]
        bg_like_y = y[cut]
        assert len(sg_like_y) + len(bg_like_y) == len(y)
        return signal_like, sg_like_weights, background_like, bg_like_weights, sg_like_y, bg_like_y
    else:
        return signal_like, sg_like_weights, background_like, bg_like_weights

