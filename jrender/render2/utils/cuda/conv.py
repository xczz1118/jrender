import jittor as jt

# x:[h,w,(d)] w:[h,w]
# x:[h,w,(d)]
def conv_for_image(x, w, overflow=0):
    flag_d=0
    if len(x.shape) == 2:
        x = x.unsqueeze(2)
        flag_d=1
    Kh, Kw= w.shape
    H, W, d = x.shape
    w = w.unsqueeze(2)
    xx = x.reindex([H, W, Kh, Kw, d], [
        'i0 + i2 - 1', 'i1 + i3 - 1', 'i4'
    ], overflow)
    ww = w.broadcast_var(xx)
    yy = xx * ww
    y = yy.sum([2, 3])
    if flag_d:
        y = y.squeeze(2)
    return y