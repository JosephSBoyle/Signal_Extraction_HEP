sg_bar = [1,2,3,4,5,6,7,3]
bg_bar = [10,9,8,7,6,5,4,3]

cut = len(sg_bar)-1

for cut, (s_, b_) in enumerate(zip(sg_bar,bg_bar)):

    sig_in_cut = 0
    bac_in_cut = 0
    for subcut, (i, j) in enumerate(zip(sg_bar,bg_bar)):
        if subcut >= cut:
            sig_in_cut += sg_bar[subcut]
            bac_in_cut += bg_bar[subcut]
    print(sig_in_cut/bac_in_cut)


for i, s, b in reversed(list(enumerate(zip(sg_bar, bg_bar)))):
    print(i)

    for j, s, b in reversed(list(zip(sg_bar, bg_bar))):
        if i >= j:
            signal_in_cut += sg_bar
            backgr_in_cut += bg_bar
    print(signal_in_cut, backgr_in_cut)
